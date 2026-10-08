"""Casting a lidar's beams into a scene (simulator independent).

The scene is a `TriangleMesh` (`twinrobo.optics.lensrender`): every drawn geom's triangles,
posed per frame, plus a per-owner albedo (linear luminance of its material). A return is:

- **range** along the beam (the hit distance, plus Gaussian noise of ``range_sigma_m``);
- **intensity**: ``albedo * cos(incidence)``, a reflectivity proxy. Simulators carry visible
  colours only, not the lidar's near-infrared reflectivity, so this is an approximation;
- dropped beyond ``max_range_m * sqrt(albedo cos / max_range_reflectivity)``: a surface
  dimmer than the datasheet's reference returns less far (the return falls with 1/r^2);
- **radial velocity** (FMCW lidars): the rate the beam's range to the hit point changes,
  positive when it recedes, from the scene and sensor poses at the next frame (kinematic
  replays set positions only, so velocities come from poses, not the simulator's qvel).

Noise is seeded by ``(spec.id, frame)``: the same frame scans the same.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .spec import LidarSpec


@dataclass
class LidarFrame:
    """One scan, as ``[rings, columns]`` images (ring 0 = top beam; NaN = no return)."""

    range: np.ndarray  # m
    intensity: np.ndarray  # reflectivity proxy, 0..1
    velocity: np.ndarray | None  # m/s along the beam, + = receding (FMCW lidars)
    elevation_deg: np.ndarray  # [rings]
    azimuth_deg: np.ndarray  # [columns], counterclockwise from +x (forward)
    time_s: np.ndarray  # [columns], when each column fires within the frame

    def valid(self) -> np.ndarray:
        return np.isfinite(self.range)

    def points(self) -> np.ndarray:
        """Returns as points ``[N, 3]`` in the sensor frame (x forward, y left, z up)."""
        el = np.radians(self.elevation_deg)[:, None]
        az = np.radians(self.azimuth_deg)[None, :]
        d = np.stack(
            np.broadcast_arrays(np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)),
            -1,
        )
        ok = self.valid()
        return (d * self.range[..., None])[ok]


class LidarTwin:
    def __init__(self, spec: LidarSpec, device: str | torch.device = "cpu"):
        self.spec = spec
        self.device = torch.device(device)
        s = spec
        if s.pattern == "spinning":
            az = np.arange(s.columns) * (360.0 / s.columns)
        else:  # raster: columns spread over the horizontal field, left to right
            az = np.linspace(s.fov_h_deg / 2, -s.fov_h_deg / 2, s.columns)
        self.azimuth_deg = az.astype(np.float64)
        self.elevation_deg = np.asarray(s.elevations_deg, np.float64)
        self.time_s = (
            np.arange(s.columns) / (s.columns * s.frame_hz)
            if s.pattern == "spinning"
            else np.zeros(s.columns)
        )
        el = torch.as_tensor(np.radians(self.elevation_deg), dtype=torch.float32)[:, None]
        a = torch.as_tensor(np.radians(self.azimuth_deg), dtype=torch.float32)[None, :]
        self.dirs = torch.stack(  # [rings, columns, 3], sensor frame
            torch.broadcast_tensors(
                torch.cos(el) * torch.cos(a), torch.cos(el) * torch.sin(a), torch.sin(el)
            ),
            -1,
        ).to(self.device)

    def _rng(self, frame: int) -> torch.Generator:
        seed = int.from_bytes(hashlib.sha1(f"{self.spec.id}|{frame}".encode()).digest()[:8])
        return torch.Generator(device=self.device).manual_seed(seed % (2**63))

    @torch.no_grad()
    def scan(
        self,
        mesh,
        albedo: Tensor,
        sensor_R: np.ndarray,
        sensor_p: np.ndarray,
        frame: int = 0,
        next_poses: tuple[Tensor, Tensor] | None = None,
        next_sensor: tuple[np.ndarray, np.ndarray] | None = None,
        dt: float | None = None,
    ) -> LidarFrame:
        """One scan of ``mesh`` (posed, `TriangleMesh.update` done) from the sensor pose.

        ``sensor_R`` (columns: sensor x, y, z axes in the world) and ``sensor_p``: the pose.
        ``albedo [owners]``: each owner's linear luminance. FMCW velocity needs the owners'
        poses (``next_poses``: R, t) and the sensor pose (``next_sensor``) ``dt`` later.
        """
        s, dev = self.spec, self.device
        R = torch.as_tensor(np.asarray(sensor_R), dtype=torch.float32, device=dev)
        p = torch.as_tensor(np.asarray(sensor_p), dtype=torch.float32, device=dev)
        d = (self.dirs.reshape(-1, 3) @ R.T).contiguous()  # world directions
        # rays start at the minimum range, so a lidar on a link does not see its own housing
        o = (p + d * s.min_range_m).contiguous()
        t, face = mesh.cast_faces(o, d, max_t=float(s.max_range_m - s.min_range_m))
        hit = face >= 0
        rng = torch.full_like(t, float("nan"))
        inten = torch.zeros_like(t)
        vel = torch.full_like(t, float("nan")) if s.fmcw else None
        idx = hit.nonzero(as_tuple=True)[0]
        if idx.numel():
            f = face[idx]
            owner = mesh.face_owner(f)
            n = mesh.face_normal(f)
            cos = (n * d[idx]).sum(1).abs().clamp(0, 1)
            refl = albedo.to(dev)[owner] * cos
            r = t[idx] + s.min_range_m
            reach = s.max_range_m * torch.sqrt(refl / s.max_range_reflectivity)
            ok = r <= reach
            g = self._rng(frame)
            if s.range_sigma_m > 0:
                r = r + torch.randn(r.shape, generator=g, device=dev) * s.range_sigma_m
            idx, r, refl, owner = idx[ok], r[ok], refl[ok], owner[ok]
            rng[idx] = r
            inten[idx] = refl.clamp(0, 1)
            if s.fmcw and next_poses is not None and next_sensor is not None and dt:
                hp = p + d[idx] * (t[idx] + s.min_range_m)[:, None]  # true (noise-free) hit point
                R0, t0 = mesh.R[owner], mesh.t[owner]
                local = torch.einsum("nji,nj->ni", R0, hp - t0)  # R0^T (hp - t0)
                R1, t1 = next_poses
                hp1 = torch.einsum("nij,nj->ni", R1.to(dev)[owner], local) + t1.to(dev)[owner]
                p1 = torch.as_tensor(np.asarray(next_sensor[1]), dtype=torch.float32, device=dev)
                v = ((hp1 - p1).norm(dim=1) - (hp - p).norm(dim=1)) / float(dt)
                if s.velocity_sigma_mps > 0:
                    v = v + torch.randn(v.shape, generator=g, device=dev) * s.velocity_sigma_mps
                vel[idx] = v
        shape = (s.rings, s.columns)
        return LidarFrame(
            range=rng.view(shape).cpu().numpy(),
            intensity=inten.view(shape).cpu().numpy(),
            velocity=None if vel is None else vel.view(shape).cpu().numpy(),
            elevation_deg=self.elevation_deg,
            azimuth_deg=self.azimuth_deg,
            time_s=self.time_s,
        )


def luminance(rgb) -> float:
    """Linear luminance of an sRGB colour (0..1), the lidar's reflectivity proxy."""
    c = np.clip(np.asarray(rgb, np.float64)[:3], 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(lin @ np.array([0.2126, 0.7152, 0.0722]))
