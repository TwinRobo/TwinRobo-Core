"""Object-space rays of a real lens: sensor pixel -> lens -> scene (simulator independent).

`LensRays` holds, for every sensor pixel, ``n`` rays per wavelength that leave the
sensor, pass through the lens, and enter the scene. They are traced once by an
optics backend (`DeepLensOptics.trace_lens_rays`) and then used by the
ray-based renderers (`twinrobo.mujoco.lensrender`). They depend only on the
lens, focus and sensor, never on the scene, so they are cached.

Conventions. The camera frame is MuJoCo's and Isaac's (USD): x right, y up, and
the camera looks along -z. A ray is ``o + s * (tx, ty, -1)``:

- ``origin`` ``(ox, oy)`` in **meters** on the camera's ``z = 0`` plane (the
  lens front vertex plane; for the bundled lenses the stop is there).
- ``tan`` ``(tx, ty)``: direction slopes. They are stored as a residual on top of
  the pixel's paraxial pinhole direction (``intrinsics``), which keeps fp16 exact
  to well below a pixel.
- ``weight``: 0 for rays the lens blocks, else ``cos^4`` of the sensor-side angle.
  A pixel's color is the weight-normalized mean of its rays' radiance times
  ``illumination``: the pixel's relative illumination (vignetting and ``cos^4``
  falloff, 1 on axis), traced densely on a coarse grid. Splitting brightness from
  blur keeps a few rays per pixel from turning vignetting into noise.

Pixel ``(u, v)`` (row 0 = top) is the image a camera outputs, i.e. upright; the
optical inversion on the sensor is handled when tracing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..geometry import CameraIntrinsics


@dataclass
class LensRays:
    intrinsics: CameraIntrinsics  # paraxial pinhole of the sensor (residual base, rectified output)
    wavelengths_um: list[float]
    origin: Tensor  # [C, H, W, n, 2] fp16, meters
    tan_res: Tensor  # [C, H, W, n, 2] fp16, slope residual to the paraxial pinhole
    weight: Tensor  # [C, H, W, n] fp16
    chief_tan: Tensor  # [H, W, 2] fp32, chief ray (green, pupil center) slopes
    illumination: Tensor  # [C, H, W] fp32, relative illumination (1 on axis)
    pupil_radius_m: float  # radius covering every ray origin
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.origin.shape[3]

    @property
    def channels(self) -> int:
        return self.origin.shape[0]

    @property
    def resolution(self) -> tuple[int, int]:
        return self.intrinsics.width, self.intrinsics.height

    @property
    def device(self) -> torch.device:
        return self.origin.device

    def nbytes(self) -> int:
        ts = (self.origin, self.tan_res, self.weight, self.chief_tan, self.illumination)
        return sum(t.numel() * t.element_size() for t in ts)

    def pinhole_tan(self, dtype=torch.float32) -> Tensor:
        """Paraxial pinhole slopes of the pixel centers, ``[H, W, 2]``."""
        K = self.intrinsics
        u = torch.arange(K.width, device=self.device, dtype=dtype)
        v = torch.arange(K.height, device=self.device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return torch.stack([(uu - K.cx) / K.fx, -(vv - K.cy) / K.fy], dim=-1)

    def rays(self, channel: int, rows: slice = slice(None)) -> tuple[Tensor, Tensor, Tensor]:
        """``(origin [h, W, n, 2], tan [h, W, n, 2], weight [h, W, n])`` as fp32 for ``rows``."""
        base = self.pinhole_tan()[rows][:, :, None, :]
        return (
            self.origin[channel, rows].float(),
            self.tan_res[channel, rows].float() + base,
            self.weight[channel, rows].float(),
        )

    def max_tan(self) -> tuple[float, float]:
        """Largest ``|tx|``, ``|ty|`` of any unblocked ray (the field a source view must cover)."""
        tx = ty = 0.0
        base = self.pinhole_tan()
        for c in range(self.channels):
            t = (self.tan_res[c].float() + base[:, :, None, :]).abs()
            ok = self.weight[c] > 0
            tx = max(tx, float(t[..., 0][ok].max()))
            ty = max(ty, float(t[..., 1][ok].max()))
        return tx, ty

    # -- rectification (undistortion) from the chief-ray map --------------------------------------
    def rectify_grid(self, iterations: int = 6) -> Tensor:
        """``grid_sample`` grid ``[1, H, W, 2]`` undistorting a lens image to the paraxial pinhole.

        For each rectified pixel (paraxial direction ``t*``), finds the sensor
        position whose chief ray has direction ``t*`` by Newton iteration on the
        chief-ray map. This is what a stereo SDK's rectification does, computed from
        the traced lens instead of a fitted distortion model.
        """
        cached = self.metadata.get("_rectify_grid")
        if cached is not None:
            return cached
        K = self.intrinsics
        W, H = K.width, K.height
        target = self.pinhole_tan()  # [H, W, 2]
        chief = self.chief_tan.permute(2, 0, 1)[None]  # [1, 2, H, W]
        # d(tan)/d(pixel) of the chief map (smooth), per pixel; rows (tx, ty), columns (u, v)
        ct = self.chief_tan
        J = torch.stack(
            [
                torch.stack(
                    [torch.gradient(ct[..., 0], dim=1)[0], torch.gradient(ct[..., 0], dim=0)[0]], -1
                ),
                torch.stack(
                    [torch.gradient(ct[..., 1], dim=1)[0], torch.gradient(ct[..., 1], dim=0)[0]], -1
                ),
            ],
            -2,
        )  # [H, W, 2, 2]
        jinv = torch.linalg.inv(J).reshape(H, W, 4).permute(2, 0, 1)[None]  # [1, 4, H, W]
        vv, uu = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        p = torch.stack([uu, vv], -1)  # initial guess: the paraxial pixel itself
        tgt = target.permute(2, 0, 1)  # [2, H, W]

        def to_grid(q):
            return torch.stack([(q[..., 0] + 0.5) / W * 2 - 1, (q[..., 1] + 0.5) / H * 2 - 1], -1)[
                None
            ]

        def sample(img, q):
            return F.grid_sample(
                img, to_grid(q), mode="bilinear", align_corners=False, padding_mode="border"
            )[0]

        for _ in range(iterations):
            err = sample(chief, p) - tgt  # [2, H, W]
            ji = sample(jinv, p).reshape(2, 2, H, W)
            p = p - torch.einsum("ijhw,jhw->hwi", ji, err)
        grid = to_grid(p)
        self.metadata["_rectify_grid"] = grid
        return grid


def concentric_disk(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Shirley-Chiu map of the unit square to the unit disk (area preserving, low distortion)."""
    a, b = 2 * u - 1, 2 * v - 1
    r = torch.where(a.abs() > b.abs(), a, b)
    phi = torch.where(
        a.abs() > b.abs(),
        (math.pi / 4) * b / a.where(a != 0, torch.ones_like(a)),
        (math.pi / 2) - (math.pi / 4) * a / b.where(b != 0, torch.ones_like(b)),
    )
    return r * torch.cos(phi), r * torch.sin(phi)


def stratified_samples(
    n: int, count: int, generator: torch.Generator, device
) -> tuple[Tensor, Tensor]:
    """``count x n`` jittered stratified samples of the unit square (``n`` split into a grid)."""
    gx = int(math.ceil(math.sqrt(n)))
    gy = int(math.ceil(n / gx))
    idx = torch.arange(n, device=device)
    sx, sy = (idx % gx).float(), (idx // gx).float()
    ju = torch.rand(count, n, generator=generator, device=device)
    jv = torch.rand(count, n, generator=generator, device=device)
    return (sx + ju) / gx, (sy + jv) / gy
