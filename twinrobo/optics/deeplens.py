"""DeepLens adapter — the only module in TwinRobo that touches DeepLens.

All DeepLens-specific interactions live here. The rest of
TwinRobo uses the `OpticsModel` interface and TwinRobo tensor conventions
(`twinrobo.frame`): depth in **positive meters**, rgb ``[B, 3, H, W]`` linear.

DeepLens conventions handled here:

- Object points for ``lens.psf``: ``x, y`` normalized field coordinates in
  ``[-1, 1]`` (``+y`` up), ``z`` = object depth in **negative millimeters**.
- ``lens.render_rgbd``: depth map in **positive millimeters**; it raises on
  non-finite or non-positive values, so invalid depth is sanitized first.
- ``lens.refocus``: focus distance in negative millimeters; ``+inf`` = infinity.
- Wavelengths in micrometers; default RGB order is ``lens.wvln_rgb`` (R, G, B).

DeepLens is imported lazily so ``import twinrobo`` stays lightweight.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..exceptions import OpticsBackendError
from ..frame import check_depth, check_rgb, valid_depth_mask
from ..geometry import CameraIntrinsics
from .base import OpticsModel
from .cache import hash_file
from .psf import PSFBank, field_grid

M_TO_MM = 1000.0
DEFAULT_FAR_M = 20.0  # also DeepLens' "approximately infinity" object depth


def deeplens_version() -> str:
    """Installed DeepLens version, including the pinned VCS commit when installed from Git."""
    import json
    from importlib import metadata

    try:
        dist = metadata.distribution("deeplens-core")
    except metadata.PackageNotFoundError:
        return "unknown"
    commit = ""
    direct = dist.read_text("direct_url.json")
    if direct:
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id", "")
    return f"{dist.version}+{commit}" if commit else dist.version


def _import_geolens():
    try:
        from deeplens import GeoLens
    except ImportError as e:  # pragma: no cover - exercised only without DeepLens
        raise OpticsBackendError(
            "DeepLens is not installed. Install TwinRobo with its pinned DeepLens "
            "dependency: pip install -e ."
        ) from e
    return GeoLens


class DeepLensOptics(OpticsModel):
    """Physical optics backed by a DeepLens geometric lens (``GeoLens``).

    Args:
        model_path: DeepLens lens file (``.json``, ``.zmx`` or ``.seq``).
        sensor_resolution: ``(width, height)`` in pixels. DeepLens keeps the
            sensor diagonal fixed and adapts the sensor size to this aspect ratio.
            When ``None``, the lens file's sensor is used.
        focus_distance_m: Object focus distance in meters (``None`` keeps the
            focus stored in the lens file; ``float('inf')`` focuses at infinity).
        device: Torch device. Defaults to CUDA when available.
        dtype: Torch dtype for the lens.
        psf_ks: PSF kernel size in pixels. Use an **odd** size. DeepLens centers
            even-size PSFs between pixels (on-axis centroid at ``ks/2 - 0.5``),
            while PSF convolution treats index ``ks // 2`` as zero shift, so even
            sizes shift the rendered image by about half a pixel.
    """

    def __init__(
        self,
        model_path: str | Path,
        sensor_resolution: tuple[int, int] | None = None,
        focus_distance_m: float | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        psf_ks: int = 65,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise OpticsBackendError(f"DeepLens lens file not found: {model_path}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        GeoLens = _import_geolens()
        self.model_path = model_path
        self.psf_ks = psf_ks
        # Requested focus; None = focus stored in the lens file (d_sensor as given).
        self.focus_distance_m = None if focus_distance_m is None else float(focus_distance_m)
        self._lens = GeoLens(filename=str(model_path), device=torch.device(device), dtype=dtype)
        if sensor_resolution is not None:
            self._lens.set_sensor_res(tuple(int(v) for v in sensor_resolution))
        if focus_distance_m is not None:
            if focus_distance_m <= 0:
                raise ValueError("focus_distance_m must be positive (meters) or inf")
            foc = (
                float("inf") if math.isinf(focus_distance_m) else -float(focus_distance_m) * M_TO_MM
            )
            self._lens.refocus(foc_dist=foc)

    # ------------------------------------------------------------------
    # Properties (TwinRobo units)
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return torch.device(self._lens.device)

    @property
    def sensor_resolution(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        w, h = self._lens.sensor_res
        return int(w), int(h)

    @property
    def pixel_pitch_um(self) -> float:
        return float(self._lens.pixel_size) * 1000.0

    @property
    def focal_length_mm(self) -> float:
        """Paraxial effective focal length computed by DeepLens (not a lens-file header)."""
        return float(self._lens.foclen)

    @property
    def f_number(self) -> float:
        """F-number computed by DeepLens from its paraxial pupil and EFL."""
        return float(self._lens.fnum)

    def intrinsics(self) -> CameraIntrinsics:
        """Paraxial pinhole intrinsics at the sensor resolution (no distortion)."""
        W, H = self.sensor_resolution
        return CameraIntrinsics.from_focal_mm(
            W, H, self.focal_length_mm, float(self._lens.pixel_size)
        )

    @property
    def wavelengths_rgb_um(self) -> list[float]:
        return [float(w) for w in self._lens.wvln_rgb]

    # ------------------------------------------------------------------
    # PSF generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_focus(self, focus_distance_m: float) -> None:
        """Refocus the lens to an object distance in meters (``inf`` = infinity)."""
        if not focus_distance_m > 0:
            raise ValueError("focus_distance_m must be positive (meters) or inf")
        self.focus_distance_m = float(focus_distance_m)
        foc = float("inf") if math.isinf(focus_distance_m) else -float(focus_distance_m) * M_TO_MM
        self._lens.refocus(foc_dist=foc)

    def generate_psf(
        self,
        field_positions: Tensor | Sequence[Sequence[float]],
        depths: Tensor | Sequence[float],
        wavelengths: Sequence[float] | None = None,
        ks: int | None = None,
        spp: int | None = None,
    ) -> Tensor:
        """Compute PSFs on a field × depth × wavelength grid.

        Args:
            field_positions: ``[N, 2]`` normalized field coordinates ``(x, y)`` in
                ``[-1, 1]`` (DeepLens convention, ``+y`` up).
            depths: ``[D]`` object depths in **meters** (positive).
            wavelengths: Wavelengths in micrometers. Defaults to the lens RGB
                wavelengths, giving an RGB PSF stack.
            ks: Kernel size in pixels (defaults to ``self.psf_ks``).
            spp: Rays per point source (DeepLens default when ``None``).

        Returns:
            ``[D, N, C, ks, ks]`` tensor on ``self.device``; each PSF sums to 1.
        """
        ks = self.psf_ks if ks is None else ks
        wavelengths = self.wavelengths_rgb_um if wavelengths is None else list(wavelengths)
        dev, dt = self.device, self._lens.dtype

        field = torch.as_tensor(field_positions, dtype=dt, device=dev).reshape(-1, 2)
        if bool((field.abs() > 1).any()):
            raise ValueError("field_positions must lie in [-1, 1]")
        depths_m = torch.as_tensor(depths, dtype=dt, device=dev).reshape(-1)
        if not bool((torch.isfinite(depths_m) & (depths_m > 0)).all()):
            raise ValueError("depths must be finite and positive (meters)")

        kwargs: dict[str, Any] = {"ks": ks}
        if spp is not None:
            kwargs["spp"] = spp

        out = []
        for d in depths_m:
            z = torch.full_like(field[:, :1], -float(d) * M_TO_MM)
            points = torch.cat([field, z], dim=-1)  # [N, 3]
            per_wvln = [
                self._lens.psf(points=points, wvln=w, **kwargs).reshape(-1, ks, ks)
                for w in wavelengths
            ]
            out.append(torch.stack(per_wvln, dim=1))  # [N, C, ks, ks]
        return torch.stack(out, dim=0)

    def sample_depths(self, near_m: float, far_m: float, num_depths: int) -> Tensor:
        """Depth layers in meters, far -> near, uniform in disparity.

        When the focus distance is known (``focus_distance_m``) and lies inside
        ``[near_m, far_m]``, it is an explicit layer, with the remaining layers
        split in proportion to the disparity range on each side. This avoids
        interpolating two defocused PSFs where the image should be sharpest.
        DeepLens' own focal-plane estimate (``calc_focal_plane``) is not used:
        it is Monte Carlo and unstable for lenses focused near infinity.
        """
        if not 0 < near_m < far_m:
            raise ValueError("need 0 < near_m < far_m")
        d_far, d_near = 1.0 / far_m, 1.0 / near_m
        focus = self.focus_distance_m
        if focus is not None and not math.isfinite(focus):
            focus = None  # infinity focus: the far layer is the sharpest one available
        if num_depths == 1:
            disp = torch.tensor([1.0 / focus if focus else d_far], dtype=torch.float64)
        elif focus is not None and near_m < focus < far_m and num_depths >= 3:
            d_f = 1.0 / focus
            n_far = min(
                max(1, round((num_depths - 1) * (d_f - d_far) / (d_near - d_far))), num_depths - 2
            )
            n_near = num_depths - 1 - n_far
            disp = torch.cat(
                [
                    torch.linspace(d_far, d_f, n_far + 1, dtype=torch.float64),
                    torch.linspace(d_f, d_near, n_near + 1, dtype=torch.float64)[1:],
                ]
            )
        else:
            disp = torch.linspace(d_far, d_near, num_depths, dtype=torch.float64)
        return (1.0 / disp).float()

    def bank_key(
        self,
        grid: tuple[int, int],
        near_m: float,
        far_m: float,
        num_depths: int,
        ks: int | None = None,
        spp: int | None = None,
    ) -> dict[str, Any]:
        """Content-based cache key for `build_psf_bank` (see `twinrobo.optics.cache`)."""
        return {
            "backend": "deeplens",
            "deeplens": deeplens_version(),
            "lens_sha256": hash_file(self.model_path),
            "sensor_resolution": list(self.sensor_resolution),
            # Requested focus, not d_sensor: DeepLens' refocus jitters d_sensor by ~1 um.
            "focus": self.focus_distance_m
            if self.focus_distance_m is not None
            else f"file:d_sensor={float(self._lens.d_sensor):.9f}",
            "wavelengths_um": self.wavelengths_rgb_um,
            "grid": list(grid),
            "near_m": float(near_m),
            "far_m": float(far_m),
            "num_depths": int(num_depths),
            "ks": int(self.psf_ks if ks is None else ks),
            "spp": spp,
        }

    def lens_rays_key(self, rays_per_pixel: int, seed: int = 0, overfill: float = 1.15) -> dict:
        """Content-based identity of `trace_lens_rays` output (lens, focus, sensor, sampling)."""
        return {
            "backend": "deeplens",
            "deeplens": deeplens_version(),
            "lens_sha256": hash_file(self.model_path),
            "sensor_resolution": list(self.sensor_resolution),
            "focus": self.focus_distance_m
            if self.focus_distance_m is not None
            else f"file:d_sensor={float(self._lens.d_sensor):.9f}",
            "wavelengths_um": self.wavelengths_rgb_um,
            "rays_per_pixel": int(rays_per_pixel),
            "seed": int(seed),
            "overfill": float(overfill),
            "candidates": 4,
            "version": 4,  # 4: rays aimed at each pixel's measured passing region
        }

    @torch.no_grad()
    def _relative_illumination(self, K, pr, pz, overfill_pr=True, grid=(48, 30), rays=256):
        """Relative illumination ``[C, H, W]`` (vignetting x cos^4), 1 on axis.

        Traced densely (``rays`` per node, stratified) on a coarse ``grid`` of the
        sensor and upsampled bilinearly: it is smooth, and per-pixel ray counts
        would turn it into noise.
        """
        import torch.nn.functional as F
        from deeplens.light import Ray

        from .lensrays import concentric_disk, stratified_samples

        L, dev = self._lens, self.device
        W, H = K.width, K.height
        gw, gh = min(grid[0], W), min(grid[1], H)
        pitch, zs = float(L.pixel_size), float(L.d_sensor)
        gu = torch.linspace(0, W - 1, gw, device=dev)
        gv = torch.linspace(0, H - 1, gh, device=dev)
        vv, uu = torch.meshgrid(gv, gu, indexing="ij")
        uu = torch.cat([uu.reshape(-1), torch.tensor([K.cx], device=dev)])  # + the optical axis
        vv = torch.cat([vv.reshape(-1), torch.tensor([K.cy], device=dev)])
        P = uu.numel()
        pu, pv = stratified_samples(rays, P, torch.Generator(device=dev).manual_seed(1), dev)
        dx, dy = concentric_disk(pu, pv)
        o = torch.stack(
            [
                (-(uu - K.cx) * pitch)[:, None].expand(P, rays),
                ((vv - K.cy) * pitch)[:, None].expand(P, rays),
                torch.full((P, rays), zs, device=dev),
            ],
            -1,
        ).reshape(-1, 3)
        t = torch.stack([dx * pr, dy * pr, torch.full_like(dx, pz)], -1).reshape(-1, 3)
        d = t - o
        cos4 = ((-d[:, 2]) / d.norm(dim=-1)) ** 4
        out = []
        for wv in self.wavelengths_rgb_um:
            ray = L.trace2obj(Ray(o.clone(), d.clone(), wv, device=dev))
            ok = (ray.is_valid > 0) & (ray.d[:, 2] < 0)
            e = torch.where(ok, cos4, torch.zeros_like(cos4)).reshape(P, rays).mean(1)
            e = e / e[-1].clamp_min(1e-9)
            img = e[:-1].reshape(1, 1, gh, gw)
            out.append(F.interpolate(img, size=(H, W), mode="bilinear", align_corners=True)[0, 0])
        return torch.stack(out).float()

    @torch.no_grad()
    def _ray_aim(self, K, pz, pr, grid=(17, 11), rays=4096, margin=1.1):
        """Per-pixel aiming disk ``(cx, cy, r)`` ``[H, W]`` each (mm) on the plane ``z = pz``.

        Rays are aimed at the region of that plane through which a pixel's light
        actually passes the lens. It is measured on a coarse field ``grid``: from each
        node, ``rays`` rays over the whole disk of radius ``pr``; the passing ones'
        centroid and their farthest distance from it (times ``margin``) set the disk,
        interpolated bilinearly to every pixel. Where DeepLens finds the exit pupil the
        disk is about the pupil's; where it cannot and falls back to the last surface
        (whose aperture can be many times the beam), this keeps most traced rays inside
        the lens instead of blocked by it. Sampling stays uniform over a disk that
        covers every passing ray, so the result is unbiased either way.
        """
        import torch.nn.functional as F
        from deeplens.light import Ray

        from .lensrays import concentric_disk, stratified_samples

        L, dev = self._lens, self.device
        W, H = K.width, K.height
        gw, gh = min(grid[0], W), min(grid[1], H)
        pitch, zs = float(L.pixel_size), float(L.d_sensor)
        gu = torch.linspace(0, W - 1, gw, device=dev)
        gv = torch.linspace(0, H - 1, gh, device=dev)
        vv, uu = torch.meshgrid(gv, gu, indexing="ij")
        P = uu.numel()
        pu, pv = stratified_samples(rays, P, torch.Generator(device=dev).manual_seed(2), dev)
        dx, dy = concentric_disk(pu, pv)
        tx, ty = dx * pr, dy * pr  # [P, rays]
        o = torch.stack(
            [
                (-(uu.reshape(-1) - K.cx) * pitch)[:, None].expand(P, rays),
                ((vv.reshape(-1) - K.cy) * pitch)[:, None].expand(P, rays),
                torch.full((P, rays), zs, device=dev),
            ],
            -1,
        ).reshape(-1, 3)
        t = torch.stack([tx, ty, torch.full_like(tx, pz)], -1).reshape(-1, 3)
        ok = torch.zeros(P, rays, dtype=torch.bool, device=dev)
        for wv in self.wavelengths_rgb_um:  # a ray passing in any channel counts
            ray = L.trace2obj(Ray(o.clone(), t - o, wv, device=dev))
            ok |= ((ray.is_valid > 0) & (ray.d[:, 2] < 0)).view(P, rays)
        cnt = ok.sum(1)
        w = ok.float()
        cx = (tx * w).sum(1) / cnt.clamp_min(1)
        cy = (ty * w).sum(1) / cnt.clamp_min(1)
        dist = torch.where(ok, torch.hypot(tx - cx[:, None], ty - cy[:, None]), 0.0)
        r = dist.amax(1) * margin + 2.0 * pr / math.sqrt(rays)  # + about one sample spacing
        few = cnt < 16  # (nearly) fully vignetted node: keep the whole disk
        cx, cy = torch.where(few, 0.0, cx), torch.where(few, 0.0, cy)
        r = torch.where(few, torch.full_like(r, pr), r.clamp(max=pr))
        maps = torch.stack([cx, cy, r]).reshape(1, 3, gh, gw)
        maps = F.interpolate(maps, size=(H, W), mode="bilinear", align_corners=True)[0]
        return maps[0], maps[1], maps[2]

    @torch.no_grad()
    def trace_lens_rays(
        self,
        rays_per_pixel: int = 8,
        seed: int = 0,
        overfill: float = 1.15,
        chunk_rays: int = 4_000_000,
        progress=None,
        candidates: int = 4,
    ):
        """Trace every sensor pixel through the lens into object space (see `LensRays`).

        Each pixel gets ``rays_per_pixel`` rays per RGB wavelength: a jittered
        point inside the pixel aimed at a stratified point of the disk on the exit
        pupil plane through which that pixel's light passes (`_ray_aim`; the
        pupil enlarged by ``overfill`` bounds it), so the stop blocks few of them.
        Rays are traced backward (sensor -> object) with DeepLens' own surface tracer.

        ``candidates``: each pixel traces ``candidates x rays_per_pixel`` rays and
        keeps ``rays_per_pixel`` that pass the lens, in random order. Strongly
        vignetted pixels then still get their full ray count (less noise); their
        brightness comes from the separately traced relative illumination.
        """
        from deeplens.light import Ray

        from .lensrays import LensRays, concentric_disk, stratified_samples

        L = self._lens
        K = self.intrinsics()
        W, H = K.width, K.height
        n = int(rays_per_pixel)
        dev = self.device
        pitch = float(L.pixel_size)
        zs = float(L.d_sensor)
        pz, pr = L.get_exit_pupil()
        pz, pr = float(pz), float(pr) * overfill
        wvlns = self.wavelengths_rgb_um
        g = torch.Generator(device=dev).manual_seed(seed)
        aim_x, aim_y, aim_r = (a.reshape(-1) for a in self._ray_aim(K, pz, pr))

        origin = torch.empty(len(wvlns), H, W, n, 2, dtype=torch.float16, device=dev)
        tan_res = torch.empty_like(origin)
        weight = torch.empty(len(wvlns), H, W, n, dtype=torch.float16, device=dev)
        m = max(1, int(candidates))
        gi = min(range(len(wvlns)), key=lambda i: abs(wvlns[i] - 0.55))

        def trace(su0, sv0, count):
            """Trace ``count`` rays for each pixel center ``(su0, sv0)`` [P]; per wavelength:
            ``(ok, ox, oy, tx, ty, w)`` each ``[P, count]``."""
            P = su0.numel()
            ju, jv = stratified_samples(count, P, g, dev)  # sub-pixel jitter
            pu, pv = stratified_samples(count, P, g, dev)  # pupil sample
            rot = torch.rand(P, 1, generator=g, device=dev) * (2 * math.pi)  # per-pixel rotation
            dx, dy = concentric_disk(pu, pv)
            c, s_ = torch.cos(rot), torch.sin(rot)
            dx, dy = dx * c - dy * s_, dx * s_ + dy * c
            pix = (sv0.long() * W + su0.long())[:, None]  # pixel -> its aiming disk
            dx = aim_x[pix] + dx * aim_r[pix]
            dy = aim_y[pix] + dy * aim_r[pix]
            su = su0[:, None] - 0.5 + ju  # sample position in pixel coordinates
            sv = sv0[:, None] - 0.5 + jv
            # Upright image -> sensor: the lens inverts the image (object +x, +y land at -x, -y).
            xs = -(su - K.cx) * pitch
            ys = (sv - K.cy) * pitch
            o = torch.stack([xs, ys, torch.full_like(xs, zs)], -1).reshape(-1, 3)
            t = torch.stack([dx, dy, torch.full_like(dx, pz)], -1).reshape(-1, 3)
            d = t - o
            cos4 = ((-d[:, 2]) / d.norm(dim=-1)) ** 4  # cos^4 of the sensor-side angle
            out = []
            for wv in wvlns:
                ray = L.trace2obj(Ray(o.clone(), d.clone(), wv, device=dev))
                ro, rd = ray.o, ray.d
                ok = (ray.is_valid > 0) & (rd[:, 2] < 0)
                dz = torch.where(ok, -rd[:, 2], torch.ones_like(rd[:, 2]))
                tx, ty = rd[:, 0] / dz, rd[:, 1] / dz
                ox = (ro[:, 0] + tx * ro[:, 2]) / 1000.0  # origin on the z = 0 plane, m
                oy = (ro[:, 1] + ty * ro[:, 2]) / 1000.0
                w = torch.where(ok, cos4, torch.zeros_like(cos4))
                out.append([x.view(P, count) for x in (ok, ox, oy, tx, ty, w)])
            return out

        def keep(res, count):
            """Keep ``count`` rays per pixel, unblocked ones first (in random order)."""
            ok = res[0]
            P, nc = ok.shape
            key = torch.rand(P, nc, generator=g, device=dev) + (~ok).float() * 2
            pick = key.argsort(dim=1)[:, :count]
            return [x.gather(1, pick) for x in res]

        rows_per_chunk = max(1, chunk_rays // (W * n))
        u = torch.arange(W, device=dev, dtype=torch.float32)
        r_max = 0.0
        for r0 in range(0, H, rows_per_chunk):
            r1 = min(H, r0 + rows_per_chunk)
            h = r1 - r0
            v = torch.arange(r0, r1, device=dev, dtype=torch.float32)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            su0, sv0 = uu.reshape(-1), vv.reshape(-1)
            res = trace(su0, sv0, n)
            # Strongly vignetted pixels (under half their rays pass, in green): retrace them
            # with more candidates so they keep n rays; brightness comes from `illumination`.
            short = (res[gi][0].sum(1) * 2 < n).nonzero(as_tuple=True)[0]
            if m > 1 and short.numel():
                more = trace(su0[short], sv0[short], n * m)
                for ci in range(len(wvlns)):
                    sel = keep(more[ci], n)
                    for x, y in zip(res[ci], sel, strict=True):
                        x[short] = y
            pix_x = ((su0 - K.cx) / K.fx)[:, None].expand(-1, n)
            pix_y = (-(sv0 - K.cy) / K.fy)[:, None].expand(-1, n)
            for ci in range(len(wvlns)):
                ok, ox, oy, tx, ty, w = res[ci]
                ox, oy = torch.where(ok, ox, 0), torch.where(ok, oy, 0)
                tx, ty = torch.where(ok, tx, pix_x), torch.where(ok, ty, pix_y)
                origin[ci, r0:r1] = torch.stack([ox, oy], -1).reshape(h, W, n, 2).half()
                tan_res[ci, r0:r1] = (
                    torch.stack([tx - pix_x, ty - pix_y], -1).reshape(h, W, n, 2).half()
                )
                weight[ci, r0:r1] = w.reshape(h, W, n).half()
                if bool(ok.any()):
                    r_max = max(r_max, float(torch.sqrt(ox[ok] ** 2 + oy[ok] ** 2).max()))
            if progress is not None:
                progress(r1 / H)

        illumination = self._relative_illumination(K, pr, pz, overfill_pr=True)

        # Chief rays (green, exit pupil center) for depth and rectification.
        vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), u, indexing="ij")
        o = torch.stack([-(uu - K.cx) * pitch, (vv - K.cy) * pitch, torch.full_like(uu, zs)], -1)
        o = o.reshape(-1, 3)
        d = torch.tensor([0.0, 0.0, pz], device=dev) - o
        ray = L.trace2obj(Ray(o, d, wvlns[gi], device=dev))
        dz = (-ray.d[:, 2]).clamp_min(1e-9)
        chief = torch.stack([ray.d[:, 0] / dz, ray.d[:, 1] / dz], -1).reshape(H, W, 2)
        return LensRays(
            intrinsics=K,
            wavelengths_um=list(wvlns),
            origin=origin,
            tan_res=tan_res,
            weight=weight,
            chief_tan=chief.float(),
            illumination=illumination,
            pupil_radius_m=r_max,
            metadata=self.lens_rays_key(n, seed, overfill),
        )

    @torch.no_grad()
    def build_psf_bank(
        self,
        grid: tuple[int, int] = (17, 11),
        near_m: float = 0.3,
        far_m: float = DEFAULT_FAR_M,
        num_depths: int = 16,
        ks: int | None = None,
        spp: int | None = None,
    ) -> PSFBank:
        """Precompute the runtime `PSFBank` on a ``(Gw, Gh)`` field grid x depth layers."""
        ks = self.psf_ks if ks is None else ks
        depths = self.sample_depths(near_m, far_m, num_depths)
        nodes = field_grid(grid, device=self.device)  # [Gh, Gw, 2]
        psf = self.generate_psf(nodes.reshape(-1, 2), depths, ks=ks, spp=spp)
        gw, gh = grid
        psfs = psf.reshape(len(depths), gh, gw, psf.shape[2], ks, ks).float().contiguous()
        return PSFBank(
            psfs=psfs,
            depths_m=depths.to(psfs.device),
            sensor_resolution=self.sensor_resolution,
            wavelengths_um=self.wavelengths_rgb_um,
            metadata=self.bank_key(grid, near_m, far_m, num_depths, ks, spp),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @torch.no_grad()
    def render_reference(
        self,
        rgb: Tensor,
        depth: Tensor,
        psf_grid: tuple[int, int] = (17, 11),
        near_m: float = 0.3,
        far_m: float = DEFAULT_FAR_M,
        num_layers: int = 16,
        depths_m: Tensor | Sequence[float] | None = None,
    ) -> Tensor:
        """High-fidelity **offline** reference render, built from DeepLens primitives.

        The full DeepLens PSF map is computed at every depth layer
        (``psf_map_rgb``), then DeepLens' block-wise, depth-interpolated PSF-map
        convolution is applied (``conv_psf_map_depth_interp``, disparity
        interpolation). It has no occlusion handling and uses one PSF per grid
        block. PSFs are recomputed on every call, so it is far too slow for
        per-frame simulation. Use it for validation and reference data only. The runtime renderer is
        `twinrobo.optics.dense_depth.DenseDepthRenderer`.

        Invalid depth (non-finite or ``<= 0``, e.g. simulator background) is
        replaced by ``far_m``. Depth outside the layer range is clamped.

        Args:
            rgb: ``[B, 3, H, W]`` linear RGB; ``(W, H)`` must equal
                ``self.sensor_resolution``.
            depth: ``[B, 1, H, W]`` depth in meters.
            psf_grid: ``(grid_w, grid_h)`` PSF map blocks.
            near_m: Nearest depth layer (m), via `sample_depths` unless ``depths_m``
                is given. Use the same values as the `PSFBank` being compared.
            far_m: Farthest depth layer (m).
            num_layers: Number of depth layers.

        Returns:
            ``[B, 3, H, W]`` optically rendered image.
        """
        from deeplens.imgsim import conv_psf_map_depth_interp

        check_rgb(rgb)
        check_depth(depth, rgb)
        W, H = self.sensor_resolution
        if rgb.shape[-1] != W or rgb.shape[-2] != H:
            raise ValueError(
                f"Image size {rgb.shape[-1]}x{rgb.shape[-2]} must match the lens sensor "
                f"resolution {W}x{H}; pass sensor_resolution when constructing DeepLensOptics."
            )
        if depths_m is None:
            depths_m = self.sample_depths(near_m, far_m, num_layers)
        depths_m = torch.as_tensor(depths_m, dtype=self._lens.dtype, device=self.device)

        rgb = rgb.to(self.device, self._lens.dtype)
        depth = depth.to(self.device, self._lens.dtype)
        depth = torch.where(valid_depth_mask(depth), depth, torch.full_like(depth, far_m))

        psf_map = torch.stack(
            [
                self._lens.psf_map_rgb(
                    grid=tuple(psf_grid), ks=self.psf_ks, depth=-float(d) * M_TO_MM
                )
                for d in depths_m
            ],
            dim=2,
        )  # [grid_h, grid_w, K, 3, ks, ks]
        return conv_psf_map_depth_interp(
            rgb, -depth * M_TO_MM, psf_map, -depths_m * M_TO_MM, interp_mode="disparity"
        )

    def render(self, rgb: Tensor, depth: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        """`OpticsModel.render`: the slow offline reference renderer.

        For runtime use, build a `PSFBank` with `build_psf_bank` and render with
        `twinrobo.optics.dense_depth.DenseDepthRenderer`.
        """
        return self.render_reference(rgb, depth)
