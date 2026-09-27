"""Parametric lens distortion (OpenCV / Brown-Conrady) and its rendering warp.

Geometry and blur are modelled separately. Blur comes from the PSF bank
(PSFs recentered on the chief ray, so it carries no geometric shift); geometry
comes from the spec's intrinsics plus this distortion model, which is what a
calibration (e.g. a vendor factory calibration file) measures.

Rendering: the simulator renders a wider pinhole *source* image with the same
focal length at the center; `DistortionWarp` resamples it onto the distorted
sensor (bilinear for color, nearest for depth, so no false mid-depths appear at
edges). The source covers the undistorted field of every sensor pixel.

Coordinates follow OpenCV: normalized ``x = (u - cx) / fx``, ``y = (v - cy) / fy``
with pixel ``(0, 0)`` at the center of the top-left pixel and ``+y`` down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..geometry import CameraIntrinsics

MODELS = ("opencv",)


@dataclass(frozen=True)
class RadialDistortion:
    """OpenCV distortion ``(k1, k2, p1, p2, k3)``: normalized undistorted -> distorted."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RadialDistortion | None:
        """From ``calibration.distortion``; ``None`` unless ``model`` is parametric."""
        if not d or d.get("model") not in MODELS:
            return None
        return cls(**{k: float(d.get(k) or 0.0) for k in ("k1", "k2", "p1", "p2", "k3")})

    @property
    def is_identity(self) -> bool:
        return not any((self.k1, self.k2, self.p1, self.p2, self.k3))

    def distort(self, x, y):
        """Undistorted normalized coordinates -> distorted (numpy arrays or tensors)."""
        r2 = x * x + y * y
        radial = 1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    def undistort(self, xd, yd, iters: int = 30):
        """Distorted -> undistorted by fixed-point iteration (OpenCV's scheme)."""
        x, y = xd, yd
        for _ in range(iters):
            r2 = x * x + y * y
            radial = 1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            dx = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            dy = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
            x, y = (xd - dx) / radial, (yd - dy) / radial
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {"model": "opencv", **{k: getattr(self, k) for k in ("k1", "k2", "p1", "p2", "k3")}}


def _border(intr: CameraIntrinsics, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Normalized distorted coordinates along the sensor's outer pixel border."""
    W, H = intr.width, intr.height
    u = np.linspace(-0.5, W - 0.5, n)
    v = np.linspace(-0.5, H - 0.5, n)
    us = np.concatenate([u, u, np.full(n, -0.5), np.full(n, W - 0.5)])
    vs = np.concatenate([np.full(n, -0.5), np.full(n, H - 0.5), v, v])
    return (us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy


def field_of_view(intr: CameraIntrinsics, dist: RadialDistortion | None) -> dict[str, float]:
    """Real horizontal / vertical / diagonal FoV in degrees (through the principal point)."""
    und = dist.undistort if dist is not None else (lambda a, b: (a, b))

    def span(ax, ay, bx, by):
        (x0, y0), (x1, y1) = und(np.array(ax), np.array(ay)), und(np.array(bx), np.array(by))
        v0, v1 = np.array([x0, y0, 1.0]), np.array([x1, y1, 1.0])
        c = v0 @ v1 / np.linalg.norm(v0) / np.linalg.norm(v1)
        return math.degrees(math.acos(float(np.clip(c, -1, 1))))

    W, H = intr.width, intr.height
    nx = lambda u: (u - intr.cx) / intr.fx  # noqa: E731
    ny = lambda v: (v - intr.cy) / intr.fy  # noqa: E731
    return {
        "h": span(nx(-0.5), 0.0, nx(W - 0.5), 0.0),
        "v": span(0.0, ny(-0.5), 0.0, ny(H - 0.5)),
        "d": span(nx(-0.5), ny(-0.5), nx(W - 0.5), ny(H - 0.5)),
    }


class DistortionWarp:
    """Resample a wide pinhole source render onto a distorted sensor.

    Args:
        intrinsics: The camera's (sensor) intrinsics.
        distortion: Its distortion model.
        margin_px: Extra source pixels around the needed field (bilinear support).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        distortion: RadialDistortion,
        margin_px: int = 2,
    ):
        self.intrinsics, self.distortion = intrinsics, distortion
        # Source pinhole: square pixels at the sensor's focal length (fy: simulators set a
        # vertical FoV), centered, covering the undistorted field of the whole sensor.
        f = float(intrinsics.fy)
        ex, ey = _border(intrinsics)
        bx, by = distortion.undistort(ex, ey)
        rx, ry = distortion.distort(bx, by)
        if (
            not (np.isfinite(bx).all() and np.isfinite(by).all())
            or max(np.abs(rx - ex).max(), np.abs(ry - ey).max()) > 1e-6
        ):
            # e.g. strong barrel whose radial curve peaks before the sensor corner
            raise ValueError(f"distortion {distortion} is not invertible over the sensor")
        half_w = math.ceil(f * float(np.abs(bx).max()) + 0.5) + margin_px
        half_h = math.ceil(f * float(np.abs(by).max()) + 0.5) + margin_px
        Ws, Hs = 2 * half_w, 2 * half_h
        self.source = CameraIntrinsics(Ws, Hs, f, f, (Ws - 1) / 2, (Hs - 1) / 2)
        self._grids: dict[tuple[str, bool], Tensor] = {}

    def _grid(self, device: torch.device, distorted: bool) -> Tensor:
        """``grid_sample`` grid [1, H, W, 2]: sensor pixel -> source position."""
        key = (str(device), distorted)
        if key not in self._grids:
            intr, src = self.intrinsics, self.source
            v, u = torch.meshgrid(
                torch.arange(intr.height, dtype=torch.float64),
                torch.arange(intr.width, dtype=torch.float64),
                indexing="ij",
            )
            x, y = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
            if distorted:
                x, y = self.distortion.undistort(x, y)
            us, vs = src.fx * x + src.cx, src.fy * y + src.cy
            g = torch.stack([2 * us / (src.width - 1) - 1, 2 * vs / (src.height - 1) - 1], -1)
            self._grids[key] = g[None].to(device=device, dtype=torch.float32)
        return self._grids[key]

    def _check(self, img: Tensor) -> None:
        if tuple(img.shape[-2:]) != (self.source.height, self.source.width):
            raise ValueError(
                f"source must be {self.source.width}x{self.source.height}, got "
                f"{img.shape[-1]}x{img.shape[-2]}"
            )

    def _sample(self, img: Tensor, distorted: bool, mode: str) -> Tensor:
        self._check(img)
        g = self._grid(img.device, distorted).expand(img.shape[0], -1, -1, -1)
        return F.grid_sample(
            img, g.to(img.dtype), mode=mode, padding_mode="border", align_corners=True
        )

    def __call__(self, rgb: Tensor, depth: Tensor) -> tuple[Tensor, Tensor]:
        """Distorted sensor images: rgb ``[B,3,H,W]`` (bilinear), depth ``[B,1,H,W]`` (nearest)."""
        return self._sample(rgb, True, "bilinear"), self._sample(depth, True, "nearest")

    def pinhole(self, img: Tensor, mode: str = "bilinear") -> Tensor:
        """The undistorted pinhole view with the sensor's intrinsics (for comparison)."""
        return self._sample(img, False, mode)

    def rectify(self, img: Tensor, mode: str = "bilinear") -> Tensor:
        """Undistort a *sensor* image onto the pinhole with the sensor's intrinsics.

        This is what vendor SDKs deliver as "rectified": the lens blur stays, the
        geometry becomes pinhole. Pixels whose ray falls outside the sensor (corners,
        for barrel distortion) repeat the border.
        """
        intr = self.intrinsics
        if tuple(img.shape[-2:]) != (intr.height, intr.width):
            raise ValueError(f"sensor image must be {intr.width}x{intr.height}")
        key = (str(img.device), "rectify")
        if key not in self._grids:
            v, u = torch.meshgrid(
                torch.arange(intr.height, dtype=torch.float64),
                torch.arange(intr.width, dtype=torch.float64),
                indexing="ij",
            )
            xd, yd = self.distortion.distort((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy)
            us, vs = intr.fx * xd + intr.cx, intr.fy * yd + intr.cy
            g = torch.stack([2 * us / (intr.width - 1) - 1, 2 * vs / (intr.height - 1) - 1], -1)
            self._grids[key] = g[None].to(device=img.device, dtype=torch.float32)
        g = self._grids[key].expand(img.shape[0], -1, -1, -1).to(img.dtype)
        return F.grid_sample(img, g, mode=mode, padding_mode="border", align_corners=True)
