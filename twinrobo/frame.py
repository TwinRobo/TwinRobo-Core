"""CameraFrame and TwinRobo tensor conventions.

Tensor conventions (the single source of truth for all TwinRobo modules):

- ``rgb``: ``torch.Tensor [B, 3, H, W]``, float, **linear** radiance-like values
  nominally in ``[0, 1]`` (not sRGB-encoded). Channel order R, G, B.
- ``depth``: ``torch.Tensor [B, 1, H, W]``, float, **metric depth in meters,
  positive**, measured along the optical axis (z-depth, not ray distance).
  Values that are non-finite (``inf``/``nan``, e.g. simulator background) or
  ``<= 0`` are **invalid**; consumers must mask or clamp them explicitly.
- Image coordinates: row 0 is the top of the image, column 0 is the left.
- Device: tensors stay on the producing device (normally CUDA). TwinRobo never
  moves per-frame data to the CPU unless the caller explicitly asks for it.

Backend-specific units (e.g. DeepLens uses millimeters and negative object
z) are converted **only** inside the corresponding adapter
(`twinrobo.optics.deeplens`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class CameraFrame:
    """One simulated camera observation."""

    rgb: Tensor

    rgb_ideal: Tensor | None = None
    rgb_optical: Tensor | None = None
    raw: Tensor | None = None

    depth: Tensor | None = None

    timestamp: float | None = None

    metadata: dict[str, Any] | None = field(default=None)

    def to(self, device: str | torch.device) -> CameraFrame:
        """Return a copy with all tensors moved to ``device`` (explicit transfer only)."""

        def move(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device)

        return CameraFrame(
            rgb=self.rgb.to(device),
            rgb_ideal=move(self.rgb_ideal),
            rgb_optical=move(self.rgb_optical),
            raw=move(self.raw),
            depth=move(self.depth),
            timestamp=self.timestamp,
            metadata=None if self.metadata is None else dict(self.metadata),
        )


def check_rgb(rgb: Tensor) -> None:
    """Raise ``ValueError`` if ``rgb`` does not follow the ``[B, 3, H, W]`` convention."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
    if not rgb.is_floating_point():
        raise ValueError(f"rgb must be a floating point tensor, got {rgb.dtype}")


def check_depth(depth: Tensor, rgb: Tensor | None = None) -> None:
    """Raise ``ValueError`` if ``depth`` does not follow the ``[B, 1, H, W]`` convention."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must have shape [B, 1, H, W], got {tuple(depth.shape)}")
    if rgb is not None and (depth.shape[0] != rgb.shape[0] or depth.shape[-2:] != rgb.shape[-2:]):
        raise ValueError(
            f"depth {tuple(depth.shape)} does not match rgb {tuple(rgb.shape)} in batch/size"
        )


def valid_depth_mask(depth: Tensor) -> Tensor:
    """Boolean mask of valid depth pixels (finite and strictly positive)."""
    return torch.isfinite(depth) & (depth > 0)
