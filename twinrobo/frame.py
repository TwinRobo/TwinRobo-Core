"""CameraFrame and TwinRobo tensor conventions.

Tensor conventions (the single source of truth for all TwinRobo modules):

- ``rgb``: ``torch.Tensor [B, 3, H, W]``, float, **linear** radiance-like values
  nominally in ``[0, 1]`` (not sRGB-encoded). Channel order R, G, B.
- ``depth``: ``torch.Tensor [B, 1, H, W]``, float, **metric depth in meters,
  positive**, measured along the optical axis (z-depth, not ray distance).
  Values that are non-finite (``inf``/``nan``, e.g. simulator background) or
  ``<= 0`` are **invalid**; consumers must mask or clamp them explicitly.
  Only cameras that output depth (``outputs.depth`` in their spec, e.g. the
  RealSense D400 series) expose it: reading ``frame.depth`` of any other camera
  raises `DepthUnavailableError`, unless the frame was made with
  ``force_depth=True`` (simulator ground truth).
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

from .exceptions import DepthUnavailableError


@dataclass(eq=False)
class CameraFrame:
    """One simulated camera observation.

    ``depth`` is readable only when ``has_depth`` is true: the camera outputs depth, or
    the frame was made with ``force_depth=True``. Otherwise reading it raises
    `DepthUnavailableError` (a real camera of that kind gives no depth).
    """

    rgb: Tensor

    rgb_ideal: Tensor | None = None
    rgb_optical: Tensor | None = None
    raw: Tensor | None = None

    depth: Tensor | None = field(default=None, repr=False)

    timestamp: float | None = None

    metadata: dict[str, Any] | None = field(default=None)

    has_depth: bool = True

    def to(self, device: str | torch.device) -> CameraFrame:
        """Return a copy with all tensors moved to ``device`` (explicit transfer only)."""

        def move(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device)

        return CameraFrame(
            rgb=self.rgb.to(device),
            rgb_ideal=move(self.rgb_ideal),
            rgb_optical=move(self.rgb_optical),
            raw=move(self.raw),
            depth=move(self._depth),
            timestamp=self.timestamp,
            metadata=None if self.metadata is None else dict(self.metadata),
            has_depth=self.has_depth,
        )


def _get_depth(self: CameraFrame) -> Tensor | None:
    if not self.has_depth:
        cam = (self.metadata or {}).get("camera_id") or "this camera"
        raise DepthUnavailableError(
            f"{cam} does not output depth (outputs.depth is false in its spec): a real "
            "camera of this kind measures none. Pass force_depth=True to get_frame() or "
            "process() to get the simulator's ground-truth depth anyway."
        )
    return self._depth


def _set_depth(self: CameraFrame, value: Tensor | None) -> None:
    self._depth = value


# A property over the dataclass field: the constructor still takes ``depth=``.
CameraFrame.depth = property(_get_depth, _set_depth, doc="Metric z-depth ``[B, 1, H, W]`` (m).")


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
