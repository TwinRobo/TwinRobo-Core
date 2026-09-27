"""Camera geometry shared by simulator adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels (pixel-index coordinates: pixel ``(0, 0)`` center is ``0``).

    For a CameraTwin these are the *paraxial* intrinsics of the lens at the spec
    resolution. They are the right pinhole for the simulator camera while
    distortion is not rendered.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2 / self.fy))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2 / self.fx))

    @classmethod
    def from_focal_mm(cls, width: int, height: int, focal_mm: float, pixel_pitch_mm: float):
        f = focal_mm / pixel_pitch_mm
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)
