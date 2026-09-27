"""Optics backend interface."""

from __future__ import annotations

from typing import Any

from torch import Tensor


class OpticsModel:
    """Maps an ideal scene observation to the optical image on the sensor.

    ``rgb`` / ``depth`` follow the conventions in `twinrobo.frame`
    (``[B, 3, H, W]`` linear, ``[B, 1, H, W]`` meters). Returns ``[B, 3, H, W]``.
    """

    def render(self, rgb: Tensor, depth: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        raise NotImplementedError


class IdentityOptics(OpticsModel):
    """Level 0 (ideal pinhole) optics: returns the input unchanged."""

    def render(self, rgb: Tensor, depth: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        return rgb
