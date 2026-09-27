"""Sensor model interface."""

from __future__ import annotations

from typing import Any

from torch import Tensor


class SensorModel:
    """Converts optical irradiance on the sensor into a raw sensor image."""

    def capture(
        self, irradiance: Tensor, exposure: float, metadata: dict[str, Any] | None = None
    ) -> Tensor:
        raise NotImplementedError


class IdealSensor(SensorModel):
    """Noise-free, unquantized sensor: raw = irradiance * exposure, clipped to [0, 1]."""

    def capture(
        self, irradiance: Tensor, exposure: float = 1.0, metadata: dict[str, Any] | None = None
    ) -> Tensor:
        return (irradiance * exposure).clamp(0.0, 1.0)
