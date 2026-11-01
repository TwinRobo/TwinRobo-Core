"""Sensor model interface."""

from __future__ import annotations

import itertools
from typing import Any

from torch import Tensor

_streams = itertools.count()


def new_stream(camera: Any = None) -> str:
    """A unique name for one adapter's frames: cameras of a twin can share a name."""
    return f"{camera}#{next(_streams)}"


class SensorModel:
    """Converts optical irradiance on the sensor into a raw sensor image."""

    #: The raw image this sensor gives: ``"rgb"`` or ``"rggb"`` (see `twinrobo.isp.ISPModel`).
    output_format = "rgb"

    def reset(self, stream: Any = None) -> None:
        """Forget what one stream of frames, or all of them, left in the sensor."""

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
