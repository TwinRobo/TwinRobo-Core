"""Sensor response: black level, saturation and quantization of the ADC."""

from __future__ import annotations

from torch import Tensor

#: Full scale of libcamera's 16-bit raw levels.
FULL_SCALE = 65535.0


def adc(signal: Tensor, black_level: float, bits: int) -> Tensor:
    """The level above black that a ``bits``-bit ADC reads for ``signal`` (16-bit, above black)."""
    step = 2.0 ** (16 - bits)
    code = ((signal + black_level) / step).round().clamp(0, 2**bits - 1)
    return (code * step - black_level).clamp(0, FULL_SCALE - black_level)
