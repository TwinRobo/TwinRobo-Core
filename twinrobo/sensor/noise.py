"""Sensor noise: libcamera's profile, read noise plus shot noise, scaled by analogue gain."""

from __future__ import annotations

import torch
from torch import Tensor


def noise_std(signal: Tensor, constant: float, slope: float, gain: float = 1.0) -> Tensor:
    """Noise standard deviation at ``signal`` (16-bit, above black), as in rpi/noise.cpp."""
    return gain**0.5 * (constant + slope * signal.clamp_min(0).sqrt())


def add_noise(signal: Tensor, std: Tensor, generator: torch.Generator) -> Tensor:
    """``signal`` plus Gaussian noise, drawn from ``generator`` (never the global state)."""
    return signal + std * torch.randn(
        signal.shape, generator=generator, device=signal.device, dtype=signal.dtype
    )
