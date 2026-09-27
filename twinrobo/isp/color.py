"""Color encodings.

Optics and sensor models act on **linear** light. Simulators such as MuJoCo
output display-encoded 8-bit images, so adapters linearize before the optics
and re-encode afterwards. Treating MuJoCo's shading as sRGB-encoded radiance
is an approximation: its lighting is not radiometric.
"""

from __future__ import annotations

import torch
from torch import Tensor


def srgb_to_linear(x: Tensor) -> Tensor:
    """sRGB-encoded values in ``[0, 1]`` -> linear (IEC 61966-2-1)."""
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: Tensor) -> Tensor:
    """Linear values -> sRGB encoding in ``[0, 1]`` (clipped)."""
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)
