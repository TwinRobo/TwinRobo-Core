"""Color filter array: the Bayer mosaic a color sensor records."""

from __future__ import annotations

from torch import Tensor


def mosaic_rggb(rgb: Tensor) -> Tensor:
    """``[B, 3, H, W]`` -> the ``[B, 1, H, W]`` RGGB mosaic."""
    m = rgb[:, 1:2].clone()
    m[..., 0::2, 0::2] = rgb[:, 0:1, 0::2, 0::2]
    m[..., 1::2, 1::2] = rgb[:, 2:3, 1::2, 1::2]
    return m
