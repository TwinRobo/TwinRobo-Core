"""Tone and detail: the tuning's tone curve, denoise and sharpening."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

#: Rec. 601 luma, as the ISP's YCbCr stage uses.
LUMA_601 = (0.299, 0.587, 0.114)


def piecewise_linear(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """``numpy.interp`` on tensors: ``fp`` at ``x``, held constant past the ends."""
    xp, fp = xp.contiguous(), fp.contiguous()
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    w = ((x - xp[i - 1]) / (xp[i] - xp[i - 1])).clamp(0, 1)
    return fp[i - 1] + (fp[i] - fp[i - 1]) * w


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur of ``[B, C, H, W]``, edges extended."""
    r = round(4 * sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = (-(t**2) / (2 * sigma**2)).exp()
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(
        F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c
    )
    return F.conv2d(
        F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c
    )


def bilateral(x: Tensor, sigma_value: float, sigma_space: float, radius: int = 2) -> Tensor:
    """Bilateral filter of each channel of ``[B, C, H, W]`` over a disk of ``radius`` pixels."""
    pad = F.pad(x, (radius,) * 4, mode="reflect")
    total, norm = torch.zeros_like(x), torch.zeros_like(x)
    H, W = x.shape[-2:]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r2 = dx * dx + dy * dy
            if r2 > radius * radius:
                continue
            y = radius + dy
            q = pad[..., y : y + H, radius + dx : radius + dx + W]
            w = (-r2 / (2 * sigma_space**2) - (q - x) ** 2 / (2 * sigma_value**2)).exp()
            total += w * q
            norm += w
    return total / norm


def rgb_to_ycc(rgb: Tensor) -> Tensor:
    """RGB -> luma, red difference, blue difference (Rec. 601, full range, centered at 0)."""
    r, g, b = rgb.unbind(1)
    y = LUMA_601[0] * r + LUMA_601[1] * g + LUMA_601[2] * b
    return torch.stack([y, 0.713 * (r - y), 0.564 * (b - y)], 1)


def ycc_to_rgb(ycc: Tensor) -> Tensor:
    y, cr, cb = ycc.unbind(1)
    r, b = y + cr / 0.713, y + cb / 0.564
    g = (y - LUMA_601[0] * r - LUMA_601[2] * b) / LUMA_601[1]
    return torch.stack([r, g, b], 1)
