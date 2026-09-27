"""Synthetic RGB-D scenes for tests, examples and benchmarks (no simulator needed)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def texture(H: int, W: int, device=None, seed: int = 0) -> Tensor:
    """Multi-scale random texture with sharp edges, ``[1, 3, H, W]`` in ``[0, 1]``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.zeros(1, 3, H, W)
    for cell, amp in ((64, 0.5), (16, 0.3), (4, 0.2)):
        coarse = torch.rand(1, 3, max(1, H // cell), max(1, W // cell), generator=g)
        img += amp * F.interpolate(coarse, size=(H, W), mode="nearest")
    return img.clamp(0, 1).to(device)


def slanted_plane(H: int, W: int, near_m: float, far_m: float, device=None, seed: int = 0):
    """Textured plane; disparity varies linearly from ``near_m`` (top) to ``far_m`` (bottom)."""
    t = torch.linspace(0, 1, H)[:, None].expand(H, W)
    depth = 1.0 / ((1 - t) / near_m + t / far_m)
    return texture(H, W, device, seed), depth[None, None].to(device)


def fronto_plane(H: int, W: int, depth_m: float, device=None, seed: int = 0):
    """Textured plane at a single depth."""
    return texture(H, W, device, seed), torch.full((1, 1, H, W), float(depth_m), device=device)


def occluder_scene(
    H: int, W: int, near_m: float, far_m: float, device=None, seed: int = 0, background_inf=True
):
    """Textured background plus a textured rectangle in front of it.

    With ``background_inf`` the background depth is ``inf``, as for a simulator sky.
    """
    rgb = texture(H, W, device, seed)
    fg = texture(H, W, device, seed + 1)
    depth = torch.full((1, 1, H, W), float("inf") if background_inf else float(far_m))
    r0, r1, c0, c1 = H // 4, 3 * H // 4, W // 3, 2 * W // 3
    rgb[..., r0:r1, c0:c1] = fg[..., r0:r1, c0:c1]
    depth[..., r0:r1, c0:c1] = near_m
    return rgb, depth.to(device)
