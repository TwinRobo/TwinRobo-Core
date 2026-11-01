"""Demosaicing: full color from an RGGB mosaic (Malvar, He and Cutler, 2004)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# The paper's 5x5 kernels, in eighths: green at red or blue; red at green in a red row;
# red at green in a blue row; red at blue. Blue uses the same ones with the rows swapped.
_G_AT_RB = [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]
_ROW = [
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0],
]
_COL = [
    [0, 0, -1, 0, 0],
    [0, -1, 4, -1, 0],
    [0.5, 0, 5, 0, 0.5],
    [0, -1, 4, -1, 0],
    [0, 0, -1, 0, 0],
]
_DIAG = [
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0],
]


def demosaic_rggb(raw: Tensor) -> Tensor:
    """An RGGB mosaic ``[B, 1, H, W]`` -> ``[B, 3, H, W]``, each channel clipped at 0."""
    k = raw.new_tensor([_G_AT_RB, _ROW, _COL, _DIAG])[:, None] / 8.0
    g_rb, row, col, diag = F.conv2d(F.pad(raw, (2, 2, 2, 2), mode="reflect"), k).unbind(1)
    m = raw[:, 0]
    H, W = m.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(H, device=raw.device), torch.arange(W, device=raw.device), indexing="ij"
    )
    red = (y % 2 == 0) & (x % 2 == 0)
    blue = (y % 2 == 1) & (x % 2 == 1)
    g_red_row = (y % 2 == 0) & (x % 2 == 1)  # green with red left and right
    g_blue_row = (y % 2 == 1) & (x % 2 == 0)
    r = torch.where(red, m, torch.where(g_red_row, row, torch.where(g_blue_row, col, diag)))
    g = torch.where(red | blue, g_rb, m)
    b = torch.where(blue, m, torch.where(g_blue_row, row, torch.where(g_red_row, col, diag)))
    return torch.stack([r, g, b], 1).clamp_min(0)
