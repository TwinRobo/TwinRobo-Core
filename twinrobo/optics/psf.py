"""Precomputed PSF bank: PSF(depth, field_y, field_x, channel) on a regular grid.

A `PSFBank` is the runtime optical representation consumed by
`twinrobo.optics.dense_depth.DenseDepthRenderer`. It is
produced offline by an optics backend (e.g. `DeepLensOptics.build_psf_bank`)
and does not depend on DeepLens itself.

Conventions:

- ``psfs``: ``[K, Gh, Gw, C, ks, ks]``. Each PSF sums to 1 and is centered at
  index ``ks // 2`` (DeepLens convention). Row 0 of a PSF is the top of the image.
- ``depths_m``: ``[K]`` object depths in meters, sorted **far -> near**
  (increasing disparity).
- Field nodes span the sensor edge to edge (``G >= 2``): node ``(i, j)`` sits at
  normalized field ``x = -1 + 2 j / (Gw - 1)``, ``y = 1 - 2 i / (Gh - 1)`` (``x``
  left -> right, ``y`` top -> bottom, row 0 = top), i.e. at pixel edge
  coordinate ``(j W / (Gw - 1), i H / (Gh - 1))``. So every pixel is
  *interpolated* between nodes, never extrapolated from the outermost node,
  which matters because PSFs change fastest at the field edge. ``G = 1`` means
  one on-axis node. See `field_grid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

BANK_FORMAT_VERSION = 1


def field_grid(grid: tuple[int, int], device=None, dtype=torch.float32) -> Tensor:
    """Normalized field positions of the bank nodes, shape ``[Gh, Gw, 2]`` as ``(x, y)``."""
    gw, gh = grid

    def axis(n):
        if n == 1:
            return torch.zeros(1, device=device, dtype=dtype)
        return torch.linspace(-1, 1, n, device=device, dtype=dtype)

    x = axis(gw)
    y = -axis(gh)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


@dataclass
class PSFBank:
    psfs: Tensor  # [K, Gh, Gw, C, ks, ks]
    depths_m: Tensor  # [K], far -> near
    sensor_resolution: tuple[int, int]  # (W, H) pixels
    wavelengths_um: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.psfs.ndim != 6 or self.psfs.shape[-1] != self.psfs.shape[-2]:
            raise ValueError(f"psfs must be [K, Gh, Gw, C, ks, ks], got {tuple(self.psfs.shape)}")
        if self.depths_m.ndim != 1 or self.depths_m.numel() != self.psfs.shape[0]:
            raise ValueError("depths_m must be [K] matching psfs.shape[0]")
        disp = 1.0 / self.depths_m
        if self.depths_m.numel() > 1 and not bool((disp[1:] > disp[:-1]).all()):
            raise ValueError("depths_m must be sorted far -> near (strictly increasing disparity)")
        self.sensor_resolution = (int(self.sensor_resolution[0]), int(self.sensor_resolution[1]))

    # ------------------------------------------------------------------
    @property
    def num_depths(self) -> int:
        return self.psfs.shape[0]

    @property
    def grid(self) -> tuple[int, int]:
        """``(Gw, Gh)``."""
        return self.psfs.shape[2], self.psfs.shape[1]

    @property
    def channels(self) -> int:
        return self.psfs.shape[3]

    @property
    def ks(self) -> int:
        return self.psfs.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.psfs.device

    def to(self, device=None, dtype=None) -> PSFBank:
        return PSFBank(
            psfs=self.psfs.to(device=device, dtype=dtype),
            depths_m=self.depths_m.to(device=device, dtype=dtype),
            sensor_resolution=self.sensor_resolution,
            wavelengths_um=self.wavelengths_um,
            metadata=dict(self.metadata),
        )

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format_version": BANK_FORMAT_VERSION,
                "psfs": self.psfs.detach().cpu(),
                "depths_m": self.depths_m.detach().cpu(),
                "sensor_resolution": list(self.sensor_resolution),
                "wavelengths_um": self.wavelengths_um,
                "metadata": self.metadata,
            },
            tmp,
        )
        tmp.replace(path)  # atomic: never leave a half-written bank in the cache

    @classmethod
    def load(cls, path: str | Path, device=None) -> PSFBank:
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format_version") != BANK_FORMAT_VERSION:
            raise ValueError(
                f"PSF bank format {data.get('format_version')} != {BANK_FORMAT_VERSION}: {path}"
            )
        bank = cls(
            psfs=data["psfs"],
            depths_m=data["depths_m"],
            sensor_resolution=tuple(data["sensor_resolution"]),
            wavelengths_um=data["wavelengths_um"],
            metadata=data["metadata"],
        )
        return bank.to(device) if device is not None else bank
