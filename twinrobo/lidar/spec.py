"""LidarSpec: a lidar's scan pattern, range and noise (``lidar.yaml``, schema 0.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import SpecError

PATTERNS = ("spinning", "raster")


@dataclass
class LidarSpec:
    """A lidar as the catalog describes it.

    ``pattern``: ``"spinning"`` (rings of beams swept through 360 degrees) or ``"raster"``
    (a fixed grid of beams, as flash and focal-plane-array lidars).
    ``elevations_deg``: beam elevations, top ring first. ``columns``: azimuth samples per
    frame (a full turn for spinning lidars, across ``fov_h_deg`` for raster ones).
    ``max_range_m`` is reached at ``max_range_reflectivity``; dimmer surfaces return less far.
    """

    id: str
    manufacturer: str | None
    product: str | None
    pattern: str
    elevations_deg: list[float]
    columns: int
    frame_hz: float
    fov_h_deg: float = 360.0
    min_range_m: float = 0.0
    max_range_m: float = 100.0
    max_range_reflectivity: float = 0.1
    range_sigma_m: float = 0.0
    fmcw: bool = False  # measures radial velocity per return (coherent / FMCW lidars)
    velocity_sigma_mps: float = 0.0
    status: str = "estimated"
    source_path: Path | None = field(default=None, repr=False)

    @property
    def rings(self) -> int:
        return len(self.elevations_deg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LidarSpec:
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()), source_path=path)

    @classmethod
    def from_dict(cls, d: dict[str, Any], source_path: Path | None = None) -> LidarSpec:
        if d.get("type") != "lidar":
            raise SpecError(f"{source_path or 'spec'}: not a lidar (type: {d.get('type')!r})")
        scan, rng = d.get("scan") or {}, d.get("range") or {}
        noise = d.get("noise") or {}
        pattern = scan.get("pattern")
        if pattern not in PATTERNS:
            raise SpecError(f"scan.pattern must be one of {PATTERNS}, got {pattern!r}")
        elev = scan.get("elevations_deg")
        if elev is None:  # uniform rings between two angles, top first
            v = scan.get("vertical_deg") or {}
            n = int(scan.get("channels") or 0)
            if n < 1 or "min" not in v or "max" not in v:
                raise SpecError("scan needs elevations_deg, or channels with vertical_deg min/max")
            lo, hi = float(v["min"]), float(v["max"])
            elev = [hi - (hi - lo) * i / max(1, n - 1) for i in range(n)]
        spec = cls(
            id=str(d["id"]),
            manufacturer=d.get("manufacturer"),
            product=d.get("product"),
            pattern=pattern,
            elevations_deg=[float(e) for e in elev],
            columns=int(scan["columns"]),
            frame_hz=float(scan.get("frame_hz", 10.0)),
            fov_h_deg=float(scan.get("fov_h_deg", 360.0 if pattern == "spinning" else 60.0)),
            min_range_m=float(rng.get("min_m", 0.0)),
            max_range_m=float(rng["max_m"]),
            max_range_reflectivity=float(rng.get("at_reflectivity", 0.1)),
            range_sigma_m=float(noise.get("range_sigma_m", 0.0)),
            fmcw=bool(d.get("fmcw", False)),
            velocity_sigma_mps=float(noise.get("velocity_sigma_mps", 0.0)),
            status=str((d.get("validation") or {}).get("status", "estimated")),
            source_path=source_path,
        )
        if spec.columns < 1 or spec.max_range_m <= spec.min_range_m:
            raise SpecError(f"{spec.id}: columns >= 1 and max_m > min_m are required")
        if not 0 < spec.max_range_reflectivity <= 1:
            raise SpecError(f"{spec.id}: range.at_reflectivity must be in (0, 1]")
        return spec
