"""The neutral lens prescription that every lens file format is read into and written from.

A `Prescription` is a rotationally symmetric, centered, sequential lens: an object surface,
the lens' surfaces in order, and an image surface. Lengths are millimeters whatever the file
used, curvatures are given as radii (``inf`` for a plane), and aspheric coefficients are
unnormalized: coefficient ``a[p]`` multiplies ``r**p`` with ``r`` in millimeters.

Anything a file says that changes the optics and cannot be held here makes the reader raise
`LensFileError`: TwinRobo never renders a lens that differs from its file without saying so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..exceptions import LensFileError

FIELD_TYPES = ("angle", "object_height", "paraxial_image_height", "real_image_height")
APERTURE_TYPES = ("epd", "fnum", "float")
#: Lens units a file may use, in millimeters.
UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "m": 1000.0}


@dataclass(frozen=True)
class Material:
    """The medium after a surface.

    ``name`` is a catalog glass, ``catalog`` the catalog the file names for it (if any).
    A model glass has no name and is defined by ``nd`` and ``vd`` alone. Air has neither.
    ``nd``/``vd`` beside a catalog name are what the file recorded, kept for reference only:
    for catalog glasses they can be stale, so a catalog glass is always resolved by name.
    """

    name: str | None = None
    catalog: str | None = None
    nd: float | None = None
    vd: float | None = None

    @property
    def is_air(self) -> bool:
        return self.name is None and self.nd is None

    @property
    def is_model(self) -> bool:
        return self.name is None and self.nd is not None

    @classmethod
    def model(cls, nd: float, vd: float) -> Material:
        return cls(nd=float(nd), vd=float(vd))

    def describe(self) -> str:
        if self.is_air:
            return "air"
        if self.is_model:
            return f"model glass nd={self.nd:g} vd={self.vd:g}"
        return f"{self.name}" + (f" ({self.catalog})" if self.catalog else "")


AIR = Material()


@dataclass
class Surface:
    """One surface and the gap after it.

    ``kind``: ``"standard"`` (sphere, conic or plane), ``"asphere"`` (a conic plus even
    polynomial terms in ``coeffs``) or ``"paraxial"`` (an ideal thin lens of
    ``focal_length`` mm, air on both sides).

    ``semi_diameter`` is the surface's extent as the file gives it. ``clear_aperture`` is a
    radius that blocks rays, where the file sets one explicitly.
    """

    kind: str = "standard"
    radius: float = math.inf
    conic: float = 0.0
    coeffs: dict[int, float] = field(default_factory=dict)
    thickness: float = 0.0
    material: Material = AIR
    semi_diameter: float | None = None
    clear_aperture: float | None = None
    stop: bool = False
    focal_length: float | None = None
    label: str = ""
    source: str = ""  # "file:line" where the surface starts, for error messages

    @property
    def curvature(self) -> float:
        return 0.0 if math.isinf(self.radius) else 1.0 / self.radius

    def sag(self, r: float) -> float:
        """Surface height (mm) at radial distance ``r`` (mm), from the vertex."""
        c = self.curvature
        z = c * r * r / (1.0 + math.sqrt(max(0.0, 1.0 - (1.0 + self.conic) * c * c * r * r)))
        for p, a in self.coeffs.items():
            z += a * r**p
        return z

    @property
    def aperture_radius(self) -> float | None:
        """The radius rays are traced through: the explicit clear aperture, else the extent."""
        return self.clear_aperture if self.clear_aperture is not None else self.semi_diameter


@dataclass
class Prescription:
    """A sequential lens: ``surfaces[0]`` is the object, ``surfaces[-1]`` the image."""

    name: str = ""
    surfaces: list[Surface] = field(default_factory=list)
    aperture_type: str | None = None  # one of APERTURE_TYPES
    aperture_value: float | None = None  # EPD (a diameter, mm) or F-number
    field_type: str = "angle"
    fields: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    field_weights: list[float] = field(default_factory=lambda: [1.0])
    wavelengths_um: list[float] = field(default_factory=lambda: [0.5876])
    wavelength_weights: list[float] = field(default_factory=lambda: [1.0])
    primary_wavelength: int = 0  # index into wavelengths_um
    glass_catalogs: list[str] = field(default_factory=list)  # search order the file names
    # Glass catalogs (.agf bytes by file name) that came with the lens, as in a .zar archive.
    packed_catalogs: dict[str, bytes] = field(default_factory=dict, repr=False)
    # Glasses saved with a DeepLens JSON lens: catalog -> glass name -> DeepLens AGF entry.
    packed_glasses: dict[str, dict[str, dict]] = field(default_factory=dict, repr=False)
    notes: list[str] = field(default_factory=list)  # approximations made while reading
    source_format: str = ""
    source_path: str = ""

    # ------------------------------------------------------------------
    @property
    def lens_surfaces(self) -> list[Surface]:
        """The surfaces between object and image."""
        return self.surfaces[1:-1]

    @property
    def object_distance(self) -> float:
        """Object to first surface (mm); ``inf`` for an object at infinity."""
        return self.surfaces[0].thickness

    @property
    def stop_index(self) -> int | None:
        stops = [i for i, s in enumerate(self.surfaces) if s.stop]
        return stops[0] if stops else None

    @property
    def image_semi_diameter(self) -> float | None:
        return self.surfaces[-1].semi_diameter

    @property
    def primary_wavelength_um(self) -> float:
        return self.wavelengths_um[self.primary_wavelength]

    @property
    def title(self) -> str:
        """The name on one line, for file headers (a line break would start a new record)."""
        return " ".join(self.name.split())

    def copy(self) -> Prescription:
        return replace(
            self,
            surfaces=[replace(s, coeffs=dict(s.coeffs)) for s in self.surfaces],
            fields=list(self.fields),
            field_weights=list(self.field_weights),
            wavelengths_um=list(self.wavelengths_um),
            wavelength_weights=list(self.wavelength_weights),
            glass_catalogs=list(self.glass_catalogs),
            packed_catalogs=dict(self.packed_catalogs),
            packed_glasses={c: dict(g) for c, g in self.packed_glasses.items()},
            notes=list(self.notes),
        )

    # ------------------------------------------------------------------
    def check(self) -> Prescription:
        """Raise `LensFileError` unless this is a lens TwinRobo can trace as written."""
        where = self.source_path or self.name or "lens"
        if len(self.surfaces) < 3:
            raise LensFileError(f"{where}: needs an object, at least one lens surface and an image")
        if not self.surfaces[0].material.is_air:
            raise LensFileError(f"{where}: object space must be air")
        img = self.surfaces[-1]
        if not math.isinf(img.radius) or img.coeffs or img.kind != "standard":
            raise LensFileError(f"{where}: the image surface must be flat (a flat sensor)")
        if not self.surfaces[-2].material.is_air:
            raise LensFileError(f"{where}: image space must be air")
        stops = [i for i, s in enumerate(self.surfaces) if s.stop]
        if len(stops) > 1:
            raise LensFileError(f"{where}: more than one aperture stop: surfaces {stops}")
        if stops and stops[0] in (0, len(self.surfaces) - 1):
            raise LensFileError(f"{where}: the stop cannot be the object or image surface")
        if not self.object_distance > 0:
            raise LensFileError(f"{where}: the object distance must be positive")
        for i, s in enumerate(self.lens_surfaces, start=1):
            at = f"{where}: surface {i}" + (f" ({s.source})" if s.source else "")
            if s.kind not in ("standard", "asphere", "paraxial"):
                raise LensFileError(f"{at}: unknown surface kind {s.kind!r}")
            if not math.isfinite(s.thickness):
                raise LensFileError(f"{at}: thickness must be finite")
            if s.kind == "paraxial":
                if s.focal_length is None or s.focal_length == 0 or math.isinf(s.focal_length):
                    raise LensFileError(f"{at}: a paraxial surface needs a finite focal length")
                if not (s.material.is_air and self.surfaces[i - 1].material.is_air):
                    raise LensFileError(f"{at}: a paraxial surface must have air on both sides")
            if any(p % 2 or p < 2 for p in s.coeffs):
                raise LensFileError(f"{at}: aspheric terms must be even powers r^2, r^4, ...")
            if s.radius == 0:
                raise LensFileError(f"{at}: radius 0 (use inf for a plane)")
        if self.aperture_type is not None and self.aperture_type not in APERTURE_TYPES:
            raise LensFileError(f"{where}: unknown aperture type {self.aperture_type!r}")
        if self.aperture_type in ("epd", "fnum") and not (self.aperture_value or 0) > 0:
            raise LensFileError(f"{where}: {self.aperture_type} must be positive")
        if self.field_type not in FIELD_TYPES:
            raise LensFileError(f"{where}: unknown field type {self.field_type!r}")
        if not self.wavelengths_um or not 0 <= self.primary_wavelength < len(self.wavelengths_um):
            raise LensFileError(f"{where}: no primary wavelength")
        return self

    def summary(self) -> str:
        """A readable prescription table."""
        lines = [f"{self.name or '(unnamed lens)'}  [{self.source_format}]"]
        ap = (
            f"{self.aperture_type} {self.aperture_value:g}"
            if self.aperture_value is not None
            else self.aperture_type or "stop size"
        )
        lines.append(
            f"aperture: {ap}; fields ({self.field_type}): "
            + ", ".join(f"{y:g}" for _, y in self.fields)
            + "; wavelengths (um): "
            + ", ".join(
                f"{w:g}" + ("*" if i == self.primary_wavelength else "")
                for i, w in enumerate(self.wavelengths_um)
            )
        )
        lines.append(f"{'#':>3} {'radius':>12} {'thickness':>11} {'material':<26} {'semi-d':>8}")
        for i, s in enumerate(self.surfaces):
            tag = "OBJ" if i == 0 else "IMG" if i == len(self.surfaces) - 1 else str(i)
            if s.stop:
                tag = "STO"
            rad = "inf" if math.isinf(s.radius) else f"{s.radius:.6g}"
            if s.kind == "paraxial":
                rad = f"f={s.focal_length:g}"
            th = "inf" if math.isinf(s.thickness) else f"{s.thickness:.6g}"
            sd = "" if s.semi_diameter is None else f"{s.semi_diameter:.4g}"
            extra = []
            if s.conic:
                extra.append(f"k={s.conic:g}")
            extra += [f"a{p}={a:.4g}" for p, a in sorted(s.coeffs.items()) if a]
            lines.append(
                f"{tag:>3} {rad:>12} {th:>11} {s.material.describe():<26} {sd:>8}"
                + (("  " + " ".join(extra)) if extra else "")
            )
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def scale_surface(s: Surface, mm_per_unit: float) -> Surface:
    """``s`` with lengths given in a lens unit converted to millimeters."""
    k = float(mm_per_unit)
    if k == 1.0:
        return s
    return replace(
        s,
        radius=s.radius * k,
        thickness=s.thickness * k,
        coeffs={p: a * k ** (1 - p) for p, a in s.coeffs.items()},
        semi_diameter=None if s.semi_diameter is None else s.semi_diameter * k,
        clear_aperture=None if s.clear_aperture is None else s.clear_aperture * k,
        focal_length=None if s.focal_length is None else s.focal_length * k,
    )
