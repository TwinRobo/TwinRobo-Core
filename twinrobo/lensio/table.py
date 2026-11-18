"""Prescription tables (``.csv``): a lens typed in from a patent or a datasheet.

One row per surface, object first and image last::

    # name: Cooke triplet
    # aperture: epd 10
    # fields: angle 0 14 20          (or x,y pairs: 0,0 0,14 10,10)
    # wavelengths: 0.4861 0.5876* 0.6563
    surface,radius,thickness,material,semi_diameter,conic,stop,a2,a4,a6
    OBJ,inf,inf,air,,,,,,
    1,21.48,2.0,N-SK16,6.5,,,,,
    ...
    IMG,inf,0,air,18.2,,,,,

``material`` is ``air``, a catalog glass (``N-BK7``, or ``N-BK7@SCHOTT`` to name the catalog),
or a model glass ``nd/vd``. ``radius`` ``inf`` is a plane. Columns ``a2`` .. ``a30`` are
aspheric terms on r^2 .. r^30 (mm); ``stop`` is ``y`` on the aperture stop. Lengths are mm.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from ..exceptions import LensFileError
from .prescription import AIR, FIELD_TYPES, Material, Prescription, Surface

COLUMNS = ["surface", "radius", "thickness", "material", "semi_diameter", "conic", "stop"]


def _num(v: str, default: float | None = None) -> float | None:
    v = v.strip()
    if not v:
        return default
    if v.lower() in ("inf", "infinity"):
        return math.inf
    return float(v)


def _material(v: str) -> Material:
    v = v.strip()
    if not v or v.lower() == "air":
        return AIR
    if "/" in v:
        nd, vd = v.split("/", 1)
        return Material.model(float(nd), float(vd))
    name, _, cat = v.partition("@")
    return Material(name=name, catalog=cat.upper() or None)


def parse_table(text: str, where: str = "<csv>") -> Prescription:
    p = Prescription(source_format="table", source_path=where)
    rows = []
    for n, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            key, _, val = line.lstrip()[1:].partition(":")
            key, val = key.strip().lower(), val.split()
            try:
                if key == "name":
                    p.name = line.split(":", 1)[1].strip()
                elif key == "aperture":
                    p.aperture_type = val[0].lower()
                    p.aperture_value = float(val[1]) if len(val) > 1 else None
                elif key == "fields":
                    if val[0] not in FIELD_TYPES:
                        raise LensFileError(f"{where}:{n}: field type must be one of {FIELD_TYPES}")
                    p.field_type = val[0]
                    p.fields = [
                        tuple(float(c) for c in v.split(",")) if "," in v else (0.0, float(v))
                        for v in val[1:]
                    ]
                    p.field_weights = [1.0] * len(p.fields)
                elif key == "wavelengths":
                    p.wavelengths_um = [float(v.rstrip("*")) for v in val]
                    p.wavelength_weights = [1.0] * len(val)
                    starred = [i for i, v in enumerate(val) if v.endswith("*")]
                    p.primary_wavelength = starred[0] if starred else len(val) // 2
            except (ValueError, IndexError) as e:
                raise LensFileError(f"{where}:{n}: cannot read [{line.strip()}]: {e}") from e
        elif line.strip():
            rows.append((n, line))
    reader = csv.DictReader(io.StringIO("\n".join(r for _, r in rows)))
    if reader.fieldnames is None or not set(COLUMNS[:4]) <= set(reader.fieldnames):
        raise LensFileError(f"{where}: the header needs columns {COLUMNS[:4]}")
    for (n, _), row in zip(rows[1:], reader, strict=True):
        at = f"{where}:{n}"
        try:
            coeffs = {}
            for key, val in row.items():
                if key and key.startswith("a") and key[1:].isdigit() and (val or "").strip():
                    coeffs[int(key[1:])] = float(val)
            s = Surface(
                radius=_num(row["radius"], math.inf),
                thickness=_num(row["thickness"], 0.0),
                material=_material(row["material"] or ""),
                semi_diameter=_num(row.get("semi_diameter") or ""),
                conic=_num(row.get("conic") or "", 0.0),
                stop=(row.get("stop") or "").strip().lower() in ("y", "yes", "1", "true"),
                coeffs={k: v for k, v in coeffs.items() if v},
                source=at,
                label=(row.get("surface") or "").strip(),
            )
        except (ValueError, TypeError) as e:
            raise LensFileError(f"{at}: cannot read row: {e}") from e
        if s.radius == 0:
            s.radius = math.inf
        s.kind = "asphere" if s.coeffs else "standard"
        p.surfaces.append(s)
    return p.check()


def read_table(path: str | Path) -> Prescription:
    path = Path(path)
    return parse_table(path.read_text(encoding="utf-8-sig"), str(path))


def format_table(p: Prescription) -> str:
    p.check()
    if any(s.kind == "paraxial" for s in p.surfaces):
        raise LensFileError("prescription tables do not hold paraxial surfaces")
    powers = sorted({k for s in p.surfaces for k in s.coeffs})
    out = io.StringIO()
    if p.title:
        out.write(f"# name: {p.title}\n")
    if p.aperture_type:
        val = "" if p.aperture_value is None else f" {p.aperture_value!r}"
        out.write(f"# aperture: {p.aperture_type}{val}\n")
    if any(x for x, _ in p.fields):  # x,y pairs; y alone when every field is on the y axis
        fields = " ".join(f"{x!r},{y!r}" for x, y in p.fields)
    else:
        fields = " ".join(repr(y) for _, y in p.fields)
    out.write(f"# fields: {p.field_type} {fields}\n")
    out.write(
        "# wavelengths: "
        + " ".join(
            repr(w) + ("*" if i == p.primary_wavelength else "")
            for i, w in enumerate(p.wavelengths_um)
        )
        + "\n"
    )
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS + [f"a{k}" for k in powers])
    for i, s in enumerate(p.surfaces):
        tag = "OBJ" if i == 0 else "IMG" if i == len(p.surfaces) - 1 else str(i)
        m = s.material
        mat = (
            "air"
            if m.is_air
            else f"{m.nd!r}/{m.vd!r}"
            if m.is_model
            else m.name + (f"@{m.catalog}" if m.catalog else "")
        )
        w.writerow(
            [
                tag,
                "inf" if math.isinf(s.radius) else repr(s.radius),
                "inf" if math.isinf(s.thickness) else repr(s.thickness),
                mat,
                "" if s.aperture_radius is None else repr(s.aperture_radius),
                repr(s.conic) if s.conic else "",
                "y" if s.stop else "",
            ]
            + [repr(s.coeffs[k]) if k in s.coeffs else "" for k in powers]
        )
    return out.getvalue()


def write_table(p: Prescription, path: str | Path) -> None:
    Path(path).write_text(format_table(p), encoding="utf-8")
