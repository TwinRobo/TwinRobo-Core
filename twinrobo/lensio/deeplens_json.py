"""DeepLens' native lens JSON: the format TwinRobo renders from.

`to_deeplens` turns a prescription into a DeepLens lens dict; `from_deeplens` reads a DeepLens
JSON file (the catalog's ``lens.json``) back into a prescription, so any lens TwinRobo renders
can be exported to the other formats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..exceptions import LensFileError
from .materials import GlassCatalogs, bundled_catalog, resolve_all
from .paraxial import Paraxial, paraxial
from .prescription import AIR, Material, Prescription, Surface

#: DeepLens' defaults (``deeplens.config``): rendering wavelengths (um), red, green, blue, and
#: the design wavelength of a lens file that names none.
WVLN_RGB = [0.656, 0.587, 0.486]
DEFAULT_WAVE = 0.587


def indices(p: Prescription, materials: list[str]) -> list[float]:
    """Refractive index after each surface at the primary wavelength."""
    from deeplens.material.materials import Material as DLMaterial

    w = float(p.primary_wavelength_um)
    return [1.0 if m == "air" else float(DLMaterial(m).refractive_index(w)) for m in materials]


def to_deeplens(
    p: Prescription,
    catalogs: GlassCatalogs | None = None,
    allow_model_glass: bool = False,
    sensor_size_mm: tuple[float, float] | None = None,
    sensor_res: tuple[int, int] | None = None,
) -> tuple[dict, Paraxial]:
    """A DeepLens lens dict for ``p``, and the paraxial data used to complete it.

    Surfaces without a semi-diameter get the paraxial marginal plus chief ray height (10%
    margin); a sensor not given is sized to the image surface's semi-diameter, else to the
    paraxial image height of the edge field. Both are recorded in ``p.notes``.
    """
    p = p.copy().check()
    catalogs = catalogs if catalogs is not None else GlassCatalogs()
    mats = resolve_all(p, catalogs, allow_model_glass)
    par = paraxial(p, indices(p, mats))
    surfs = []
    for i, s in enumerate(p.lens_surfaces, start=1):
        r = s.aperture_radius
        if i == par.stop_index and p.aperture_type in ("epd", "fnum"):
            # Zemax and CODE V define the EPD paraxially; DeepLens takes the pupil from the
            # stop's size. The stop gets the size that gives the file's paraxial EPD.
            r = par.stop_radius
        elif r is None or r <= 0:
            r = 1.1 * (par.marginal[i - 1] + par.chief[i - 1])
            p.notes.append(f"surface {i}: no semi-diameter; using {r:.4g} mm (paraxial rays)")
        mat2 = mats[i]
        d = {"r": float(r), "d_next": float(s.thickness), "mat2": mat2}
        flat = math.isinf(s.radius)
        if s.kind == "paraxial":
            d.update(type="ThinLens", f=float(s.focal_length))
        elif (
            i == par.stop_index and flat and not s.coeffs and mats[i - 1] == "air" and mat2 == "air"
        ):
            d.update(type="Aperture")
            del d["mat2"]
        elif s.coeffs or s.conic:
            powers = sorted(s.coeffs)
            use_ai2 = 2 in powers
            top = max(powers) if powers else 4
            ai = [s.coeffs.get(k, 0.0) for k in range(2 if use_ai2 else 4, top + 1, 2)]
            d.update(
                type="Aspheric", roc=0.0 if flat else float(s.radius), k=float(s.conic),
                ai=[float(a) for a in ai], use_ai2=use_ai2,
            )  # fmt: skip
        elif flat:
            d.update(type="Plane")
        else:
            d.update(type="Spheric", roc=float(s.radius))
        if i == par.stop_index:
            d["is_aperture"] = True
        surfs.append(d)

    data: dict = {
        "info": f"{p.name or 'lens'} (imported from {Path(p.source_path).name or p.source_format})",
        "primary_wvln": float(p.primary_wavelength_um),
        "surfaces": surfs,
    }
    if math.isfinite(p.object_distance):
        data["obj_depth"] = -float(p.object_distance)
    if p.aperture_type in ("epd", "fnum"):
        data["enpd"] = float(par.epd)
    if sensor_size_mm is not None:
        if sensor_res is None:
            raise LensFileError("sensor_size_mm needs sensor_res")
        (w, h), (W, H) = sensor_size_mm, sensor_res
        if abs(w * H - h * W) > 1e-6 * w * H:
            raise LensFileError(f"sensor size {w} x {h} mm does not match {W} x {H} pixels")
        # DeepLens keeps the diagonal and derives the size from the resolution's aspect.
        data["r_sensor"] = math.hypot(w, h) / 2
    else:
        r_img = p.image_semi_diameter or par.image_height
        if not r_img or r_img <= 0:
            raise LensFileError(f"{p.source_path}: no image size; give the sensor size")
        if not p.image_semi_diameter:
            p.notes.append(f"image: no semi-diameter; sensor radius {r_img:.4g} mm (paraxial)")
        data["r_sensor"] = float(r_img)
    if sensor_res is not None:
        data["sensor_res"] = [int(v) for v in sensor_res]
    # Catalog glasses DeepLens does not know by a bare name (a given .agf, or one of two
    # bundled glasses with the same name) travel with the lens: `register` loads them.
    glass = catalogs.glass_data(mats)
    if glass:
        data["glass_data"] = glass
    data["notes"] = list(dict.fromkeys(p.notes))
    return data, par


def write_deeplens(p: Prescription, path: str | Path, **kwargs) -> dict:
    data, _ = to_deeplens(p, **kwargs)
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def _material(name: str) -> Material:
    if name.lower() == "air":
        return AIR
    from .materials import bundled_catalog

    cat = bundled_catalog(name)
    if cat:  # a bare DeepLens key: keep the catalog it means in DeepLens (F2 is SCHOTT's)
        return Material(name=name.upper(), catalog=cat)
    if name.startswith("agf:"):  # a catalog glass registered by twinrobo.lensio.materials
        _, cat, glass, _ = name.split(":", 3)
        return Material(name=glass.upper(), catalog=cat.upper())
    if "/" in name:
        nd, vd = name.split("/", 1)
        return Material.model(float(nd), float(vd))
    return Material(name=name)


def from_deeplens(path: str | Path) -> Prescription:
    """A DeepLens lens JSON file as a prescription (Spheric, Aspheric, Plane, Aperture and
    ThinLens surfaces; decentered or tilted surfaces are rejected)."""
    path = Path(path)
    data = json.loads(path.read_text())
    where = str(path)
    p = Prescription(name=str(data.get("info", "")), source_format="deeplens", source_path=where)
    if data.get("glass_data"):
        from .materials import register

        register(data["glass_data"])
        for key, entry in data["glass_data"].items():  # agf:<catalog>:<name>:<hash>
            _, cat, glass, _ = key.split(":", 3)
            known = p.packed_glasses.setdefault(cat.upper(), {})
            if glass in known and known[glass] != entry:
                raise LensFileError(
                    f"{where}: saves two different glasses named {cat.upper()} {glass}; "
                    "they cannot be told apart by name"
                )
            known[glass] = entry
    obj = Surface(thickness=-float(data["obj_depth"]) if "obj_depth" in data else math.inf)
    surfs = [obj]
    for i, d in enumerate(data["surfaces"], start=1):
        at = f"{where}: surface {i}"
        if list(d.get("pos_xy", [0, 0])) != [0, 0] or list(d.get("vec_local", [0, 0, 1])) != [
            0,
            0,
            1,
        ]:
            raise LensFileError(f"{at}: decentered or tilted surfaces cannot be exported")
        if d.get("is_square"):
            raise LensFileError(f"{at}: square apertures cannot be exported")
        typ = d["type"]
        s = Surface(
            thickness=float(d["d_next"]),
            semi_diameter=float(d["r"]),
            material=_material(d.get("mat2", "air")),
            stop=bool(d.get("is_aperture", False)) or typ in ("Aperture", "Stop"),
            source=at,
        )
        if typ in ("Spheric", "Aspheric"):
            roc = d.get("roc")
            c = (1.0 / roc if roc else 0.0) if roc is not None else float(d.get("c", 0.0))
            s.radius = math.inf if c == 0 else 1.0 / c
            if typ == "Aspheric":
                s.conic = float(d.get("k", 0.0))
                ai = list(d.get("ai", []))
                start = 2 if d.get("use_ai2", True) else 4
                s.coeffs = {start + 2 * j: float(a) for j, a in enumerate(ai) if a}
                s.kind = "asphere" if s.coeffs else "standard"
        elif typ in ("Plane", "Aperture", "Stop"):
            pass
        elif typ == "ThinLens":
            s.kind, s.focal_length = "paraxial", float(d["f"])
        else:
            raise LensFileError(f"{at}: DeepLens surface type {typ} cannot be exported")
        surfs.append(s)
    # A glass saved with the lens replaces the catalog's glass of that name; a lens that uses
    # both under one name cannot be told apart once read.
    bare = {
        (d.get("mat2", "").upper(), cat)
        for d in data["surfaces"]
        if not str(d.get("mat2", "")).startswith("agf:")
        and (cat := (bundled_catalog(str(d.get("mat2", "air"))) or ""))
    }
    for cat, glasses in p.packed_glasses.items():
        for glass in glasses:
            if (glass.upper(), cat) in bare:
                raise LensFileError(
                    f"{where}: uses {cat} {glass} twice, the catalog's and a replacement; "
                    "they cannot be told apart by name"
                )
    r_sensor = data.get("r_sensor")
    if r_sensor is None and data.get("sensor_size"):
        r_sensor = math.hypot(*data["sensor_size"]) / 2
    surfs.append(Surface(semi_diameter=None if r_sensor is None else float(r_sensor)))
    p.surfaces = surfs
    if data.get("enpd"):
        p.aperture_type, p.aperture_value = "epd", float(data["enpd"])
    else:
        p.aperture_type = "float"
    w = float(data.get("primary_wvln", DEFAULT_WAVE))
    p.wavelengths_um = [WVLN_RGB[2], w, WVLN_RGB[0]] if w == DEFAULT_WAVE else [w]
    p.wavelength_weights = [1.0] * len(p.wavelengths_um)
    p.primary_wavelength = p.wavelengths_um.index(w)
    if r_sensor and data.get("foclen"):
        half = math.degrees(math.atan(float(r_sensor) / float(data["foclen"])))
        p.fields = [(0.0, 0.0), (0.0, 0.7 * half), (0.0, half)]
        p.field_weights = [1.0, 1.0, 1.0]
    return p.check()
