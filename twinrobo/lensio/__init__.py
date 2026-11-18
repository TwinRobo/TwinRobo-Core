"""Lens design files in and out of TwinRobo.

Read a lens from optical design software and render through it, or export any TwinRobo lens
for it:

=============================  ======  ======  ================================================
Format                         Read    Write   Notes
=============================  ======  ======  ================================================
Zemax OpticStudio ``.zmx``     yes     yes     UTF-16 (default), UTF-8 or ANSI
Zemax archive ``.zar``         yes     no      needs ``pip install zmxtools``; glass catalogs
                                               packed in the archive are used
CODE V ``.seq``                yes     yes     the lens-entry command subset
Prescription table ``.csv``    yes     yes     for lenses typed in from patents and datasheets
DeepLens ``.json``             yes     yes     TwinRobo's rendering format
=============================  ======  ======  ================================================

Lenses are centered, rotationally symmetric and sequential: spheres, conics, even aspheres (and
odd aspheres with only even terms, extended aspheres) and ideal paraxial lenses. Tilts,
decenters, mirrors, freeforms, gradient index, diffractive surfaces, zoom configurations and
curved images are rejected with the file and line that use them.

::

    from twinrobo import lensio

    lens = lensio.load_lens("design.zmx")        # a Prescription
    print(lens.summary())
    lensio.save_lens(lens, "design.seq")         # to CODE V
    lensio.convert("design.zmx", "lens.json")    # to DeepLens, for a CameraSpec

``python -m twinrobo.lensio info|convert ...`` does the same from the shell.
"""

from __future__ import annotations

from pathlib import Path

from ..exceptions import LensFileError
from .materials import GlassCatalogs
from .prescription import AIR, Material, Prescription, Surface

READ_FORMATS = (".zmx", ".zar", ".seq", ".csv", ".json")
WRITE_FORMATS = (".zmx", ".seq", ".csv", ".json")

__all__ = [
    "AIR",
    "GlassCatalogs",
    "LensFileError",
    "Material",
    "Prescription",
    "Surface",
    "READ_FORMATS",
    "WRITE_FORMATS",
    "load_lens",
    "save_lens",
    "convert",
]


def load_lens(
    path: str | Path,
    strict: bool = True,
    encoding: str | None = None,
) -> Prescription:
    """Read a lens file into a `Prescription`, by its extension (see `READ_FORMATS`).

    ``strict=False`` reads past records the reader does not know (listing them in
    ``notes``) instead of raising. Records known to change the optics always raise.
    """
    path = Path(path)
    if not path.is_file():
        raise LensFileError(f"lens file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".zmx":
        from .zemax import read_zmx

        return read_zmx(path, encoding=encoding, strict=strict)
    if ext == ".zar":
        from .zemax import read_zar

        p, agf = read_zar(path, strict=strict)
        p.packed_catalogs = agf
        return p
    if ext == ".seq":
        from .codev import read_seq

        return read_seq(path, strict=strict)
    if ext == ".csv":
        from .table import read_table

        return read_table(path)
    if ext == ".json":
        from .deeplens_json import from_deeplens

        return from_deeplens(path)
    raise LensFileError(f"{path}: unknown lens format {ext!r}; reads {', '.join(READ_FORMATS)}")


def save_lens(
    p: Prescription,
    path: str | Path,
    catalogs: GlassCatalogs | None = None,
    allow_model_glass: bool = False,
    **kwargs,
) -> None:
    """Write ``p`` in the format of ``path``'s extension (see `WRITE_FORMATS`).

    ``catalogs`` resolve its glasses (by default DeepLens' and any packed with ``p``);
    ``allow_model_glass`` lets a glass in no catalog use the file's nd/vd where a format needs
    the glass's index (a CODE V export of a lens that floats its aperture). For ``.json``
    (DeepLens), other ``kwargs`` go to `twinrobo.lensio.deeplens_json.to_deeplens`.
    """
    path = Path(path)
    ext = path.suffix.lower()
    catalogs = catalogs if catalogs is not None else catalogs_for(p)
    if ext in (".zmx", ".seq", ".csv"):
        from .materials import with_catalogs

        # Name each glass's catalog: a bare name can mean different glasses in different
        # catalogs (F2 is in SCHOTT and in CDGM).
        p = with_catalogs(p, catalogs, one_search_order=ext == ".zmx")
    if ext == ".zmx":
        from .zemax import write_zmx

        write_zmx(p, path, **kwargs)
    elif ext == ".seq":
        from .codev import write_seq

        if p.aperture_type in (None, "float"):  # CODE V has no "float by stop size"
            p = with_paraxial_epd(p, catalogs, allow_model_glass)
        write_seq(p, path)
    elif ext == ".csv":
        from .table import write_table

        write_table(p, path)
    elif ext == ".json":
        from .deeplens_json import write_deeplens

        write_deeplens(p, path, catalogs=catalogs, allow_model_glass=allow_model_glass, **kwargs)
    else:
        raise LensFileError(
            f"{path}: unknown lens format {ext!r}; writes {', '.join(WRITE_FORMATS)}"
        )


def catalogs_for(p: Prescription, agf: list[str | Path] | None = None) -> GlassCatalogs:
    """The glass catalogs to resolve ``p``'s glasses in: DeepLens', any packed in the ``.zar``
    archive ``p`` came from, and ``agf`` files, which are searched first."""
    cats = GlassCatalogs()
    for name, data in p.packed_catalogs.items():
        cats.add_agf(Path(name).stem, data)
    for cat, glasses in p.packed_glasses.items():
        cats.add_glasses(cat, glasses)
    for f in agf or ():
        cats.add_agf(Path(f).stem, f)
    return cats


def with_paraxial_epd(
    p: Prescription, catalogs: GlassCatalogs, allow_model_glass: bool = False
) -> Prescription:
    """``p`` with its aperture given as the paraxial EPD its stop size produces."""
    from .deeplens_json import indices
    from .materials import resolve_all
    from .paraxial import paraxial

    q = p.copy()
    par = paraxial(q, indices(q, resolve_all(q, catalogs, allow_model_glass)))
    q.aperture_type, q.aperture_value = "epd", par.epd
    q.notes.append(f"aperture: stop size given as its paraxial EPD {par.epd:.6g} mm")
    return q


def convert(src: str | Path, dst: str | Path, strict: bool = True, **kwargs) -> Prescription:
    """Read ``src`` and write it as ``dst``; returns the prescription."""
    p = load_lens(src, strict=strict)
    save_lens(p, dst, **kwargs)
    return p
