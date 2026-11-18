"""Zemax OpticStudio sequential lens files (``.zmx``) and archives (``.zar``).

Reading follows OpticStudio's documented surface semantics. The ``.zmx`` record layout itself
has no public grammar, so the positions of fields inside a record follow files OpticStudio
writes (and RayOptics' reader); `docs/imaging/lens-files.md` lists what has been checked
against files from OpticStudio and what has not.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..exceptions import LensFileError
from .prescription import AIR, UNIT_MM, Material, Prescription, Surface, scale_surface

FIELD_CODES = {0: "angle", 1: "object_height", 2: "paraxial_image_height", 3: "real_image_height"}

#: Records that do not change the traced geometry of a centered sequential lens: display,
#: tolerancing, optimization, file references, solves and pickups (the file stores the values
#: they produced), environment defaults (20 C, 1 atm is what the glass data assume).
IGNORED = {
    "VERS", "NOTE", "PFIL", "LANG", "ENVD", "GFAC", "RAIM", "PUSH", "SDMA", "ROPD", "PICB",
    "POLS", "GLRS", "GSTD", "NSCD", "COFN", "BLNK", "TOL", "TOLE", "MNUM", "MOFF", "PZUP",
    "FIMP", "TCED", "TCMM", "RGLA", "TRAC", "MNCA", "MNEA", "MNCG", "MNEG", "MXCA", "MXCG",
    "OPDX", "PUPD", "EFFL", "HIDE", "MAZH", "PPAR", "VPAR", "EDGE", "VCON", "UDAD", "USAP",
    "PKUP", "SLAB", "POPS", "COMM", "MEMA", "VDXN", "VDYN", "VCXN", "VCYN", "VANN", "ZVDX",
    "ZVDY", "ZVCX", "ZVCY", "ZVAN", "WWGN", "WAVN", "ISNA", "VDSZ", "DMFS", "TOTR", "PMAG",
    "FLAP", "MIRR", "THEM", "XFIE", "XDIS",
}  # fmt: skip
#: Records whose presence means the file describes something TwinRobo cannot trace.
UNSUPPORTED = {
    "OBNA": "an object-space numerical aperture (use ENPD, FNUM or float by stop size)",
    "SQAP": "a rectangular aperture",
    "ELAP": "an elliptical aperture",
    "OBSC": "an obscuration",
    "OBDC": "a decentered aperture",
    "UDAP": "a user-defined aperture",
    "COAT": "a coating (TwinRobo does not model coatings)",
}
SURFACE_TYPES = {"STANDARD", "EVENASPH", "ODDASPHE", "ODDASPH", "XASPHERE", "PARAXIAL", "COORDBRK"}
UNITS = {"MM": "mm", "CM": "cm", "IN": "in", "INCH": "in", "METER": "m", "METERS": "m", "M": "m"}


def decode(data: bytes, encoding: str | None = None) -> str:
    """Text of a ``.zmx`` file: UTF-16 (OpticStudio's default), UTF-8, or ANSI."""
    if encoding:
        return data.decode(encoding)
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data[1:200:2].count(0) > 50:  # UTF-16 LE without a byte order mark
        return data.decode("utf-16-le")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252")  # Windows ANSI; pass ``encoding`` for another code page


def _f(tok: str) -> float:
    t = tok.strip().upper()
    if t in ("INFINITY", "INF", "1.#INF"):
        return math.inf
    return float(tok)


def read_zmx(path: str | Path, encoding: str | None = None, strict: bool = True) -> Prescription:
    """Read a Zemax ``.zmx`` file. ``strict=False`` turns unknown records into notes."""
    path = Path(path)
    return parse_zmx(decode(path.read_bytes(), encoding), str(path), strict=strict)


def parse_zmx(text: str, where: str = "<zmx>", strict: bool = True) -> Prescription:
    p = Prescription(source_format="zemax", source_path=where)
    unit = "mm"
    surfs: list[dict] = []
    cur: dict | None = None
    ftyp: list[str] = []
    xf: list[float] = []
    yf: list[float] = []
    fw: list[float] = []
    waves: dict[int, tuple[float, float]] = {}
    wavl: list[float] = []
    wwgt: list[float] = []
    pwav = 1
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        tok = rest.split()
        at = f"{where}:{n}"

        def bad(msg: str, at=at, line=line):
            raise LensFileError(f"{at}: {msg} [{line}]")

        try:
            if cmd == "SURF":
                cur = {"source": at, "type": "STANDARD", "parm": {}, "xdat": {}, "glas": None}
                surfs.append(cur)
            elif cmd == "MODE":
                if tok and tok[0] != "SEQ":
                    bad("only sequential (MODE SEQ) files are supported")
            elif cmd == "NAME":
                p.name = rest.strip().strip('"')
            elif cmd == "UNIT":
                if not tok or tok[0].upper() not in UNITS:
                    bad("unknown lens unit")
                unit = UNITS[tok[0].upper()]
            elif cmd == "ENPD":
                p.aperture_type, p.aperture_value = "epd", _f(tok[0])
            elif cmd == "FNUM":
                p.aperture_type, p.aperture_value = "fnum", _f(tok[0])
            elif cmd == "FLOA":
                p.aperture_type, p.aperture_value = "float", None
            elif cmd == "GCAT":
                p.glass_catalogs = [t.upper() for t in tok]
            elif cmd == "FTYP":
                ftyp = tok
            elif cmd in ("XFLN", "XFLD"):
                xf = [_f(t) for t in tok]
            elif cmd in ("YFLN", "YFLD"):
                yf = [_f(t) for t in tok]
            elif cmd in ("FWGN", "FWGT"):
                fw = [_f(t) for t in tok]
            elif cmd == "WAVM":
                waves[int(tok[0])] = (_f(tok[1]), _f(tok[2]) if len(tok) > 2 else 1.0)
            elif cmd == "WAVL":
                wavl = [_f(t) for t in tok]
            elif cmd == "WWGT":
                wwgt = [_f(t) for t in tok]
            elif cmd == "PWAV":
                pwav = int(tok[0])
            elif cmd in UNSUPPORTED:
                if cmd == "COAT" and not strict:
                    p.notes.append(f"{at}: coating ignored")
                else:
                    bad(f"the file uses {UNSUPPORTED[cmd]}")
            elif cmd == "CONF":
                bad("multi-configuration (zoom) lenses are not supported")
            elif cur is not None and cmd in ("TYPE", "CURV", "CONI", "PARM", "XDAT", "DISZ",
                                             "DIAM", "CLAP", "STOP", "GLAS"):  # fmt: skip
                if cmd == "TYPE":
                    if tok[0] not in SURFACE_TYPES:
                        bad(f"surface type {tok[0]} is not supported")
                    cur["type"] = tok[0]
                elif cmd == "CURV":
                    cur["curv"] = _f(tok[0])
                elif cmd == "CONI":
                    cur["conic"] = _f(tok[0])
                elif cmd == "PARM":
                    cur["parm"][int(tok[0])] = _f(tok[1])
                elif cmd == "XDAT":
                    cur["xdat"][int(tok[0])] = _f(tok[1])
                elif cmd == "DISZ":
                    cur["disz"] = _f(tok[0])
                elif cmd == "DIAM":
                    cur["diam"] = _f(tok[0])
                elif cmd == "CLAP":  # CLAP <min radius> <max radius> ...
                    if _f(tok[0]) != 0:
                        bad("an annular aperture")
                    cur["clap"] = _f(tok[1])
                elif cmd == "STOP":
                    cur["stop"] = True
                elif cmd == "GLAS":
                    cur["glas"] = (tok, at)
            elif cmd in IGNORED:
                pass
            elif strict:
                bad(f"unknown record {cmd}; pass strict=False to ignore it")
            else:
                p.notes.append(f"{at}: ignored record {cmd}")
        except (ValueError, IndexError) as e:
            raise LensFileError(f"{at}: cannot read [{line}]: {e}") from e

    if not surfs:
        raise LensFileError(f"{where}: no surfaces (SURF records)")
    p.surfaces = _build_surfaces(surfs, where)

    # Fields: FTYP <type> <telecentric> <number of fields> <number of wavelengths> ...
    code = int(_f(ftyp[0])) if ftyp else 0
    if code not in FIELD_CODES:
        raise LensFileError(f"{where}: field type {code} is not supported")
    p.field_type = FIELD_CODES[code]
    nf = int(_f(ftyp[2])) if len(ftyp) > 2 else max(len(xf), len(yf), 1)
    nw = int(_f(ftyp[3])) if len(ftyp) > 3 else None
    xf, yf = (xf + [0.0] * nf)[:nf], (yf + [0.0] * nf)[:nf]
    p.fields = list(zip(xf, yf, strict=True))
    p.field_weights = (fw + [1.0] * nf)[:nf]
    if waves:
        slots = sorted(waves)[:nw] if nw else sorted(waves)
        p.wavelengths_um = [waves[i][0] for i in slots]
        p.wavelength_weights = [waves[i][1] for i in slots]
        if pwav not in slots:
            raise LensFileError(f"{where}: primary wavelength {pwav} is not defined")
        p.primary_wavelength = slots.index(pwav)
    elif wavl:
        p.wavelengths_um = wavl
        p.wavelength_weights = (wwgt + [1.0] * len(wavl))[: len(wavl)]
        p.primary_wavelength = pwav - 1

    k = UNIT_MM[unit]
    p.surfaces = [scale_surface(s, k) for s in p.surfaces]
    if p.aperture_type == "epd":
        p.aperture_value *= k
    if p.field_type != "angle":
        p.fields = [(x * k, y * k) for x, y in p.fields]
    return p.check()


def _glass(tok: list[str], at: str) -> Material:
    """A ``GLAS`` record: ``GLAS <name> <solve> <?> <nd> <vd> <dPgF> ...``."""
    name = tok[0]
    if name.upper() == "MIRROR":
        raise LensFileError(f"{at}: mirrors are not supported")
    solve = int(_f(tok[1])) if len(tok) > 1 else 0
    nd = _f(tok[3]) if len(tok) > 3 else None
    vd = _f(tok[4]) if len(tok) > 4 else None
    if solve == 1 or name == "___BLANK":  # a model glass: defined by its index and Abbe number
        if not nd or nd <= 1 or not vd:
            raise LensFileError(f"{at}: a model glass needs nd > 1 and vd > 0")
        return Material.model(nd, vd)
    if solve not in (0, 1):
        raise LensFileError(f"{at}: glass solve type {solve} is not supported")
    return Material(name=name, nd=nd or None, vd=vd or None)


def _build_surfaces(surfs: list[dict], where: str) -> list[Surface]:
    out: list[Surface] = []
    for i, d in enumerate(surfs):
        at = d["source"]
        typ = d["type"]
        parm = {k: v for k, v in d["parm"].items() if v != 0}
        curv = d.get("curv", 0.0)
        s = Surface(
            radius=math.inf if curv == 0 else 1.0 / curv,
            conic=d.get("conic", 0.0),
            thickness=d.get("disz", 0.0),
            semi_diameter=d.get("diam"),
            clear_aperture=d.get("clap"),
            stop=d.get("stop", False),
            material=AIR if d["glas"] is None else _glass(*d["glas"]),
            source=at,
            label=f"surface {i}",
        )
        if typ == "STANDARD":
            if parm:
                raise LensFileError(f"{at}: a STANDARD surface with parameters {sorted(parm)}")
        elif typ == "EVENASPH":  # PARM i -> r^(2i), i = 1..8
            s.kind = "asphere"
            s.coeffs = {2 * j: v for j, v in parm.items()}
        elif typ in ("ODDASPHE", "ODDASPH"):  # PARM i -> r^i, i = 1..8
            odd = sorted(j for j in parm if j % 2)
            if odd:
                raise LensFileError(f"{at}: odd aspheric terms r^{odd} are not supported")
            s.kind = "asphere"
            s.coeffs = dict(parm)
        elif typ == "XASPHERE":  # XDAT 1 = terms, 2 = normalization radius, i+2 -> rho^(2i)
            x = d["xdat"]
            nterms, rn = int(x.get(1, 0)), x.get(2, 0.0)
            if parm:
                raise LensFileError(f"{at}: extended asphere parameters {sorted(parm)}")
            coef = {2 * j: x.get(j + 2, 0.0) for j in range(1, nterms + 1)}
            coef = {p: a for p, a in coef.items() if a}
            if coef and not rn > 0:
                raise LensFileError(f"{at}: extended asphere without a normalization radius")
            s.kind = "asphere"
            s.coeffs = {p: a / rn**p for p, a in coef.items()}
        elif typ == "PARAXIAL":  # PARM 1 = focal length, PARM 2 = OPD mode
            s.kind = "paraxial"
            s.focal_length = d["parm"].get(1, 0.0)
            s.radius = math.inf
        elif typ == "COORDBRK":  # PARM 1, 2 = decenter, 3, 4, 5 = tilt, 6 = order
            moved = [j for j in (1, 2, 3, 4, 5) if parm.get(j)]
            if moved:
                raise LensFileError(f"{at}: a tilted or decentered coordinate break")
            if d["glas"] is not None or s.stop:
                raise LensFileError(f"{at}: a coordinate break with glass or stop")
            s.kind = "coordbrk"
        if s.kind == "asphere" and not s.coeffs:
            s.kind = "standard"
        out.append(s)
    # A coordinate break with no decenter or tilt only moves along the axis: fold it into the
    # gap before it (the medium is unchanged across it).
    merged: list[Surface] = []
    for s in out:
        if s.kind == "coordbrk":
            if not merged:
                raise LensFileError(f"{s.source}: the object surface is a coordinate break")
            merged[-1].thickness += s.thickness
        else:
            merged.append(s)
    return merged


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _num(v: float) -> str:
    return "INFINITY" if math.isinf(v) else repr(float(v))


def format_zmx(p: Prescription) -> str:
    """Text of a ``.zmx`` file for ``p`` (millimeters)."""
    p.check()
    rows = ["MODE SEQ", f"NAME {p.title}", "UNIT MM X W X CM MR CPMM"]
    if p.aperture_type == "epd":
        rows.append(f"ENPD {_num(p.aperture_value)}")
    elif p.aperture_type == "fnum":
        rows.append(f"FNUM {_num(p.aperture_value)} 0")
    else:
        rows.append("FLOA")
    if p.glass_catalogs:
        rows.append("GCAT " + " ".join(p.glass_catalogs))
    code = {v: k for k, v in FIELD_CODES.items()}[p.field_type]
    rows.append(f"FTYP {code} 0 {len(p.fields)} {len(p.wavelengths_um)} 0 0 0")
    rows.append("XFLN " + " ".join(_num(x) for x, _ in p.fields))
    rows.append("YFLN " + " ".join(_num(y) for _, y in p.fields))
    rows.append("FWGN " + " ".join(_num(w) for w in p.field_weights))
    for i, (w, wt) in enumerate(zip(p.wavelengths_um, p.wavelength_weights, strict=True), 1):
        rows.append(f"WAVM {i} {_num(w)} {_num(wt)}")
    rows.append(f"PWAV {p.primary_wavelength + 1}")
    for i, s in enumerate(p.surfaces):
        rows.append(f"SURF {i}")
        if s.stop:
            rows.append("  STOP")
        if s.kind == "paraxial":
            rows += ["  TYPE PARAXIAL", f"  PARM 1 {_num(s.focal_length)}", "  PARM 2 0"]
        elif s.coeffs:
            top = max(s.coeffs)
            if top <= 16:
                rows.append("  TYPE EVENASPH")
                rows += [f"  PARM {p_ // 2} {_num(a)}" for p_, a in sorted(s.coeffs.items())]
            else:  # beyond r^16: an extended asphere, normalization radius 1 mm
                rows += ["  TYPE XASPHERE", f"  XDAT 1 {top // 2} 0 0 0 0 0 0", "  XDAT 2 1 0"]
                rows += [
                    f"  XDAT {p_ // 2 + 2} {_num(a)} 0 0 0 0 0 0"
                    for p_, a in sorted(s.coeffs.items())
                ]
        else:
            rows.append("  TYPE STANDARD")
        if s.kind != "paraxial":
            rows.append(f'  CURV {_num(s.curvature)} 0 0 0 0 ""')
            if s.conic:
                rows.append(f"  CONI {_num(s.conic)}")
        m = s.material
        if m.is_model:
            rows.append(f"  GLAS ___BLANK 1 0 {_num(m.nd)} {_num(m.vd)} 0 0 0 0 0 0")
        elif not m.is_air:
            rows.append(f"  GLAS {m.name} 0 0 {_num(m.nd or 0)} {_num(m.vd or 0)} 0 0 0 0 0 0")
        rows.append(f"  DISZ {_num(s.thickness)}")
        if s.semi_diameter is not None:
            rows.append(f'  DIAM {_num(s.semi_diameter)} 1 0 0 1 ""')
        if s.clear_aperture is not None:
            rows.append(f"  CLAP 0 {_num(s.clear_aperture)} 0")
    return "\n".join(rows) + "\n"


def write_zmx(p: Prescription, path: str | Path, encoding: str = "utf-16") -> None:
    """Write ``p`` as a ``.zmx`` file, by default in UTF-16 with a byte order mark."""
    Path(path).write_text(format_zmx(p).replace("\n", "\r\n"), encoding=encoding, newline="")


def read_zar(path: str | Path, strict: bool = True) -> tuple[Prescription, dict[str, bytes]]:
    """Read a Zemax archive: its lens and the glass catalogs (``.agf``) packed with it.

    The archive format is undocumented; this uses the ``zmxtools`` package
    (``pip install zmxtools``, AGPL-3.0).
    """
    try:
        from zmxtools import zar
    except ImportError as e:
        raise LensFileError(
            "reading a .zar archive needs zmxtools: pip install zmxtools "
            "(or unpack it in OpticStudio and use the .zmx)"
        ) from e
    zmx = None
    agf: dict[str, bytes] = {}
    for item in zar.read(str(path)):
        name = Path(item.file_name).name
        if name.upper().endswith(".ZMX"):
            zmx = (name, item.unpacked_contents)
        elif name.upper().endswith(".AGF"):
            agf[name] = item.unpacked_contents
    if zmx is None:
        raise LensFileError(f"{path}: the archive holds no .zmx lens")
    return parse_zmx(decode(zmx[1]), f"{path}!{zmx[0]}", strict=strict), agf
