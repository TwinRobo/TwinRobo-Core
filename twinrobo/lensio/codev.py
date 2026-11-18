"""CODE V sequence files (``.seq``): the lens-entry subset of the command language.

A ``.seq`` file is a program. This reads the commands that enter a centered sequential lens
(``S``/``SO``/``SI`` surface rows and their qualifiers, system data) and rejects anything else
that could change the optics: macros, solves on later surfaces, zoom positions, tilts and
decenters. Aspheric coefficients follow CODE V's lettering, A = r^4 ... H = r^18, J = r^20
(there is no I).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ..exceptions import LensFileError
from .prescription import AIR, UNIT_MM, Material, Prescription, Surface, scale_surface

ASPHERE_LETTERS = {"A": 4, "B": 6, "C": 8, "D": 10, "E": 12, "F": 14, "G": 16, "H": 18, "J": 20}
FIELD_COMMANDS = {
    "YAN": ("angle", 1), "XAN": ("angle", 0),
    "YOB": ("object_height", 1), "XOB": ("object_height", 0),
    "YIM": ("paraxial_image_height", 1), "XIM": ("paraxial_image_height", 0),
    "YRI": ("real_image_height", 1), "XRI": ("real_image_height", 0),
}  # fmt: skip
DIMS = {"M": "mm", "C": "cm", "I": "in"}
#: Commands that do not change a centered sequential lens' traced geometry.
IGNORED = {
    "LEN", "GO", "INI", "TIT", "TITLE", "VUY", "VLY", "VUX", "VLX", "DOR", "WTF", "CWL",
    "SLB", "CCY", "CCX", "THC", "GLC", "PIM", "NAO", "TEM", "PRE", "DDM", "PRV", "PWL",
    "END", "RDM", "DIM", "WL", "REF", "WTW", "EPD", "FNO", "STO", "SPH", "CON", "ASP",
    "K", "CIR", "SO", "SI", "S", "ZOO", "IMC",
}  # fmt: skip
#: Commands that describe something TwinRobo cannot trace.
UNSUPPORTED = {
    "ADE": "a tilted surface", "BDE": "a tilted surface", "CDE": "a tilted surface",
    "XDE": "a decentered surface", "YDE": "a decentered surface", "ZDE": "a decentered surface",
    "DAR": "a decenter-and-return", "BEN": "a bend", "REV": "a reverse decenter",
    "REX": "a rectangular aperture", "REY": "a rectangular aperture",
    "ELX": "an elliptical aperture", "ELY": "an elliptical aperture",
    "OBS": "an obscuration", "HOL": "a hole", "EDG": "an edge aperture",
    "XTO": "a toroidal surface", "YTO": "a toroidal surface", "SPS": "a special surface",
    "DIF": "a diffractive surface", "HOE": "a holographic surface", "GRT": "a grating",
    "ZOO": "zoom positions", "NA": "an image-space numerical aperture",
    "NAO": "an object-space numerical aperture", "XOF": "an aperture offset",
    "YOF": "an aperture offset", "GLB": "a global reference", "RET": "a coordinate return",
    "PIK": "a pickup", "MIR": "a mirror", "REFL": "a mirror",
}  # fmt: skip
IGNORED -= set(UNSUPPORTED)


def split_commands(text: str) -> list[tuple[list[str], int]]:
    """Commands with their line numbers: comments (!) removed, & continuations joined,
    ``;`` separators split outside quotes."""
    out: list[tuple[list[str], int]] = []
    pending, start = "", 0
    for n, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not pending:
            start = n
        if line.rstrip().endswith("&"):
            pending += line.rstrip()[:-1] + " "
            continue
        line, pending = pending + line, ""
        for part in _split_unquoted(line, ";"):
            toks = re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", part)
            if toks:
                out.append((toks, start))
    return out


def _split_unquoted(line: str, sep: str) -> list[str]:
    """``line`` split at ``sep`` outside '...' and "..." strings (each quote closes only its
    own kind, so an apostrophe inside double quotes is text)."""
    parts, cur, quote = [], [], None
    for ch in line:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == sep:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def _strip_comment(line: str) -> str:
    return _split_unquoted(line, "!")[0]


def _glass(tok: str, at: str) -> Material:
    """``NAME``, ``NAME_CATALOG``, a fictitious ``nd:vd`` or ``nnnn.vvvv`` glass."""
    if ":" in tok:
        nd, vd = (float(v) for v in tok.split(":", 1))
        return Material.model(nd, vd)
    if re.fullmatch(r"\d+\.\d+", tok):  # 5168.6417 -> nd 1.5168, vd 64.17
        whole, frac = tok.split(".")
        return Material.model(1.0 + int(whole) / 10 ** len(whole), float("0." + frac) * 100.0)
    if re.fullmatch(r"\d{6}", tok):
        raise LensFileError(f"{at}: glass code {tok}: name the catalog glass instead")
    if tok.upper() in ("REFL", "MIRROR"):
        raise LensFileError(f"{at}: mirrors are not supported")
    name, _, cat = tok.partition("_")
    return Material(name=name, catalog=cat.upper() or None)


def read_seq(path: str | Path, strict: bool = True) -> Prescription:
    path = Path(path)
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    return parse_seq(text, str(path), strict=strict)


def parse_seq(text: str, where: str = "<seq>", strict: bool = True) -> Prescription:
    p = Prescription(source_format="codev", source_path=where)
    radius_mode = True
    unit = "mm"
    surfs: list[Surface] = []
    image_seen = False
    fields: dict[str, dict[int, list[float]]] = {}
    wts_f: list[float] = []
    wl: list[float] = []
    wtw: list[float] = []
    ref = 1
    for toks, n in split_commands(text):
        at = f"{where}:{n}"
        cmd = toks[0].upper()
        args = toks[1:]
        line = " ".join(toks)

        def bad(msg: str, at=at, line=line):
            raise LensFileError(f"{at}: {msg} [{line}]")

        def num(i: int, args=args, bad=bad) -> float:
            try:
                return float(args[i])
            except (IndexError, ValueError):
                bad("expected a number")

        if re.fullmatch(r"S\d+", cmd) or (
            cmd in ("S", "SO", "SI") and args and re.fullmatch(r"S\d+", args[0].upper())
        ):
            bad("surface pointers (S<n>) are not supported; give surfaces in order")
        if args and cmd not in ("S", "SO", "SI", "TIT", "TITLE", "LEN", "RDM", "DIM", "DDM"):
            if re.fullmatch(r"(S\d+|SO|SI|SA|SS)(\.\.\S+)?", args[0].upper()):
                bad("surface qualifiers are not supported; put the command after its surface")
        if cmd in ("SO", "S", "SI"):
            if image_seen:
                bad("a surface after the image")
            if cmd == "SO" and surfs:
                bad("the object must be the first surface")
            if cmd != "SO" and not surfs:
                surfs.append(Surface(thickness=math.inf, label="object", source=at))
            v = num(0)
            radius = (math.inf if v == 0 else v) if radius_mode else (math.inf if v == 0 else 1 / v)
            th = num(1) if len(args) > 1 else 0.0
            if cmd == "SO" and th >= 1e10:
                th = math.inf  # CODE V's conventional infinite object distance
            mat = _glass(args[2], at) if len(args) > 2 else AIR
            if len(args) > 3:
                bad("unexpected data after the glass")
            surfs.append(Surface(radius=radius, thickness=th, material=mat, source=at, label=cmd))
            image_seen = cmd == "SI"
        elif cmd == "RDM":
            radius_mode = not (args and args[0].upper().startswith("N"))
        elif cmd == "DIM":
            if not args or args[0].upper() not in DIMS:
                bad("DIM must be M, C or I")
            unit = DIMS[args[0].upper()]
        elif cmd in ("TIT", "TITLE"):
            p.name = " ".join(args).strip("'\"")
        elif cmd == "EPD":
            p.aperture_type, p.aperture_value = "epd", num(0)
        elif cmd == "FNO":
            p.aperture_type, p.aperture_value = "fnum", num(0)
        elif cmd == "WL":
            wl = [num(i) / 1000.0 for i in range(len(args))]
        elif cmd == "WTW":
            wtw = [num(i) for i in range(len(args))]
        elif cmd == "REF":
            ref = int(num(0))
        elif cmd in FIELD_COMMANDS:
            ftype, axis = FIELD_COMMANDS[cmd]
            fields.setdefault(ftype, {})[axis] = [num(i) for i in range(len(args))]
        elif cmd == "WTF":
            wts_f = [num(i) for i in range(len(args))]
        elif cmd in ("STO", "SPH", "CON", "ASP", "K", "CIR", "CUY", "RDY", "THI", "GLA") or (
            cmd in ASPHERE_LETTERS
        ):
            if not surfs:
                bad(f"{cmd} before the first surface")
            s = surfs[-1]
            if cmd == "STO":
                if args:
                    bad("STO with data; mark the stop with STO after its S row")
                s.stop = True
            elif cmd == "SPH":
                s.kind, s.coeffs, s.conic = "standard", {}, 0.0
            elif cmd in ("CON", "ASP"):
                s.kind = "asphere" if cmd == "ASP" else "standard"
            elif cmd == "K":
                s.conic = num(0)
            elif cmd in ASPHERE_LETTERS:
                if s.kind != "asphere":
                    bad(f"{cmd} without ASP")
                s.coeffs[ASPHERE_LETTERS[cmd]] = num(0)
            elif cmd == "CIR":
                if args and args[0].upper() in ("EDG", "HOL", "OBS"):
                    bad("edge apertures, holes and obscurations are not supported")
                if args and args[0].upper() == "CLR":
                    args = args[1:]
                s.clear_aperture = s.semi_diameter = num(0, args)
            elif cmd in ("CUY", "RDY"):
                v = num(0)
                if cmd == "CUY":
                    s.radius = math.inf if v == 0 else 1 / v
                else:
                    s.radius = math.inf if v == 0 else v
            elif cmd == "THI":
                s.thickness = num(0)
            elif cmd == "GLA":
                s.material = _glass(args[0], at)
        elif cmd in UNSUPPORTED:
            bad(f"the file uses {UNSUPPORTED[cmd]}")
        elif cmd in IGNORED:
            pass
        elif strict:
            bad(f"unknown command {cmd}; pass strict=False to ignore it")
        else:
            p.notes.append(f"{at}: ignored command {cmd}")

    if not image_seen:
        raise LensFileError(f"{where}: no image surface (SI)")
    for s in surfs:
        s.coeffs = {k: v for k, v in s.coeffs.items() if v}
        if s.kind == "asphere" and not s.coeffs:
            s.kind = "standard"
    # The image row's thickness is a defocus: move the image by it.
    surfs[-2].thickness += surfs[-1].thickness
    surfs[-1].thickness = 0.0
    p.surfaces = surfs

    if wl:
        p.wavelengths_um = wl
        p.wavelength_weights = (wtw + [1.0] * len(wl))[: len(wl)]
        if not 1 <= ref <= len(wl):
            raise LensFileError(f"{where}: reference wavelength {ref} is not defined")
        p.primary_wavelength = ref - 1
    if fields:
        if len(fields) > 1:
            raise LensFileError(f"{where}: fields of more than one type")
        ftype, xy = next(iter(fields.items()))
        nf = max(len(v) for v in xy.values())
        xs, ys = (xy.get(0, []) + [0.0] * nf)[:nf], (xy.get(1, []) + [0.0] * nf)[:nf]
        p.field_type = ftype
        p.fields = list(zip(xs, ys, strict=True))
        p.field_weights = (wts_f + [1.0] * nf)[:nf]

    k = UNIT_MM[unit]
    p.surfaces = [scale_surface(s, k) for s in p.surfaces]
    if p.aperture_type == "epd":
        p.aperture_value *= k
    if p.field_type != "angle":
        p.fields = [(x * k, y * k) for x, y in p.fields]
    return p.check()


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _num(v: float) -> str:
    return repr(float(v))


def _fictitious(nd: float, vd: float) -> str:
    whole = f"{nd - 1.0:.6f}"[2:].rstrip("0") or "0"
    frac = f"{vd / 100.0:.6f}"[2:].rstrip("0") or "0"
    if not 1.0 < nd < 2.0 or not 0.0 < vd < 100.0:
        raise LensFileError(f"model glass nd={nd} vd={vd} has no CODE V fictitious-glass code")
    return f"{whole}.{frac}"


def format_seq(p: Prescription) -> str:
    """Text of a ``.seq`` file for ``p`` (millimeters, radius mode)."""
    p.check()
    # A title in the quote it does not contain (CODE V strings cannot escape a quote).
    name = p.title
    q = "'" if "'" not in name else '"'
    title = name if q not in name else name.replace('"', "'").replace("'", "")
    rows = ["RDM;LEN", f"TITLE {q}{title}{q}", "DIM M"]
    if p.aperture_type == "epd":
        rows.append(f"EPD {_num(p.aperture_value)}")
    elif p.aperture_type == "fnum":
        rows.append(f"FNO {_num(p.aperture_value)}")
    else:
        raise LensFileError(
            "CODE V needs an entrance pupil diameter or F-number; this lens floats its aperture "
            "by stop size (set aperture_type to 'epd' first)"
        )
    # CODE V lists wavelengths longest first.
    order = sorted(range(len(p.wavelengths_um)), key=lambda i: -p.wavelengths_um[i])
    rows.append("WL " + " ".join(_num(p.wavelengths_um[i] * 1000.0) for i in order))
    rows.append(f"REF {order.index(p.primary_wavelength) + 1}")
    rows.append("WTW " + " ".join(_num(p.wavelength_weights[i]) for i in order))
    xcmd = {"angle": "XAN", "object_height": "XOB", "paraxial_image_height": "XIM",
            "real_image_height": "XRI"}[p.field_type]  # fmt: skip
    rows.append(xcmd + " " + " ".join(_num(x) for x, _ in p.fields))
    rows.append(xcmd.replace("X", "Y", 1) + " " + " ".join(_num(y) for _, y in p.fields))
    rows.append("WTF " + " ".join(_num(w) for w in p.field_weights))
    obj = p.surfaces[0]
    rows.append(f"SO {_num(0)} {_num(1e10 if math.isinf(obj.thickness) else obj.thickness)}")
    for s in p.lens_surfaces:
        if s.kind == "paraxial":
            raise LensFileError("CODE V export of paraxial surfaces is not supported")
        if any(k not in ASPHERE_LETTERS.values() for k in s.coeffs):
            raise LensFileError(
                f"CODE V's ASP holds r^4 .. r^20; this lens has terms r^{sorted(s.coeffs)}"
            )
        m = s.material
        glass = ""
        if m.is_model:  # the digit-string form: 1.60311 / 60.6 -> 60311.606
            glass = " " + _fictitious(m.nd, m.vd)
        elif not m.is_air:
            # CODE V spells catalog glasses without hyphens: NBK7_SCHOTT for N-BK7.
            glass = f" {m.name.replace('-', '')}" + (f"_{m.catalog}" if m.catalog else "")
        rad = 0.0 if math.isinf(s.radius) else s.radius
        rows.append(f"S {_num(rad)} {_num(s.thickness)}{glass}")
        if s.stop:
            rows.append("  STO")
        if s.coeffs:
            rows.append("  ASP")
            rows.append(f"  K {_num(s.conic)}")
            letter = {v: k for k, v in ASPHERE_LETTERS.items()}
            rows += [f"  {letter[k]} {_num(a)}" for k, a in sorted(s.coeffs.items())]
        elif s.conic:
            rows += ["  CON", f"  K {_num(s.conic)}"]
        if s.aperture_radius is not None:
            rows.append(f"  CIR {_num(s.aperture_radius)}")
    img = p.surfaces[-1]
    rows.append("SI 0.0 0.0")
    if img.semi_diameter is not None:
        rows.append(f"  CIR {_num(img.semi_diameter)}")
    rows.append("GO")
    return "\n".join(rows) + "\n"


def write_seq(p: Prescription, path: str | Path) -> None:
    Path(path).write_text(format_seq(p), encoding="ascii")
