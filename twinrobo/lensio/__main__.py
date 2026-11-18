"""Lens files from the shell.

python -m twinrobo.lensio info design.zmx
python -m twinrobo.lensio convert design.zmx design.seq
python -m twinrobo.lensio convert design.zmx lens.json --agf MYGLASS.agf
"""

from __future__ import annotations

import argparse
import sys

from . import READ_FORMATS, WRITE_FORMATS, catalogs_for, load_lens, save_lens
from .deeplens_json import indices
from .materials import resolve_all
from .paraxial import paraxial


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m twinrobo.lensio", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    info = sub.add_parser("info", help="print a lens file's prescription and first-order data")
    info.add_argument("lens", help=f"a lens file ({', '.join(READ_FORMATS)})")
    conv = sub.add_parser("convert", help="convert a lens file to another format")
    conv.add_argument("src", help=f"input ({', '.join(READ_FORMATS)})")
    conv.add_argument("dst", help=f"output ({', '.join(WRITE_FORMATS)})")
    for p in (info, conv):
        p.add_argument("--lenient", action="store_true", help="skip records it does not know")
        p.add_argument("--agf", action="append", default=[], help="a glass catalog (.agf)")
        p.add_argument(
            "--model-glass",
            action="store_true",
            help="use the file's nd/vd for glasses in no catalog (approximates dispersion)",
        )
    args = ap.parse_args(argv)

    if args.cmd == "info":
        p = load_lens(args.lens, strict=not args.lenient)
        cats = catalogs_for(p, args.agf)
        mats = resolve_all(p, cats, args.model_glass)
        par = paraxial(p, indices(p, mats))
        print(p.summary())
        print(
            f"first order (paraxial, {p.primary_wavelength_um:g} um): EFL {par.efl:.6g} mm, "
            f"BFL {par.bfl:.6g} mm, EPD {par.epd:.6g} mm, F/{par.fnum:.4g}, "
            f"image height {par.image_height:.6g} mm"
        )
        return 0
    p = load_lens(args.src, strict=not args.lenient)
    save_lens(p, args.dst, catalogs=catalogs_for(p, args.agf), allow_model_glass=args.model_glass)
    print(f"{args.src} -> {args.dst}")
    for n in dict.fromkeys(p.notes):
        print(f"note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
