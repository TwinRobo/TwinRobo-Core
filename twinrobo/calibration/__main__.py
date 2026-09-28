"""Command line: calibrate a camera, one step at a time, into one results file.

    python -m twinrobo.calibration board --out board.png            # print it, measure a square
    python -m twinrobo.calibration geometry views/*.png --board board.json
    python -m twinrobo.calibration stereo --left L/*.png --right R/*.png --board board.json
    python -m twinrobo.calibration edges near/*.png --distance-m 0.4 --roi 900,500,160,160
    python -m twinrobo.calibration edges far/*.png --distance-m 2.0 --roi 900,500,160,160
    python -m twinrobo.calibration focus --spec twinrobo/catalog/<id>/camera.yaml
    python -m twinrobo.calibration flatfield flat/*.png
    python -m twinrobo.calibration write --spec twinrobo/catalog/<id>/camera.yaml \\
        --out twinrobo/catalog/<id>/camera.yaml [--module .../module.yaml]

Every step reads and updates ``--results`` (default ``calib.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .geometry import Board, GeometryResult, StereoResult, calibrate, calibrate_stereo, load_images


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))
    print(f"-> {path}")


def _board(args, results) -> Board:
    if args.board:
        return Board.from_json(json.loads(Path(args.board).read_text()))
    if "board" in results:
        return Board.from_json(results["board"])
    raise SystemExit("no board: pass --board board.json (from the `board` step)")


def cmd_board(args, results):
    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    board = Board(sx, sy, args.square_mm / 1000, args.marker_mm / 1000, args.dictionary)
    import imageio.v3 as iio

    iio.imwrite(args.out, board.image(args.px_per_square))
    Path(args.out).with_suffix(".json").write_text(json.dumps(board.to_json(), indent=2))
    results["board"] = board.to_json()
    print(
        f"board {sx}x{sy} -> {args.out} (+ .json). Print it flat at 100 % scale, measure one "
        "square, and if it is not "
        f"{args.square_mm} mm, fix square_m / marker_m in the .json (marker = square x "
        f"{args.marker_mm / args.square_mm:.3f})."
    )


def cmd_geometry(args, results):
    g = calibrate(load_images(args.images), _board(args, results), fix_k3=args.fix_k3)
    results["geometry"] = asdict(g)
    fov = g.fov_deg()
    print(
        f"{g.views}/{len(args.images)} views, {g.corners} corners, RMS {g.rms_px:.3f} px | "
        f"fx {g.fx:.2f} fy {g.fy:.2f} cx {g.cx:.2f} cy {g.cy:.2f} | "
        f"k1 {g.k1:.5f} k2 {g.k2:.5f} p1 {g.p1:.5f} p2 {g.p2:.5f} k3 {g.k3:.5f} | "
        f"pinhole FoV {fov['h']:.1f} x {fov['v']:.1f} deg"
    )
    worst = sorted(zip(g.per_view_rms_px, range(len(g.per_view_rms_px)), strict=True))[-3:]
    if g.rms_px > 1.0:
        print(
            "warning: RMS > 1 px; drop blurred views (worst: "
            + ", ".join(f"#{i} {e:.2f} px" for e, i in reversed(worst))
            + ")"
        )


def cmd_stereo(args, results):
    board = _board(args, results)
    left, right = load_images(args.left), load_images(args.right)
    gl, gr = calibrate(left, board), calibrate(right, board)
    s = calibrate_stereo(left, right, board, gl, gr)
    results.update({"geometry": asdict(gl), "geometry_right": asdict(gr), "stereo": asdict(s)})
    print(
        f"stereo: {s.views} pairs, RMS {s.rms_px:.3f} px | baseline {1000 * s.baseline_m:.2f} mm | "
        f"right eye t {s.translation_m} m, rpy {s.rotation_rpy_deg} deg"
    )


def cmd_edges(args, results):
    from .sharpness import FREQS, EdgeMeasurement, sample_mtf, slanted_edge_mtf

    rois = [tuple(int(v) for v in r.split(",")) for r in args.roi] or [None]
    edges = results.setdefault("edges", [])
    for path, im in zip(args.images, load_images(args.images), strict=True):
        h, w = im.shape[:2]
        for roi in rois:
            x0, y0, rw, rh = roi if roi else (0, 0, w, h)
            e = slanted_edge_mtf(im[y0 : y0 + rh, x0 : x0 + rw])
            m = EdgeMeasurement(
                distance_m=args.distance_m,
                field=EdgeMeasurement.field_of((x0 + rw / 2 - 0.5, y0 + rh / 2 - 0.5), w, h),
                mtf=sample_mtf(e.freq, e.mtf, FREQS),  # per pixel of this capture
                vertical=e.vertical,
                source=f"{Path(path).name}@{x0},{y0},{rw},{rh}",
            )
            edges.append({**m.to_json(), "image_width": w, "angle_deg": e.angle_deg})
            print(
                f"{m.source}: MTF50 {e.mtf50:.3f} cy/px "
                f"({'vertical' if e.vertical else 'horizontal'} edge, {e.angle_deg:.1f} deg), "
                f"field ({m.field[0]:+.2f}, {m.field[1]:+.2f})"
            )


def cmd_focus(args, results):
    import numpy as np

    from ..camera import build_reference_optics
    from ..spec import CameraSpec
    from .sharpness import FREQS, EdgeMeasurement, fit_focus

    spec = CameraSpec.from_yaml(args.spec)
    optics = build_reference_optics(spec, device=args.device)
    if optics is None:
        raise SystemExit("the spec has no lens model (lens.deeplens_model)")
    W = spec.sensor.resolution.width
    edges = results.get("edges") or []
    ms = []
    for e in edges:  # MTF is per pixel of the capture: resample it per pixel of the spec
        scale = e.get("image_width", W) / W  # capture pixels per spec pixel
        mtf = np.interp(FREQS / scale, FREQS, e["mtf"], right=0.0)
        ms.append(
            EdgeMeasurement(
                e["distance_m"], tuple(e["field"]), list(mtf), e["vertical"], e["source"]
            )
        )
    fit = fit_focus(optics, ms)
    results["focus"] = {
        "focus_m": fit.focus_m,
        "rms_error": fit.rms_error,
        "per_edge": fit.per_edge,
    }
    print(f"focus {fit.focus_m:.3f} m (MTF50 RMS error {fit.rms_error:.4f} cy/px)")
    for e in fit.per_edge:
        print(
            f"  {e['source']}: {e['distance_m']} m measured {e['mtf50']:.3f}, "
            f"model {e['model_mtf50']:.3f}"
        )


def cmd_flatfield(args, results):
    from .vignetting import flat_field

    v = flat_field(load_images(args.images), black_level=args.black_level)
    results["vignetting"] = asdict(v)
    print(
        f"vignetting: corner {v.corner:.3f} of center, coeffs {v.to_spec()['coeffs']}, "
        f"fit RMS {v.rms_residual:.4f}"
    )


def cmd_write(args, results):
    from .spec_writer import measured_camera_yaml, measured_module_yaml
    from .vignetting import Vignetting

    geometry = GeometryResult(**results["geometry"]) if "geometry" in results else None
    focus = (results.get("focus") or {}).get("focus_m")
    vig = Vignetting(**results["vignetting"]) if "vignetting" in results else None
    if not (geometry or focus or vig):
        raise SystemExit("nothing measured yet in the results file")
    text = measured_camera_yaml(args.spec, geometry, focus, vig, notes=args.notes or "")
    Path(args.out).write_text(text)
    print(f"camera -> {args.out}")
    if args.module:
        if "stereo" not in results:
            raise SystemExit("no stereo result; run the `stereo` step")
        out = args.module_out or args.module
        Path(out).write_text(measured_module_yaml(args.module, StereoResult(**results["stereo"])))
        print(f"module -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m twinrobo.calibration",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results", default="calib.json", help="results file (read and updated)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="make a printable ChArUco board")
    b.add_argument("--out", default="board.png")
    b.add_argument("--squares", default="11x8")
    b.add_argument("--square-mm", type=float, default=15.0)
    b.add_argument("--marker-mm", type=float, default=11.0)
    b.add_argument("--dictionary", default="DICT_5X5_100")
    b.add_argument("--px-per-square", type=int, default=120)
    g = sub.add_parser("geometry", help="intrinsics + distortion from ChArUco views")
    g.add_argument("images", nargs="+")
    g.add_argument("--board")
    g.add_argument("--fix-k3", action="store_true")
    s = sub.add_parser("stereo", help="both eyes + the right eye's pose from simultaneous pairs")
    s.add_argument("--left", nargs="+", required=True)
    s.add_argument("--right", nargs="+", required=True)
    s.add_argument("--board")
    e = sub.add_parser("edges", help="slanted-edge MTF50 at one target distance")
    e.add_argument("images", nargs="+")
    e.add_argument("--distance-m", type=float, required=True)
    e.add_argument("--roi", action="append", default=[], help="x,y,w,h (repeatable)")
    f = sub.add_parser("focus", help="fit the focus distance to the measured edges")
    f.add_argument("--spec", required=True)
    f.add_argument("--device", default=None)
    v = sub.add_parser("flatfield", help="relative illumination from flat fields")
    v.add_argument("images", nargs="+")
    v.add_argument("--black-level", type=float, default=0.0)
    w = sub.add_parser("write", help="write the measured camera.yaml (and module.yaml)")
    w.add_argument("--spec", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--module")
    w.add_argument("--module-out")
    w.add_argument("--notes", help="extra provenance lines for the header")
    args = ap.parse_args(argv)
    path = Path(args.results)
    results = _load(path)
    {
        "board": cmd_board,
        "geometry": cmd_geometry,
        "stereo": cmd_stereo,
        "edges": cmd_edges,
        "focus": cmd_focus,
        "flatfield": cmd_flatfield,
        "write": cmd_write,
    }[args.cmd](args, results)
    if args.cmd != "write":
        _save(path, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
