"""Make a surrogate DeepLens lens for a catalog camera whose prescription is not published.

The source design is scaled uniformly so that its paraxial focal length equals
the camera's (defocus blur scales with f^2 / N, so matching f matters most),
its sensor matches the camera's sensor diagonal (DeepLens derives the pixel
pitch from it), and its stop is resized to the camera's f-number.

The surrogate models *blur* only. TwinRobo renders PSFs recentered on the
chief ray, and the camera's geometry (intrinsics + distortion) comes from the
catalog spec, so the surrogate's own field mapping does not leak into images.

Scaling by ``s``: lengths (``d``, ``d_next``, ``r``, ``roc``, sensor) x s,
curvature ``c`` / s, even aspheric terms ``a_2n`` x s^(1 - 2n), conic unchanged.
DeepLens' own JSON writer rounds to 4 decimals (which destroys the scaled
aspheric terms), so the JSON is scaled here and only the stop radius DeepLens
solves for is copied back.

Usage:
    python tools/catalog/make_surrogate_lens.py SRC.json OUT.json \
        --focal-mm 2.2368 --fnum 2.2 --sensor-diag-mm 6.7893 --info "..."
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

LENGTH_KEYS = ("d", "d_next", "r", "roc")


def scale_lens(lens: dict, s: float) -> dict:
    out = copy.deepcopy(lens)
    for k in ("foclen", "r_sensor", "d_sensor"):
        if k in out:
            out[k] = out[k] * s
    if "sensor_size" in out:
        out["sensor_size"] = [v * s for v in out["sensor_size"]]
    for surf in out["surfaces"]:
        for k in LENGTH_KEYS:
            if k in surf:
                surf[k] = surf[k] * s
        if "c" in surf:
            surf["c"] = surf["c"] / s
        if "pos_xy" in surf:
            surf["pos_xy"] = [v * s for v in surf["pos_xy"]]
        if "ai" in surf:
            # DeepLens: `ai` starts at a2 unless use_ai2 is explicitly False (then at a4).
            first = 2 if surf.get("use_ai2", True) else 4
            surf["ai"] = [a * s ** (1 - (first + 2 * i)) for i, a in enumerate(surf["ai"])]
            for i in range(2, 32, 2):
                if f"ai{i}" in surf:
                    surf[f"ai{i}"] = surf[f"ai{i}"] * s ** (1 - i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--focal-mm", type=float, required=True, help="target paraxial focal length")
    ap.add_argument("--fnum", type=float, required=True)
    ap.add_argument("--sensor-diag-mm", type=float, required=True)
    ap.add_argument("--info", default=None)
    args = ap.parse_args()

    import torch
    from deeplens import GeoLens

    src = json.loads(Path(args.src).read_text())
    with torch.no_grad():
        f0 = GeoLens(filename=args.src, device="cpu").foclen
    s = args.focal_mm / f0
    lens = scale_lens(src, s)
    r_sensor = args.sensor_diag_mm / 2
    side = args.sensor_diag_mm / math.sqrt(2)
    lens["r_sensor"], lens["sensor_size"] = r_sensor, [side, side]
    lens["info"] = args.info or f"{src.get('info', '')} scaled x{s:.5f}"
    out = Path(args.out)
    out.write_text(json.dumps(lens, indent=2))

    with torch.no_grad():
        g = GeoLens(filename=str(out), device="cpu")
        g.set_fnum(args.fnum)
        aper_r = float(g.surfaces[g.aper_idx].r)
    for surf in lens["surfaces"]:
        if surf["type"] == "Aperture":
            surf["r"] = aper_r
    lens["foclen"], lens["fnum"] = args.focal_mm, args.fnum
    out.write_text(json.dumps(lens, indent=2))

    with torch.no_grad():
        g = GeoLens(filename=str(out), device="cpu")
    print(
        json.dumps(
            {
                "scale": s,
                "foclen_mm": round(float(g.foclen), 4),
                "fnum": round(float(g.fnum), 3),
                "r_sensor_mm": g.r_sensor,
                "rfov_deg": round(math.degrees(float(g.rfov)), 2),
                "stop_radius_mm": round(aper_r, 5),
            }
        )
    )


if __name__ == "__main__":
    main()
