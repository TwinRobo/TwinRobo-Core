"""Cross-check TwinRobo's lens file import and export against RayOptics, an independent tracer.

TwinRobo exports each lens to ``.zmx`` and ``.seq``, reads those files back and traces real
rays through them with DeepLens. RayOptics (https://github.com/mjhoptics/ray-optics) reads the
same files with its own readers, glass catalogs and tracer, and traces the same rays. Agreement
means both programs read the files the same way; it is not a comparison with OpticStudio or
CODE V themselves.

RayOptics needs its own environment (it pins GUI packages TwinRobo does not use)::

    python tools/validate/lensio_rayoptics.py export OUT            # TwinRobo's environment
    /path/to/rayoptics-env/bin/python tools/validate/lensio_rayoptics.py compare OUT

``export`` writes the lens files and TwinRobo's results to ``OUT``; ``compare`` traces them in
RayOptics and prints the differences.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LENSES = {
    "triplet": ROOT / "tests/data/lensio/triplet.csv",
    **{
        str(p.parent.relative_to(ROOT / "twinrobo/catalog")).replace("/", "_"): p
        for p in sorted((ROOT / "twinrobo/catalog").glob("**/lens.json"))
    },
}
FIELDS = (0.0, 0.5, 1.0)  # fraction of the edge field angle
PUPIL = (-0.7, -0.35, 0.0, 0.35, 0.7)  # fraction of the entrance pupil radius
START = 10.0  # rays start this far (mm) in front of the first vertex or the entrance pupil
#: Agreement required (um): image heights, and focal lengths (DeepLens' EFL vs RayOptics').
TOL_Y_UM = 0.1
TOL_EFL_UM = 0.1


def rays_for(par, half_field_deg: float):
    """Rays as (y0, z0, angle_deg): aimed at points of the paraxial entrance pupil."""
    z0 = min(par.enp, 0.0) - START
    out = []
    for f in FIELDS:
        a = f * half_field_deg
        for py in PUPIL:
            y_ep = py * par.epd / 2
            out.append((y_ep + math.tan(math.radians(a)) * (z0 - par.enp), z0, a))
    return out


def export(out: Path) -> None:
    import torch

    from twinrobo import lensio
    from twinrobo.lensio.deeplens_json import indices
    from twinrobo.lensio.materials import resolve_all
    from twinrobo.lensio.paraxial import paraxial
    from twinrobo.optics.deeplens import DeepLensOptics

    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, src in LENSES.items():
        p = lensio.load_lens(src)
        cats = lensio.catalogs_for(p)
        par = paraxial(p, indices(p, resolve_all(p, cats)))
        lens_fov = math.degrees(math.atan(par.image_height / abs(par.efl))) if par.efl else 0
        half = max((abs(y) for _, y in p.fields), default=0.0) or lens_fov
        half = 0.8 * min(half, lens_fov) if p.field_type == "angle" else 0.8 * lens_fov
        rays = rays_for(par, half)
        q = lensio.materials.with_catalogs(p, cats)
        glasses = [
            [s.material.catalog, s.material.name] if s.material.catalog else None
            for s in q.surfaces
        ]
        entry = {"efl": par.efl, "bfl": par.bfl, "epd": par.epd, "rays": rays,
                 "glasses": glasses, "formats": {}}  # fmt: skip
        for ext in ("zmx", "seq"):
            f = out / f"{name}.{ext}"
            lensio.save_lens(p, f)
            o = DeepLensOptics(f, device="cpu", dtype=torch.float64)
            L = o._lens
            from deeplens.light import Ray

            w = float(p.primary_wavelength_um)
            org = torch.tensor([[0.0, y0, z0] for y0, z0, _ in rays], dtype=torch.float64)
            dirs = torch.tensor(
                [[0.0, math.sin(math.radians(a)), math.cos(math.radians(a))] for *_, a in rays],
                dtype=torch.float64,
            )
            r = L.trace2sensor(Ray(org, dirs, w, device="cpu"))
            entry["formats"][ext] = {
                "deeplens_efl": float(L.foclen),
                "y_image": [float(v) for v in r.o[:, 1]],
                "valid": [bool(v) for v in r.is_valid.bool()],
                "image_z": float(L.d_sensor),
            }
        results[name] = entry
        print(f"exported {name}")
    (out / "twinrobo.json").write_text(json.dumps(results, indent=1))


def compare(out: Path) -> int:
    import numpy as np
    from opticalglass.glassfactory import create_glass
    from rayoptics.environment import open_model
    from rayoptics.raytr import trace

    results = json.loads((out / "twinrobo.json").read_text())
    print(
        f"{'lens':<34} {'fmt':<4} {'EFL TwinRobo':>13} {'EFL RayOptics':>14} {'dEFL um':>8} "
        f"{'rays':>5} {'max |dy| um':>12}"
    )
    worst = 0.0
    missed: list[str] = []
    for name, entry in results.items():
        for ext, tw in entry["formats"].items():
            opm = open_model(str(out / f"{name}.{ext}"), info=False)
            sm = opm["seq_model"]
            # RayOptics' .zmx reader looks glasses up in its own catalog order, ignoring the
            # file's GCAT (OpticStudio searches GCAT's catalogs in order): put back the
            # catalog's glass where it chose another catalog's glass of the same name.
            swapped = []
            for g, want in zip(sm.gaps, entry["glasses"], strict=False):
                if want and not type(g.medium).__name__.upper().startswith(want[0]):
                    swapped.append(f"{g.medium.name()} {type(g.medium).__name__} -> {want}")
                    g.medium = create_glass(want[1], want[0])
            osp = opm["optical_spec"]
            opm.update_model()
            fod = opm["analysis_results"]["parax_data"].fod
            wvl = osp["wvls"].central_wvl  # nm, the reference wavelength
            # Start the rays on a plane START mm before the first vertex (or the pupil).
            z0 = entry["rays"][0][1]
            sm.gaps[0].thi = -z0
            opm.update_model()
            dys = []
            for (y0, _, a), y_tw, ok in zip(entry["rays"], tw["y_image"], tw["valid"], strict=True):
                if not ok:
                    continue
                d = np.array([0.0, math.sin(math.radians(a)), math.cos(math.radians(a))])
                try:
                    ray, _, _ = trace.trace(sm, np.array([0.0, y0, 0.0]), d, wvl)
                except Exception as e:  # noqa: BLE001 - RayOptics raises on a missed surface
                    missed.append(f"{name}.{ext} ray {y0:.4g} mm, {a:.3g} deg: {e}")
                    continue
                y_ro = ray[-1][0][1]
                dys.append(abs(y_ro - y_tw) * 1000.0)
            mx = max(dys) if dys else float("nan")
            worst = max(worst, mx if dys else 0.0)
            if not dys:
                missed.append(f"{name}.{ext}: no ray compared")
            elif not all(math.isfinite(d) for d in dys) or mx > TOL_Y_UM:
                missed.append(f"{name}.{ext}: image heights differ by {mx:.4g} um")
            d_efl = (tw["deeplens_efl"] - fod.efl) * 1000.0
            if not abs(d_efl) <= TOL_EFL_UM:  # also catches nan
                missed.append(f"{name}.{ext}: focal lengths differ by {d_efl:.4g} um")
            print(
                f"{name:<34} {ext:<4} {tw['deeplens_efl']:>13.6f} {fod.efl:>14.6f} "
                f"{d_efl:>8.3f} {len(dys):>5} {mx:>12.4f}"
                + (f"  (RayOptics glass replaced: {'; '.join(swapped)})" if swapped else "")
            )
    print(f"worst image-height difference: {worst:.4f} um")
    # A ray TwinRobo traces and RayOptics does not is a disagreement, not a skipped ray.
    print(f"tolerances: image height {TOL_Y_UM} um, focal length {TOL_EFL_UM} um")
    for m in missed:
        print(f"DISAGREE: {m}")
    return 1 if missed else 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("export", "compare"):
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    sys.exit(export(out) if sys.argv[1] == "export" else compare(out))
