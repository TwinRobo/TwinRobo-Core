"""Fit a catalog camera's geometry to its datasheet fields of view.

Datasheets give a horizontal, vertical and diagonal FoV, usually through the lens'
distortion. This finds square-pixel intrinsics (``fx = fy``, centered principal
point) and, for wide lenses, OpenCV ``k1, k2`` that reproduce them
(`twinrobo.optics.distortion.field_of_view`). With ``--pinhole`` only ``fx`` is fitted
(least squares on H and V), for lenses whose datasheet distortion is ~1 % or less.
With ``--fx`` the focal length is held (e.g. from the published focal length and
pixel pitch) and only the distortion is fitted; ``--k3`` adds a third radial term.

Check which sensor area a datasheet's FoV refers to: many stereo cameras quote it
for the full array while streaming a center crop (same ``fx`` and distortion, a
smaller FoV).

    python tools/catalog/fit_fov_geometry.py 1920 1200 110 80 120          # ZED X 2.2 mm
    python tools/catalog/fit_fov_geometry.py 1920 1080 69.4 42.5 77 --pinhole
    python tools/catalog/fit_fov_geometry.py 4608 2592 102 67 120 --fx 1964.29 --k3
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import least_squares

from twinrobo.geometry import CameraIntrinsics
from twinrobo.optics.distortion import RadialDistortion, field_of_view


def fit(
    width: int,
    height: int,
    hfov: float,
    vfov: float,
    dfov: float | None,
    pinhole: bool = False,
    fx: float | None = None,
    k3: bool = False,
) -> dict:
    """Square-pixel ``fx`` (unless given) and ``k1 k2 [k3]`` reproducing the datasheet FoVs."""
    cx, cy = (width - 1) / 2, (height - 1) / 2
    target = np.array([hfov, vfov] + ([dfov] if dfov and not pinhole else []))
    nk = 0 if pinhole else (3 if k3 else 2)

    def unpack(p):
        f, ks = (fx, p) if fx is not None else (p[0], p[1:])
        ks = list(ks) + [0.0] * (3 - len(ks))
        return f, ks

    def fov(p):
        f, ks = unpack(p)
        dist = None if nk == 0 else RadialDistortion(k1=ks[0], k2=ks[1], k3=ks[2])
        v = field_of_view(CameraIntrinsics(width, height, f, f, cx, cy), dist)
        return np.array([v["h"], v["v"]] + ([v["d"]] if len(target) == 3 else []))

    fx0 = width / 2 / math.tan(math.radians(hfov) / 2)
    p0 = ([] if fx is not None else [fx0]) + [0.0] * nk
    if not p0:
        raise ValueError("nothing to fit: --fx with --pinhole")

    def monotonic(p):
        """Penalty keeping r_d(r) = r (1 + k1 r^2 + k2 r^4 + k3 r^6) increasing out to past the
        sensor corner: a real lens never folds the image (and the spec must stay invertible)."""
        if nk == 0:
            return np.zeros(1)
        f, ks = unpack(p)
        r_corner = math.hypot(width / 2, height / 2) / f
        r = np.linspace(0, 2.5 * r_corner, 200)  # the undistorted radius can exceed the corner's
        r2 = r * r
        dr = 1 + 3 * ks[0] * r2 + 5 * ks[1] * r2**2 + 7 * ks[2] * r2**3
        rd = r * (1 + ks[0] * r2 + ks[1] * r2**2 + ks[2] * r2**3)
        reach = rd <= 1.02 * r_corner  # the part of the curve the sensor uses
        return np.array([100.0 * max(0.0, 0.05 - float(dr[reach].min()))])

    res = least_squares(lambda p: fov(p) - target, p0, x_scale="jac")
    if monotonic(res.x)[0] > 0:  # the best fit folds the image: fit again, constrained
        res = least_squares(
            lambda p: np.concatenate([fov(p) - target, monotonic(p)]), p0, x_scale="jac"
        )
    got = fov(res.x)
    f, ks = unpack(res.x)
    out = {"fx": round(float(f), 3), "fy": round(float(f), 3), "cx": cx, "cy": cy}
    for name, k in zip(("k1", "k2", "k3"), ks[:nk], strict=False):
        out[name] = round(float(k), 5)
    out["fov_fit"] = {k: round(float(v), 2) for k, v in zip("hvd", got, strict=False)}
    out["fov_target"] = {k: float(v) for k, v in zip("hvd", target, strict=False)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("hfov", type=float)
    ap.add_argument("vfov", type=float)
    ap.add_argument("dfov", type=float, nargs="?")
    ap.add_argument("--pinhole", action="store_true", help="fit fx only (no distortion)")
    ap.add_argument("--fx", type=float, help="hold fx (px) and fit only the distortion")
    ap.add_argument("--k3", action="store_true", help="also fit k3")
    a = ap.parse_args()
    out = fit(a.width, a.height, a.hfov, a.vfov, a.dfov, a.pinhole, a.fx, a.k3)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
