"""Automatic white balance: Raspberry Pi's Bayesian search over the tuning's grey curve.

Ported from libcamera, src/ipa/rpi/controller/rpi/awb.cpp and awb_bayes.cpp:
Copyright (C) 2025, Raspberry Pi Ltd, BSD-2-Clause (twinrobo/isp/tunings/LICENSE).

It runs on 32x32 zone statistics. They are read to the CPU: the search is a few hundred small
evaluations, faster there than on a GPU.
"""

from __future__ import annotations

import numpy as np

#: libcamera's defaults, and PiSP's 32x32 AWB regions.
DELTA_LIMIT, COARSE_STEP, MIN_G, ZONES = 0.2, 0.2, 32, 32


def _pwl(t: float, x: np.ndarray, y: np.ndarray) -> float:
    """ipa::Pwl::eval: linear, extended past the end points."""
    if t <= x[0]:
        return float(y[0] + (y[1] - y[0]) * (t - x[0]) / (x[1] - x[0]))
    if t >= x[-1]:
        return float(y[-1] + (y[-1] - y[-2]) * (t - x[-1]) / (x[-1] - x[-2]))
    return float(np.interp(t, x, y))


def _prior_at(priors: list[dict], lux: float) -> np.ndarray:
    """The AWB prior (log likelihood against color temperature) interpolated at ``lux``."""
    pri = [(p["lux"], np.array(p["prior"], float).reshape(-1, 2)) for p in priors]
    if lux <= pri[0][0]:
        return pri[0][1]
    if lux >= pri[-1][0]:
        return pri[-1][1]
    k = max(i for i, (level, _) in enumerate(pri) if level <= lux)
    (l0, p0), (l1, p1) = pri[k], pri[k + 1]
    xs = np.union1d(p0[:, 0], p1[:, 0])
    a, b = np.interp(xs, *p0.T), np.interp(xs, *p1.T)
    return np.stack([xs, a + (b - a) * (lux - l0) / (l1 - l0)], 1)


def _vertex(a, b, c) -> float:
    """interpolateQuadatric: the extremum of the parabola through 3 points, within them."""
    eps = 1e-3
    cax, cay, bax, bay = c[0] - a[0], c[1] - a[1], b[0] - a[0], b[1] - a[1]
    den = 2 * (bay * cax - cay * bax)
    if abs(den) > eps:
        return max(a[0], min(c[0], (bay * cax * cax - cay * bax * bax) / den + a[0]))
    return a[0] if a[1] < c[1] - eps else (c[0] if c[1] < a[1] - eps else b[0])


def bayes_awb(zones: np.ndarray, tuning, lux: float) -> tuple[float, float, float]:
    """``(color temperature, red gain, blue gain)`` from ``[ZONES**2, 3]`` raw zone means (16-bit).

    Without zones bright enough to measure, the prior alone decides.
    """
    awb = tuning.awb
    curve = tuning.ct_curve
    lo, hi = awb["modes"]["auto"]["lo"], awb["modes"]["auto"]["hi"]
    trans_neg, trans_pos = awb["transverse_neg"], awb["transverse_pos"]
    zone = zones[zones[:, 1] >= MIN_G]
    zr, zb = zone[:, 0] / (zone[:, 1] + 1), zone[:, 2] / (zone[:, 1] + 1)
    pri = _prior_at(awb["priors"], lux)
    pscale = len(zone) / float(ZONES * ZONES)

    def ct_r(t):
        return _pwl(t, curve[:, 0], curve[:, 1])

    def ct_b(t):
        return _pwl(t, curve[:, 0], curve[:, 2])

    def prior(t):
        return pscale * float(np.interp(t, pri[:, 0], pri[:, 1]))

    def d2(gr, gb):
        return float(np.minimum((gr * zr - 1) ** 2 + (gb * zb - 1) ** 2, DELTA_LIMIT).sum())

    pts, t = [], lo
    while True:  # coarse search along the curve
        pts.append((t, d2(1 / ct_r(t), 1 / ct_b(t)) - prior(t)))
        if t == hi:
            break
        t = min(t + t / 10 * COARSE_STEP, hi)
    bp = max(1, min(int(np.argmin([p[1] for p in pts])), len(pts) - 2))
    t = _vertex(pts[bp - 1], pts[bp], pts[bp + 1])
    # fine search: along the curve and across it
    step, nsteps = t / 10 * COARSE_STEP * 0.1, 5
    tr = np.array(
        [
            ct_b(t + nsteps * step) - ct_b(t - nsteps * step),
            -(ct_r(t + nsteps * step) - ct_r(t - nsteps * step)),
        ]
    )
    tr = tr / np.linalg.norm(tr)
    span = trans_neg + trans_pos
    nd = int(min(12, max(3, np.floor(span * 100 + 0.5) + 1)))
    best = None
    for i in range(-(nsteps + nd), nsteps + nd + 1):
        tt = t + i * step
        rc, bc = ct_r(tt), ct_b(tt)
        xs = [-trans_neg + span * j / (nd - 1) for j in range(nd)]
        ys = [d2(1 / (rc + tr[0] * x), 1 / (bc + tr[1] * x)) - prior(tt) for x in xs]
        j = max(1, min(int(np.argmin(ys)), nd - 2))
        x = _vertex((xs[j - 1], ys[j - 1]), (xs[j], ys[j]), (xs[j + 1], ys[j + 1]))
        r, b = rc + tr[0] * x, bc + tr[1] * x
        ll = d2(1 / r, 1 / b) - prior(tt)
        if best is None or ll < best[0]:
            best = (ll, tt, r, b)
    _, t, r, b = best
    return float(t), float(1 / r), float(1 / b)
