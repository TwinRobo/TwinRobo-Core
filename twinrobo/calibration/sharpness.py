"""Sharpness from slanted edges (ISO 12233 style), and the lens focus that explains it.

A slanted edge is a straight dark/bright boundary tilted a few degrees off the
pixel grid. Every row samples the edge at a slightly different sub-pixel phase, so
projecting all pixels onto the edge normal gives the edge spread function (ESF)
at 4x the pixel rate. Its derivative is the line spread function (LSF), and the
LSF's Fourier magnitude is the MTF. MTF50, the frequency where contrast halves,
is the usual single-number sharpness.

`fit_focus` compares measured MTF curves at known object distances with the lens
model (DeepLens) refocused to candidate distances, and returns the focus distance
that explains the captures best: the one number of the blur model a catalog entry
cannot take from a datasheet when the camera's focus is fixed and unpublished.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EdgeMTF:
    """MTF of one slanted edge. ``freq`` in cycles per pixel (Nyquist = 0.5)."""

    freq: np.ndarray
    mtf: np.ndarray
    angle_deg: float  # edge tilt from the pixel axis it is closest to
    vertical: bool  # True: a near-vertical edge (MTF along x)

    @property
    def mtf50(self) -> float:
        return mtf_at(self.freq, self.mtf, 0.5)


def mtf_at(freq: np.ndarray, mtf: np.ndarray, level: float) -> float:
    """First frequency where ``mtf`` falls to ``level`` (linear interpolation)."""
    below = np.nonzero(mtf < level)[0]
    if len(below) == 0:
        return float(freq[-1])
    i = int(below[0])
    if i == 0:
        return 0.0
    f0, f1, m0, m1 = freq[i - 1], freq[i], mtf[i - 1], mtf[i]
    return float(f0 + (m0 - level) * (f1 - f0) / (m0 - m1))


def slanted_edge_mtf(roi: np.ndarray, oversample: int = 4) -> EdgeMTF:
    """MTF of the single slanted edge in ``roi`` (grayscale or RGB, linear or sRGB).

    The ROI should hold one straight edge, tilted about 2-15 degrees from vertical
    or horizontal, with some flat dark and bright area on both sides. Use linear
    (not gamma-encoded) pixel values when available; sRGB images overstate MTF a
    little.
    """
    x = np.asarray(roi, dtype=np.float64)
    if x.ndim == 3:
        x = x[..., :3] @ np.array([0.2126, 0.7152, 0.0722])  # luminance
    gy, gx = np.gradient(x)
    vertical = np.abs(gx).sum() >= np.abs(gy).sum()
    if not vertical:
        x = x.T  # measure a near-horizontal edge as a near-vertical one
    h, w = x.shape
    d = np.abs(np.diff(x, axis=1))  # per-row derivative
    cols = np.arange(w - 1) + 0.5
    weight = d.sum(1)
    ok = weight > 1e-9
    if ok.sum() < 8:
        raise ValueError("no edge found in the ROI")
    centroid = (d * cols).sum(1)[ok] / weight[ok]
    rows = np.arange(h)[ok]
    slope, intercept = np.polyfit(rows, centroid, 1)  # edge: x = slope * y + intercept
    angle = math.degrees(math.atan(slope))
    if abs(angle) < 1.0:
        raise ValueError(f"edge tilted {angle:.2f} deg; tilt it 2-15 deg off the pixel grid")
    # signed distance of every pixel center to the edge line, along the edge normal
    yy, xx = np.mgrid[0:h, 0:w]
    dist = (xx - (slope * yy + intercept)) * math.cos(math.atan(slope))
    bins = np.round(dist * oversample).astype(int)
    lo = bins.min()
    count = np.bincount((bins - lo).ravel())
    total = np.bincount((bins - lo).ravel(), weights=x.ravel())
    have = count > 0
    esf = np.interp(np.arange(len(count)), np.nonzero(have)[0], total[have] / count[have])
    if esf[-1] < esf[0]:
        esf = esf[::-1]  # dark -> bright
    lsf = np.diff(esf)
    peak = int(np.argmax(lsf))
    n = len(lsf)
    window = 0.54 + 0.46 * np.cos(np.pi * (np.arange(n) - peak) / max(peak, n - peak))
    spectrum = np.abs(np.fft.rfft(lsf * window))
    mtf = spectrum / spectrum[0]
    freq = np.fft.rfftfreq(n, d=1.0 / oversample)  # cycles per pixel
    keep = freq <= 1.0
    return EdgeMTF(freq[keep], mtf[keep], abs(angle), bool(vertical))


def psf_mtf(psf: np.ndarray, vertical_edge: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """MTF of a sampled PSF ``[ks, ks]`` across a vertical (or horizontal) edge."""
    p = np.asarray(psf, dtype=np.float64)
    lsf = p.sum(0) if vertical_edge else p.sum(1)
    n = len(lsf)
    pad = np.zeros(max(256, n))
    pad[:n] = lsf
    spectrum = np.abs(np.fft.rfft(pad))
    return np.fft.rfftfreq(len(pad)), spectrum / spectrum[0]


#: Frequencies (cycles per pixel) at which edge MTF curves are stored and compared.
FREQS = np.linspace(0.0, 1.0, 41)


def sample_mtf(freq: np.ndarray, mtf: np.ndarray, at: np.ndarray = FREQS) -> list[float]:
    """An MTF curve resampled at ``at`` (0 beyond the measured range)."""
    return [float(v) for v in np.interp(at, freq, mtf, right=0.0)]


@dataclass
class EdgeMeasurement:
    """A measured edge: where it was, and its MTF.

    ``field``: the ROI center in DeepLens' normalized field coordinates (``[-1, 1]``,
    ``+y`` up); ``distance_m``: the target's distance from the camera; ``mtf``: the
    edge's MTF at `FREQS`, in cycles per pixel **of the spec's resolution**.
    """

    distance_m: float
    field: tuple[float, float]
    mtf: list[float]
    vertical: bool = True
    source: str | None = None

    @property
    def mtf50(self) -> float:
        return mtf_at(FREQS, np.asarray(self.mtf), 0.5)

    @staticmethod
    def field_of(
        roi_center_px: tuple[float, float], width: int, height: int
    ) -> tuple[float, float]:
        cx, cy = roi_center_px
        return ((cx + 0.5) / width * 2 - 1, -((cy + 0.5) / height * 2 - 1))

    def to_json(self) -> dict:
        return {**asdict(self), "mtf50": self.mtf50}


def model_mtf(optics, distance_m: float, edges: list[EdgeMeasurement]) -> list[np.ndarray]:
    """The lens model's MTF (green) at `FREQS` for edges at one distance, at its current focus."""
    green = optics.wavelengths_rgb_um[len(optics.wavelengths_rgb_um) // 2]
    psf = optics.generate_psf([e.field for e in edges], [distance_m], wavelengths=[green])[0]
    out = []
    for e, p in zip(edges, psf, strict=True):
        f, mtf = psf_mtf(p[0].detach().cpu().numpy(), e.vertical)
        out.append(np.interp(FREQS, f, mtf, right=0.0))
    return out


def _curve_error(optics, measurements: list[EdgeMeasurement], nyquist_only: bool = True) -> float:
    by_distance: dict[float, list[EdgeMeasurement]] = {}
    for m in measurements:
        by_distance.setdefault(m.distance_m, []).append(m)
    keep = FREQS <= 0.5 if nyquist_only else slice(None)
    err = []
    for d, ms in by_distance.items():
        for m, curve in zip(ms, model_mtf(optics, d, ms), strict=True):
            err.append((curve - np.asarray(m.mtf))[keep])
    return float(np.sqrt(np.mean(np.square(np.concatenate(err)))))


@dataclass
class FocusFit:
    focus_m: float
    rms_error: float  # MTF error over 0-0.5 cycles/px, all edges
    table: list[dict]  # evaluated candidates: {"focus_m", "rms_error"}
    per_edge: list[dict]  # at the best focus: measured vs model MTF50 per edge


def fit_focus(optics, measurements: list[EdgeMeasurement], coarse: int = 15) -> FocusFit:
    """The focus distance at which the lens model's MTF best matches the measured edges.

    ``optics`` is a `twinrobo.optics.deeplens.DeepLensOptics`; it is refocused while
    fitting and left at the best focus. The search is in diopters (1/m), 10 cm to
    infinity: a coarse grid, then a golden-section refinement around its best point.
    Measure edges at several distances in front of and behind the expected focus: one
    distance cannot tell near from far focus.
    """
    if len({m.distance_m for m in measurements}) < 2:
        raise ValueError("measure edges at two or more distances to fit the focus")
    table: dict[float, float] = {}

    def cost(diopters: float) -> float:
        d = round(float(diopters), 6)
        if d not in table:
            optics.set_focus(1 / d if d > 0 else float("inf"))
            table[d] = _curve_error(optics, measurements)
        return table[d]

    grid = np.linspace(0.0, 10.0, coarse)
    best = min(grid, key=cost)
    step = grid[1] - grid[0]
    lo, hi = max(0.0, best - step), min(10.0, best + step)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(12):
        if cost(a) < cost(b):
            hi, b = b, a
            a = hi - g * (hi - lo)
        else:
            lo, a = a, b
            b = lo + g * (hi - lo)
    d_best = min(table, key=table.get)
    focus = 1 / d_best if d_best > 0 else float("inf")
    optics.set_focus(focus)
    per_edge = []
    for m in measurements:
        (curve,) = model_mtf(optics, m.distance_m, [m])
        per_edge.append({**m.to_json(), "model_mtf50": mtf_at(FREQS, curve, 0.5)})
    rows = [
        {"focus_m": 1 / d if d > 0 else float("inf"), "rms_error": e}
        for d, e in sorted(table.items())
    ]
    return FocusFit(focus, table[d_best], rows, per_edge)
