"""Relative illumination (vignetting) from flat-field captures.

Point the camera at a uniformly lit, featureless surface (a diffuser over the lens,
a light panel or an evenly lit white wall), defocus if needed, and average a few
frames at a mid-level exposure. The falloff towards the corners is the camera's
relative illumination: lens vignetting, cos^4 falloff and the sensor's chief-ray
angle response together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Vignetting:
    """Radial relative illumination ``I(r) = 1 + a2 r^2 + a4 r^4 + a6 r^6``.

    ``r`` is the distance from the image center over the half-diagonal (1 at the
    corners); ``corner`` is ``I(1)``.
    """

    a2: float
    a4: float
    a6: float
    corner: float
    rms_residual: float
    width: int
    height: int

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r2 = np.square(r)
        return 1 + self.a2 * r2 + self.a4 * r2**2 + self.a6 * r2**3

    def to_spec(self) -> dict:
        return {
            "model": "radial_poly",
            "radius": "half_diagonal",
            "coeffs": [round(float(v), 6) for v in (self.a2, self.a4, self.a6)],
            "corner": round(float(self.corner), 4),
        }


def flat_field(
    images: list[np.ndarray], black_level: float = 0.0, saturation: float | None = None
) -> Vignetting:
    """Fit relative illumination to flat-field images (averaged; linear values preferred).

    The center is normalized to 1. Pixels at or above ``saturation`` (default: the
    integer type's maximum for integer images, none for float images) are ignored;
    keep the exposure below clipping.
    """
    if saturation is None and np.issubdtype(np.asarray(images[0]).dtype, np.integer):
        saturation = float(np.iinfo(np.asarray(images[0]).dtype).max)
    stack = np.stack([np.asarray(im, dtype=np.float64) for im in images])
    if stack.ndim == 4:
        stack = stack[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    img = stack.mean(0) - black_level
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / np.hypot(w / 2, h / 2)
    center = np.median(img[r < 0.05])
    if center <= 0:
        raise ValueError("the image center is black; check exposure and black_level")
    rel = img / center
    ok = (stack.max(0) < saturation) if saturation is not None else np.ones_like(img, bool)
    rr, vv = r[ok].ravel(), rel[ok].ravel()
    if len(rr) > 400_000:  # a random subset is plenty for three coefficients
        idx = np.random.default_rng(0).choice(len(rr), 400_000, replace=False)
        rr, vv = rr[idx], vv[idx]
    r2 = rr**2
    A = np.stack([r2, r2**2, r2**3], 1)
    coeffs, *_ = np.linalg.lstsq(A, vv - 1, rcond=None)
    model = Vignetting(*coeffs, 0.0, 0.0, w, h)
    model.corner = float(model(np.array(1.0)))
    model.rms_residual = float(np.sqrt(np.mean((model(rr) - vv) ** 2)))
    return model
