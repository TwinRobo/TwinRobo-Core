"""Raspberry Pi camera tunings: what a sensor and its ISP are calibrated to do.

The tuning files are libcamera's public ones (``tunings/``, BSD-2-Clause, see its LICENSE),
measured by Raspberry Pi on each sensor: black level, noise profile, auto-exposure targets,
white-balance curves and priors, color matrices, tone curve, denoise and sharpen strengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..sensor.response import FULL_SCALE

TUNINGS = Path(__file__).parent / "tunings"

#: Native size and ADC depth of each bundled sensor: (width, height, bits).
SENSORS = {
    "imx708": (4608, 2592, 10),
    "imx708_wide": (4608, 2592, 10),
    "imx219": (3280, 2464, 10),
    "imx477": (4056, 3040, 12),
    "imx296": (1456, 1088, 12),
    "ov5647": (2592, 1944, 10),
    "imx500": (4056, 3040, 10),
}


@dataclass(frozen=True)
class Tuning:
    """The parts of a libcamera tuning that the sensor and ISP use.

    Levels are libcamera's: 16-bit, black level included in ``black_level``.
    """

    name: str
    fingerprint: str  # SHA-256 of the file: the same name can hold another tuning
    native_size: tuple[int, int] | None
    bits: int
    black_level: float
    noise_constant: float
    noise_slope: float
    y_target: float  # AGC: metered luma, at the low-lux end (indoor)
    metering: np.ndarray  # centre-weighted zone weights, square
    highlight_q: tuple[float, float]  # AGC: the mean luma of these quantiles ...
    highlight_y: float  # ... reaches at least this level
    ct_curve: np.ndarray  # [N, 3]: color temperature, r/g and b/g of a grey
    ccms: tuple[tuple[float, np.ndarray], ...]  # (color temperature, 3x3)
    awb: dict[str, Any]  # rpi.awb: priors, search range, transverse limits
    tone_curve: np.ndarray  # [N, 2]: 16-bit in, 16-bit out
    sdn_deviation: float
    sdn_strength: float
    cdn_strength: float
    sharpen_threshold: float
    sharpen_strength: float
    sharpen_limit: float

    @property
    def full_scale(self) -> float:
        """The largest signal above black, 16-bit."""
        return FULL_SCALE - self.black_level

    def grey(self, ct: float) -> tuple[float, float]:
        """r/g, b/g of a grey at color temperature ``ct`` (linear between the curve's points)."""
        c = self.ct_curve
        return float(np.interp(ct, c[:, 0], c[:, 1])), float(np.interp(ct, c[:, 0], c[:, 2]))

    def ccm(self, ct: float) -> np.ndarray:
        """The color matrix at ``ct``, interpolated between the tuning's."""
        cts = [c for c, _ in self.ccms]
        ct = float(np.clip(ct, cts[0], cts[-1]))
        k = max(0, min(len(cts) - 2, int(np.searchsorted(cts, ct)) - 1))
        w = (ct - cts[k]) / (cts[k + 1] - cts[k])
        return (1 - w) * self.ccms[k][1] + w * self.ccms[k + 1][1]


def list_tunings() -> list[str]:
    """Names of the bundled tunings."""
    return sorted(p.stem for p in TUNINGS.glob("*.json"))


def load_tuning(
    name: str | Path, native_size: tuple[int, int] | None = None, bits: int | None = None
) -> Tuning:
    """A bundled tuning by name (``"imx708"``), or a Raspberry Pi libcamera tuning by path.

    Files need the ``rpi.*`` algorithms of the bundled color tunings. A tuning does not hold
    the sensor's ``native_size`` or ``bits``: bundled ones have theirs, and a file by path
    needs them given, or it gets 10 bits and no noise scaling for smaller frames.
    """
    import hashlib
    import json

    path = TUNINGS / f"{name}.json"
    if not path.exists():
        path = Path(name)
    if not path.exists():
        raise ValueError(f"no tuning {name!r}; bundled: {', '.join(list_tunings())}")
    text = path.read_bytes()
    raw = json.loads(text)
    try:
        algo = {k: v for a in raw["algorithms"] for k, v in a.items()}
        agc = algo["rpi.agc"]["channels"][0]
        weights = np.array(agc["metering_modes"]["centre-weighted"]["weights"], float)
        constraint = agc["constraint_modes"]["normal"][0]
        denoise = algo["rpi.denoise"]["normal"]
        sharpen = algo["rpi.sharpen"]
        ct_curve = np.array(algo["rpi.awb"]["ct_curve"], float).reshape(-1, 3)
        algo["rpi.awb"]["priors"]
        ccms = tuple(
            (float(c["ct"]), np.array(c["ccm"], float).reshape(3, 3))
            for c in algo["rpi.ccm"]["ccms"]
        )
        n = round(weights.size**0.5)
        known = SENSORS.get(path.stem) if path.parent.resolve() == TUNINGS.resolve() else None
        size = known[:2] if known else None
        bits = bits if bits is not None else known[2] if known else 10
        native_size = tuple(native_size) if native_size is not None else size
        return Tuning(
            name=path.stem,
            fingerprint=hashlib.sha256(text).hexdigest(),
            native_size=native_size,
            bits=bits,
            black_level=float(algo["rpi.black_level"]["black_level"]),
            noise_constant=float(algo["rpi.noise"]["reference_constant"]),
            noise_slope=float(algo["rpi.noise"]["reference_slope"]),
            y_target=float(np.array(agc["y_target"]).reshape(-1, 2)[0, 1]),
            metering=weights.reshape(n, n),
            highlight_q=(float(constraint["q_lo"]), float(constraint["q_hi"])),
            highlight_y=float(np.array(constraint["y_target"]).reshape(-1, 2)[0, 1]),
            ct_curve=ct_curve,
            ccms=ccms,
            awb=algo["rpi.awb"],
            tone_curve=np.array(algo["rpi.contrast"]["gamma_curve"], float).reshape(-1, 2),
            sdn_deviation=float(denoise["sdn"]["deviation_no_tdn"]),
            sdn_strength=float(denoise["sdn"]["strength_no_tdn"]),
            cdn_strength=float(denoise["cdn"]["strength"]),
            sharpen_threshold=float(sharpen["threshold"]),
            sharpen_strength=float(sharpen["strength"]),
            sharpen_limit=float(sharpen["limit"]),
        )
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise ValueError(f"{path}: not a full color tuning, missing {e}") from e
