"""A real sensor from its libcamera tuning: auto-exposure, Bayer mosaic, noise and ADC."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import SensorModel
from .cfa import mosaic_rggb
from .noise import add_noise, noise_std
from .response import adc

if TYPE_CHECKING:
    from ..isp.tuning import Tuning

#: Luma weights of libcamera's AGC statistics.
AGC_LUMA = (0.299, 0.587, 0.114)
#: libcamera's AGC filter speed, per frame.
AGC_SPEED = 0.2
#: Below this metered luma a frame has nothing to expose for: it keeps the last exposure.
AGC_MIN_LUMA = 1e-3


@dataclass
class _Stream:
    """One stream's state: a twin, and so its sensor, serves all of its cameras."""

    exposure: Tensor | None = None
    generator: torch.Generator | None = None


class TunedSensor(SensorModel):
    """Scene light -> a Bayer raw image, as this sensor would record it.

    ``capture`` runs the camera's auto-exposure: centre-weighted luma to the tuning's target,
    the mean of the brightest pixels at least at its highlight level, filtered from frame to
    frame. Its
    ``exposure`` argument multiplies the result. ``raw`` is ``[B, 1, H, W]`` RGGB, 0 to 1 of
    full scale above black. ``metadata["sensor"]`` reports what was applied.

    Each stream of frames keeps its own exposure and noise sequence: ``metadata["stream"]``,
    which simulator adapters set (see `new_stream`), else ``metadata["camera"]``. `reset`
    starts a stream again. The same ``seed`` and order of creation reproduce its noise. In a
    batch, a position is one camera; a change of batch size restarts the exposure.

    ``metadata["illumination"]`` ``[C, H, W]``, the relative illumination that lens-shading
    correction removed, gives corners the higher noise a corrected image has there.

    Simulator images have no absolute light level, so exposure is not limited by a shutter
    and the gain does not follow the scene: set ``analogue_gain`` and the ISP's ``lux`` for
    the light level you simulate.

    The sensor's spectral sensitivities are not public, so scene colors enter the raw image
    through the inverse of the tuning's daylight color matrix and white balance.
    A frame rendered smaller than the native sensor is a downscaled readout: its noise falls
    with the square root of the pixels averaged.

    Args:
        tuning: A `twinrobo.isp.tuning.Tuning`.
        analogue_gain: Sensor gain (the default suits a room of about 300 lux).
        seed: Seed of the noise.
        noise: ``False`` leaves noise out (the mosaic, exposure and ADC remain).
    """

    output_format = "rggb"

    def __init__(
        self, tuning: Tuning, analogue_gain: float = 2.0, seed: int = 0, noise: bool = True
    ):
        self.tuning = tuning
        self.analogue_gain = float(analogue_gain)
        self.seed = int(seed)
        self.noise = noise
        self._streams: dict[Any, _Stream] = {}
        grey_r, grey_b = tuning.grey(6500.0)
        self._unprocess = np.diag([grey_r, 1.0, grey_b]) @ np.linalg.inv(tuning.ccm(6500.0))

    def reset(self, stream: Any = None) -> None:
        if stream is None:
            self._streams.clear()
        else:
            self._streams.pop(stream, None)

    def _auto_exposure(self, rgb: Tensor, stream: _Stream) -> Tensor:
        """The exposure of each image in the batch, ``[B]``."""
        t = self.tuning
        y = (rgb * rgb.new_tensor(AGC_LUMA)[None, :, None, None]).sum(1, keepdim=True).float()
        zones = F.adaptive_avg_pool2d(y, t.metering.shape)
        weights = y.new_tensor(t.metering)
        metered = (zones[:, 0] * weights).sum((1, 2)) / weights.sum()
        flat, _ = y.flatten(1).sort(dim=1)
        n = flat.shape[1]
        lo = min(int(t.highlight_q[0] * n), n - 1)
        hi = max(int(np.ceil(t.highlight_q[1] * n)), lo + 1)
        highlight = flat[:, lo:hi].mean(1)
        want = torch.maximum(
            t.y_target / metered.clamp_min(1e-6), t.highlight_y / highlight.clamp_min(1e-6)
        )
        old = stream.exposure
        fresh = old is None or old.shape != want.shape or old.device != want.device
        want = torch.where(
            metered < AGC_MIN_LUMA, torch.ones_like(want) if fresh else old, want
        )  # black frames, such as a covered lens, say nothing about the next frame
        if fresh:
            stream.exposure = want
        else:
            stream.exposure = (1 - AGC_SPEED) * old + AGC_SPEED * want
        return stream.exposure

    def capture(
        self, irradiance: Tensor, exposure: float = 1.0, metadata: dict[str, Any] | None = None
    ) -> Tensor:
        H, W = irradiance.shape[-2:]
        t = self.tuning
        meta = metadata or {}
        name = meta.get("stream", meta.get("camera"))
        stream = self._streams.setdefault(name, _Stream())
        auto = self._auto_exposure(irradiance, stream)
        light = irradiance.float() * (auto * exposure)[:, None, None, None]
        unprocess = light.new_tensor(self._unprocess)
        raw = torch.einsum("ij,bjhw->bihw", unprocess, light).clamp_min(0)
        raw = mosaic_rggb(raw) * t.full_scale
        if self.noise:
            if stream.generator is None or stream.generator.device != raw.device:
                seed = self.seed + zlib.crc32(str(name).encode())
                stream.generator = torch.Generator(raw.device).manual_seed(seed)
            shade = meta.get("illumination")
            shade = 1.0 if shade is None else mosaic_rggb(shade[None]).clamp_min(0.02)
            std = noise_std(raw * shade, t.noise_constant, t.noise_slope, self.analogue_gain)
            raw = add_noise(raw, std / shade * self.noise_scale(H, W), stream.generator)
        out = adc(raw, t.black_level, t.bits) / t.full_scale
        if metadata is not None:
            metadata["sensor"] = {
                "tuning": t.name,
                "exposure": auto.tolist(),
                "analogue_gain": self.analogue_gain,
                "noise_scale": self.noise_scale(H, W),
            }
        return out

    def noise_scale(self, height: int, width: int) -> float:
        """Noise of a ``width x height`` readout, relative to a native pixel's."""
        if self.tuning.native_size is None:
            return 1.0
        w, h = self.tuning.native_size
        return min(width * height / (w * h), 1.0) ** 0.5
