"""A real camera's ISP from its libcamera tuning.

The strengths, matrices and curves are the tuning's. The Pi's hardware kernels are not public:
standard methods with those strengths stand in for them (Malvar demosaic, bilateral denoise in
a variance-stabilised domain, chroma blur, thresholded unsharp mask). Lens shading correction
is not applied: render with ``shading="corrected"``, which gives the same flat image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .awb import ZONES, bayes_awb
from .base import ISPModel
from .color import srgb_to_linear
from .demosaic import demosaic_rggb
from .tone import bilateral, gaussian_blur, piecewise_linear, rgb_to_ycc, ycc_to_rgb

if TYPE_CHECKING:
    from .tuning import Tuning


class TunedISP(ISPModel):
    """Bayer raw -> the image this camera outputs, from a `twinrobo.sensor.tuned.TunedSensor`.

    The camera's output is display-encoded. It returns that image decoded to linear, the
    convention of ``frame.rgb``, so ``linear_to_srgb(frame.rgb)`` and
    `twinrobo.plugins.mujoco.camera.to_uint8` give the camera's exact pixels.
    ``metadata["isp"]`` reports the white balance.

    Args:
        tuning: A `twinrobo.isp.tuning.Tuning`.
        analogue_gain: The sensor's gain: it sets the noise that denoise and sharpen expect.
        lux: Scene illuminance for the white-balance prior.
    """

    input_format = "rggb"

    def __init__(self, tuning: Tuning, analogue_gain: float = 2.0, lux: float = 300.0):
        self.tuning = tuning
        self.analogue_gain = float(analogue_gain)
        self.lux = float(lux)

    def process(self, raw: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        t = self.tuning
        full = t.full_scale
        rgb = demosaic_rggb(raw)
        zones = F.adaptive_avg_pool2d(rgb * full, ZONES).permute(0, 2, 3, 1).flatten(1, 2)
        awb = [bayes_awb(z, t, self.lux) for z in zones.double().cpu().numpy()]
        scale = (metadata or {}).get("sensor", {}).get("noise_scale", 1.0)
        slope = self.analogue_gain**0.5 * t.noise_slope * scale / full**0.5
        # spatial denoise where the noise is even, its standard deviation about 1
        stable = 2.0 * rgb.clamp_min(0).sqrt() / slope
        stable = (
            t.sdn_strength * bilateral(stable, t.sdn_deviation, 1.5) + (1 - t.sdn_strength) * stable
        )
        rgb = (stable * slope / 2.0) ** 2
        gains = rgb.new_tensor([[r, 1.0, b] for _, r, b in awb])[..., None, None]
        ccm = torch.stack([rgb.new_tensor(t.ccm(ct)) for ct, _, _ in awb])
        rgb = torch.einsum("bij,bjhw->bihw", ccm, rgb * gains).clamp(0, 1)
        ycc = rgb_to_ycc(rgb)
        ycc[:, 1:] = (
            t.cdn_strength * gaussian_blur(ycc[:, 1:], 2.0) + (1 - t.cdn_strength) * ycc[:, 1:]
        )
        rgb = ycc_to_rgb(ycc).clamp(0, 1)
        curve = rgb.new_tensor(t.tone_curve)
        out = piecewise_linear(rgb * 65535.0, curve[:, 0], curve[:, 1]) / 65535.0
        ycc = rgb_to_ycc(out)
        luma = ycc[:, :1]
        detail = luma - gaussian_blur(luma, 1.0)
        noise = slope * luma.clamp(1e-4, 1).sqrt()
        d = detail.sign() * (detail.abs() - t.sharpen_threshold * noise).clamp_min(0)
        limit = t.sharpen_limit * 0.1
        ycc[:, :1] = luma + (d * t.sharpen_strength).clamp(-limit, limit)
        if metadata is not None:
            metadata["isp"] = {
                "tuning": t.name,
                "awb_ct_k": [a[0] for a in awb],
                "awb_gains": [[a[1], a[2]] for a in awb],
            }
        return srgb_to_linear(ycc_to_rgb(ycc).clamp(0, 1))
