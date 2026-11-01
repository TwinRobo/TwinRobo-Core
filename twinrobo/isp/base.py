"""ISP model interface."""

from __future__ import annotations

from typing import Any

from torch import Tensor


class ISPModel:
    """Converts a raw sensor image into the camera's output image.

    Subclass it and override `process` to use your own ISP, classical or learned::

        from twinrobo.isp.color import srgb_to_linear

        class Gamma(ISPModel):
            def process(self, raw, metadata=None):
                return srgb_to_linear(raw.clamp(0, 1) ** 0.8)

    What a `process` must do:

    - ``raw`` has the sensor's `~twinrobo.sensor.SensorModel.output_format`, which
      ``input_format`` must equal: ``"rgb"`` is scene-linear ``[B, 3, H, W]``, ``"rggb"`` a
      Bayer mosaic ``[B, 1, H, W]``, 0 to 1 of full scale above black. Do not change it in
      place: it is ``frame.raw``.
    - Return float ``[B, 3, H, W]`` on the same device, with the same size and pixel positions:
      depth and rectification assume them. It is converted to the frame's float type. The
      image is linear, like ``frame.rgb``: decode a display-encoded result with
      `twinrobo.isp.color.srgb_to_linear`.
    - Keep anything carried between frames per stream: ``metadata["stream"]``, else
      ``metadata["camera"]``, and forget it in `reset`.
    - Report what you like in ``metadata["isp"]``: it reaches ``frame.metadata``.
    """

    #: The raw image this ISP takes.
    input_format = "rgb"

    def reset(self, stream: Any = None) -> None:
        """Forget what one stream of frames, or all of them, left in the ISP."""

    def process(self, raw: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        raise NotImplementedError


class IdentityISP(ISPModel):
    """Pass-through ISP: output equals raw."""

    def process(self, raw: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        return raw
