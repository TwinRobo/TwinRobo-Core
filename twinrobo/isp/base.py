"""ISP model interface."""

from __future__ import annotations

from typing import Any

from torch import Tensor


class ISPModel:
    """Converts a raw sensor image into the camera's output image."""

    def process(self, raw: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        raise NotImplementedError


class IdentityISP(ISPModel):
    """Pass-through ISP: output equals raw."""

    def process(self, raw: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        return raw
