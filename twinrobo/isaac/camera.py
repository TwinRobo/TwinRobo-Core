"""Isaac Sim-attached CameraTwin.

Status: skeleton (Phase 2). ``attach``/``get_frame`` are not implemented yet.
"""

from __future__ import annotations

from ..camera import CameraTwin as _CoreCameraTwin
from ..frame import CameraFrame
from .bridge import require_isaac


class CameraTwin(_CoreCameraTwin):
    """CameraTwin bound to an Isaac Sim camera prim."""

    prim_path: str | None = None

    def attach(self, prim_path: str) -> None:
        require_isaac()
        raise NotImplementedError("Isaac attachment is Phase 2")

    def get_frame(self) -> CameraFrame:
        raise NotImplementedError("Isaac frame acquisition is Phase 2")
