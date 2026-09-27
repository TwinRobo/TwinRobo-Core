"""Isaac Sim <-> CameraTwin tensor bridge.

Converts Isaac annotator outputs (RGB, metric depth) into CameraTwin tensor
conventions (`twinrobo.frame`) while staying on the GPU.

Status: not implemented (Phase 2).
"""

from __future__ import annotations

from ..exceptions import SimulatorError


def require_isaac():
    """Import and return the ``isaacsim`` module, or raise `SimulatorError`."""
    try:
        import isaacsim
    except ImportError as e:
        raise SimulatorError("NVIDIA Isaac Sim is not available in this Python environment.") from e
    return isaacsim
