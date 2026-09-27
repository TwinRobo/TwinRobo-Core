"""Isaac Sim availability check."""

from __future__ import annotations

from ..exceptions import SimulatorError


def require_isaac():
    """Import and return the ``isaacsim`` module, or raise `SimulatorError`."""
    try:
        import isaacsim
    except ImportError as e:
        raise SimulatorError(
            "NVIDIA Isaac Sim is not available in this Python environment "
            "(run inside Isaac, e.g. docker/isaac/run.sh)"
        ) from e
    return isaacsim
