"""NVIDIA Isaac Sim adapter.

`IsaacCameraTwin` renders a USD camera's view through a CameraTwin. Nothing here
imports Isaac Sim at module import time, so ``import twinrobo.plugins.isaac`` works
without Isaac installed; run it inside Isaac (see ``docker/isaac/``).
"""

from .camera import IsaacCameraTwin

__all__ = ["IsaacCameraTwin"]
