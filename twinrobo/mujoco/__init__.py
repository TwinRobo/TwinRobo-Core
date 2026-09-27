"""MuJoCo simulator adapter.

`MujocoCameraTwin` renders ideal RGB + metric depth from a MuJoCo camera at the
CameraTwin's resolution and field of view, then runs the CameraTwin pipeline.
Rendering goes through a `RenderBackend`:

- `MujocoRenderer` uses ``mujoco.Renderer``, for plain MuJoCo scenes.
- `RobosuiteRenderer` uses robosuite's own ``sim.render``, for robosuite and
  LIBERO. A second GL context in the same process corrupts robosuite's
  offscreen context, so inside robosuite always use this backend.

MuJoCo is imported lazily; ``import twinrobo.mujoco`` does not need it.
"""

from .camera import MujocoCameraTwin, MujocoRenderer, RenderBackend, RobosuiteRenderer, to_uint8

__all__ = ["MujocoCameraTwin", "MujocoRenderer", "RobosuiteRenderer", "RenderBackend", "to_uint8"]
