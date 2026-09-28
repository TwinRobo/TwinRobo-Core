"""Simulator plugins: a `CameraTwin` rendering from a simulator's camera.

- `twinrobo.plugins.mujoco`: MuJoCo (plain ``mujoco`` and robosuite), with camera
  mounts on robot links and the lens-ray methods.
- `twinrobo.plugins.isaac`: NVIDIA Isaac Sim (any USD camera prim), all rendering
  methods; runs inside Isaac (see ``docker/isaac``).

Any other renderer can use `CameraTwin.process` (RGB + depth), or implement a lens
scene for the lens-ray methods (`twinrobo.optics.lensrender`). Plugins import their
simulator lazily, so importing them works without it installed.
"""
