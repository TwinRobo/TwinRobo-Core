"""Robot-learning environments and datasets, replayed so any camera can be placed in them.

- `twinrobo.datasets.libero`: LIBERO tasks and demo datasets (robosuite 1.4), with
  extra robots and multi-camera variants, exact demo replay and a camera-swapping
  env wrapper (`CameraTwinLiberoEnv`).
- `twinrobo.datasets.robocasa`: RoboCasa kitchens and LeRobot episodes (robosuite 1.5),
  each episode replayed in its own generated kitchen.

These render through `twinrobo.plugins.mujoco`.
"""
