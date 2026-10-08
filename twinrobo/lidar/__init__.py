"""LiDAR twins: catalog lidars ray cast into simulator scenes.

A `LidarSpec` (``lidar.yaml`` in the catalog) describes a lidar's scan pattern, range and
noise; `LidarTwin.scan` casts its beams into a scene's triangles (the same GPU BVH the lens
ray caster uses) and returns a `LidarFrame`: range, intensity and, for FMCW lidars, radial
velocity as ``[rings, columns]`` images, with points on demand. Simulator plugins supply the
scene and the poses (`twinrobo.plugins.mujoco.lidar`).

Frames follow ROS REP-103: x forward, y left, z up.
"""

from .registry import LidarRegistry
from .scan import LidarFrame, LidarTwin
from .spec import LidarSpec

__all__ = ["LidarFrame", "LidarRegistry", "LidarSpec", "LidarTwin"]
