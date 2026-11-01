"""CameraTwin: physically grounded digital twins of real cameras for robotics simulation."""

from .camera import CameraTwin
from .device import DeviceTwin
from .exceptions import (
    CameraTwinError,
    CatalogError,
    DepthUnavailableError,
    OpticsBackendError,
    SimulatorError,
    SpecError,
)
from .frame import CameraFrame
from .lidar import LidarFrame, LiDARTwin, LidarTwin
from .registry import CatalogRegistry
from .rig import Observation
from .spec import CameraSpec
from .stereo import StereoTwin
from .version import __version__

__all__ = [
    "__version__",
    "CameraTwin",
    "CameraFrame",
    "CameraSpec",
    "DeviceTwin",
    "LiDARTwin",
    "LidarFrame",
    "LidarTwin",
    "Observation",
    "StereoTwin",
    "CatalogRegistry",
    "CameraTwinError",
    "CatalogError",
    "DepthUnavailableError",
    "OpticsBackendError",
    "SimulatorError",
    "SpecError",
]
