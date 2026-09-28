"""CameraTwin: physically grounded digital twins of real cameras for robotics simulation."""

from .camera import CameraTwin
from .exceptions import (
    CameraTwinError,
    CatalogError,
    DepthUnavailableError,
    OpticsBackendError,
    SimulatorError,
    SpecError,
)
from .frame import CameraFrame
from .registry import CatalogRegistry
from .spec import CameraSpec
from .version import __version__

__all__ = [
    "__version__",
    "CameraTwin",
    "CameraFrame",
    "CameraSpec",
    "CatalogRegistry",
    "CameraTwinError",
    "CatalogError",
    "DepthUnavailableError",
    "OpticsBackendError",
    "SimulatorError",
    "SpecError",
]
