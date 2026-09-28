"""TwinRobo exception hierarchy."""


class CameraTwinError(Exception):
    """Base class for all TwinRobo errors."""


class SpecError(CameraTwinError):
    """A CameraSpec is malformed or uses an unsupported schema version."""


class CatalogError(CameraTwinError):
    """A camera ID cannot be resolved in the catalog."""


class OpticsBackendError(CameraTwinError):
    """An optics backend (e.g. DeepLens) is unavailable or failed."""


class DepthUnavailableError(CameraTwinError):
    """A frame's depth was read; its camera has no simulated depth output (``outputs.depth``)."""


class SimulatorError(CameraTwinError):
    """A simulator adapter (e.g. Isaac Sim) is unavailable or failed."""
