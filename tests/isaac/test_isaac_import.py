import importlib.util

import pytest

from twinrobo import CameraSpec, CameraTwin, SimulatorError
from twinrobo.isaac import IsaacCameraTwin
from twinrobo.registry import BUILTIN_CATALOG


@pytest.mark.skipif(importlib.util.find_spec("isaacsim") is not None, reason="Isaac installed")
def test_attach_without_isaac_raises():
    spec = CameraSpec.from_yaml(BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml")
    twin = CameraTwin.from_spec(spec, build_psf=False, device="cpu")
    with pytest.raises(SimulatorError, match="Isaac Sim is not available"):
        IsaacCameraTwin(twin, "/World/Camera")
