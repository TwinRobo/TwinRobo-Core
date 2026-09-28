import importlib.util
from pathlib import Path

import pytest

from twinrobo import CameraSpec, CameraTwin, SimulatorError
from twinrobo.plugins.isaac import IsaacCameraTwin


@pytest.mark.skipif(importlib.util.find_spec("isaacsim") is not None, reason="Isaac installed")
def test_attach_without_isaac_raises():
    spec = CameraSpec.from_yaml(
        Path(__file__).resolve().parent.parent.parent / "data" / "cellphone80deg" / "camera.yaml"
    )
    twin = CameraTwin.from_spec(spec, build_psf=False, device="cpu")
    with pytest.raises(SimulatorError, match="Isaac Sim is not available"):
        IsaacCameraTwin(twin, "/World/Camera")
