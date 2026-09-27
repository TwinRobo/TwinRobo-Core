import importlib.util

import pytest

from twinrobo import SimulatorError
from twinrobo.isaac import CameraTwin


@pytest.mark.skipif(importlib.util.find_spec("isaacsim") is not None, reason="Isaac installed")
def test_attach_without_isaac_raises():
    with pytest.raises(SimulatorError):
        CameraTwin().attach("/World/cam")
