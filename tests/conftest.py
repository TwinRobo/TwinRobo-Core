from pathlib import Path

import pytest

from twinrobo.registry import BUILTIN_CATALOG


@pytest.fixture
def example_spec_path() -> Path:
    return BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml"
