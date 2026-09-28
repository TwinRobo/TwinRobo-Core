from pathlib import Path

import pytest


@pytest.fixture
def example_spec_path() -> Path:
    """A camera for tests: a DeepLens design lens on a hypothetical sensor (not in the catalog)."""
    return Path(__file__).resolve().parent / "data" / "cellphone80deg" / "camera.yaml"
