"""One headless Isaac Sim app for the Isaac tests.

`SimulationApp.close` ends the process, so it runs after pytest's report
(`pytest_unconfigure`), not in a fixture teardown.
"""

import importlib.util

import pytest

_APP = []


@pytest.fixture(scope="session")
def app():
    if importlib.util.find_spec("isaacsim") is None:
        pytest.skip("needs NVIDIA Isaac Sim (docker/isaac/run.sh)")
    if not _APP:
        from isaacsim import SimulationApp

        _APP.append(SimulationApp({"headless": True}))
    return _APP[0]


def pytest_unconfigure(config):
    if _APP:
        _APP.pop().close()
