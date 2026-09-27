"""LIBERO plugin (needs LIBERO_ROOT, LIBERO_CONFIG_PATH, robosuite and a GL backend)."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("robosuite")
if not (os.environ.get("LIBERO_CONFIG_PATH") and os.environ.get("LIBERO_ROOT")):
    pytest.skip("set LIBERO_ROOT and LIBERO_CONFIG_PATH", allow_module_level=True)

from twinrobo import CameraTwin  # noqa: E402
from twinrobo.geometry import CameraIntrinsics  # noqa: E402
from twinrobo.libero import CameraTwinLiberoEnv, make_env  # noqa: E402

pytestmark = [pytest.mark.libero, pytest.mark.gl]


@pytest.fixture(scope="module")
def env():
    env, task, init = make_env("libero_spatial", 0, resolution=128)
    env.seed(0)
    env.reset()
    env.init = init
    yield env
    env.close()


def test_init_states_load_safely(env):
    assert env.init.shape[0] == 50 and env.init.dtype == np.float64


def test_identity_twin_reproduces_libero_observation(env):
    """Identity optics at LIBERO's own FoV and resolution: the observation is unchanged."""
    fy = 64 / np.tan(np.radians(45.0) / 2)
    ident = CameraTwin(intrinsics=CameraIntrinsics(128, 128, fy, fy, 63.5, 63.5))
    wrapped = CameraTwinLiberoEnv(env, {"agentview": ident}, keep_ideal=True)
    obs = wrapped.set_init_state(env.init[0])
    diff = np.abs(obs["agentview_image"].astype(int) - obs["agentview_image_ideal"])
    assert diff.max() <= 1  # sRGB round trip only
    assert (
        obs["agentview_image"].shape == (128, 128, 3) and obs["agentview_image"].dtype == np.uint8
    )


def test_fovy_restored_after_render(env):
    import mujoco

    fy = 20 / np.tan(np.radians(30.0) / 2)
    narrow = CameraTwin(intrinsics=CameraIntrinsics(64, 40, fy, fy, 31.5, 19.5))
    wrapped = CameraTwinLiberoEnv(env, {"agentview": narrow})
    wrapped.set_init_state(env.init[0])
    m = env.sim.model._model
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    assert float(m.cam_fovy[cam]) == pytest.approx(45.0)
