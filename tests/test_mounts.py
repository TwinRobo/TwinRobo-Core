import numpy as np
import pytest

from twinrobo.mujoco.mounts import CameraMount, look_rotation, rpy_from_rotation


@pytest.mark.parametrize("rpy", [(0, 0, 0), (127, 31, 0), (-60, -20, 15), (179, 80, -30)])
def test_rpy_round_trip(rpy):
    R = CameraMount("c", rpy_deg=rpy).rotation()
    assert np.allclose(CameraMount("c", rpy_deg=rpy_from_rotation(R)).rotation(), R, atol=1e-6)


def test_look_rotation_is_upright():
    R = look_rotation([1.0, 1.0, -0.5])
    f = -R[:, 2]
    assert np.allclose(f, np.array([1, 1, -0.5]) / np.linalg.norm([1, 1, -0.5]))
    assert abs(R[2, 0]) < 1e-9 and R[2, 1] > 0  # no roll: x horizontal, y has an up component
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
