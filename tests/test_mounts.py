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


def test_quat_needs_no_mujoco_and_matches_it():
    from twinrobo.mujoco.mounts import CameraMount, mat_to_quat

    rng = np.random.default_rng(0)
    for rpy in [
        (0, 0, 0),
        (180, 0, 0),
        (0, 180, 0),
        (90, -90, 45),
        *rng.uniform(-180, 180, (50, 3)),
    ]:
        m = CameraMount("c", rpy_deg=tuple(float(v) for v in rpy))
        q = m.quat()
        w, x, y, z = q
        R = np.array(  # rotation matrix of q, back again
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        assert np.allclose(R, m.rotation(), atol=1e-9) and q[0] >= 0
    mujoco = pytest.importorskip("mujoco")
    for rpy in rng.uniform(-180, 180, (20, 3)):
        R = CameraMount("c", rpy_deg=tuple(float(v) for v in rpy)).rotation()
        ref = np.zeros(4)
        mujoco.mju_mat2Quat(ref, R.reshape(-1))
        ref = -ref if ref[0] < 0 else ref
        assert np.allclose(mat_to_quat(R), ref, atol=1e-9)
