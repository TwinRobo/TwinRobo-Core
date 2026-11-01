"""Camera mounts on a real MuJoCo model (no rendering)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from twinrobo.plugins.mujoco.mounts import (  # noqa: E402
    HOST_CAMERA,
    CameraMount,
    body_id,
    expand_mount,
    mount_world_pose,
    mountable_bodies,
    mounted,
    pair_baseline,
    reattach,
    set_pair_baseline,
    snap_to_body,
)

XML = f"""
<mujoco><worldbody>
  <camera name="{HOST_CAMERA}" pos="9 9 9"/>
  <camera name="far"/>
  <body name="prop" pos="3 0 0"><geom type="box" size=".1 .1 .1"/></body>
  <body name="robot0_link" pos="0 0 1" euler="0 0 90">
    <joint name="j" type="hinge"/>
    <geom type="box" size=".05 .05 .2" group="1"/>
    <camera name="eye_L" pos="0 0 .3"/>
    <camera name="eye_R" pos="0.1 0 .3"/>
    <camera name="same" pos="0 0 .3"/>
    <body name="gripper0_finger" pos="0 0 .4"><geom type="box" size=".02 .02 .02" group="1"/></body>
  </body>
</worldbody></mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "c", "pos": (0, 0)}, "three values"),
        ({"name": "c", "fovy": 200}, "fovy"),
        ({"name": "bad name"}, "camera name"),
        ({"name": "c", "baseline_mm": 50}, "set module"),
        ({"name": "c", "module": "x/y", "baseline_mm": 0.5}, "baseline_mm must"),
    ],
)
def test_a_bad_mount_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CameraMount(**kwargs)


def test_bodies(scene):
    model, _ = scene
    assert body_id(model, "world") == 0
    assert mountable_bodies(model) == ["world", "robot0_link", "gripper0_finger"]
    with pytest.raises(ValueError, match="no body"):
        body_id(model, "nothing")


def test_mounted_moves_the_host_camera_then_restores_it(scene):
    model, data = scene
    mount = CameraMount("c", "robot0_link", (0.1, 0.2, 0.3), (10, 20, 0), fovy=33.0)
    h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, HOST_CAMERA)
    before = model.cam_pos[h].copy(), int(model.cam_bodyid[h]), float(model.cam_fovy[h])
    with mounted(model, data, mount) as host:
        assert host == HOST_CAMERA
        p, R = mount_world_pose(model, data, mount)
        assert data.cam_xpos[h] == pytest.approx(p)
        assert data.cam_xmat[h].reshape(3, 3) == pytest.approx(R)
        assert model.cam_fovy[h] == 33.0
    assert model.cam_pos[h] == pytest.approx(before[0])
    assert (int(model.cam_bodyid[h]), float(model.cam_fovy[h])) == before[1:]


def test_mounted_restores_after_an_error(scene):
    model, data = scene
    h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, HOST_CAMERA)
    before = model.cam_pos[h].copy()
    with pytest.raises(RuntimeError), mounted(model, data, CameraMount("c")):
        raise RuntimeError
    assert model.cam_pos[h] == pytest.approx(before)


def test_mounted_needs_its_host(scene):
    model, data = scene
    with (
        pytest.raises(ValueError, match="host camera"),
        mounted(model, data, CameraMount("c"), "x"),
    ):
        pass


def test_reattach_keeps_the_world_pose(scene):
    model, data = scene
    mount = CameraMount("c", "world", (1.0, 0.5, 2.0), (30, -20, 0))
    moved = reattach(model, data, mount, "robot0_link")
    p0, R0 = mount_world_pose(model, data, mount)
    p1, R1 = mount_world_pose(model, data, moved)
    assert moved.body == "robot0_link"
    assert p1 == pytest.approx(p0, abs=1e-3)
    assert R1 == pytest.approx(R0, abs=1e-3)


def test_snap_puts_the_camera_outside_the_link_looking_at_the_target(scene):
    model, data = scene
    target = np.array([1.0, 0.5, 1.0])
    snapped = snap_to_body(model, data, CameraMount("c", "world", (1, 0, 1)), "robot0_link", target)
    p, R = mount_world_pose(model, data, snapped)
    assert snapped.body == "robot0_link"
    assert -R[:, 2] == pytest.approx((target - p) / np.linalg.norm(target - p), abs=1e-3)
    b = body_id(model, "robot0_link")
    local = np.abs(data.xmat[b].reshape(3, 3).T @ (p - data.xpos[b]))
    assert np.any(local > np.array([0.05, 0.05, 0.2]))  # not inside the box


def test_snap_to_the_world_keeps_the_pose(scene):
    model, data = scene
    mount = CameraMount("c", "world", (1, 0, 1), (10, 0, 0))
    assert snap_to_body(model, data, mount, "world", np.zeros(3)).pos == pytest.approx(mount.pos)


def test_pair_baseline_moves_both_eyes_about_their_middle(scene):
    model, _ = scene
    assert pair_baseline(model, "eye_L", "eye_R") == pytest.approx(0.1)
    assert set_pair_baseline(model, "eye_L", "eye_R", 0.06) == pytest.approx(0.1)
    assert pair_baseline(model, "eye_L", "eye_R") == pytest.approx(0.06)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in ("eye_L", "eye_R")]
    assert model.cam_pos[ids].mean(0) == pytest.approx([0.05, 0, 0.3])


@pytest.mark.parametrize(
    "left, right, baseline, match",
    [
        ("eye_L", "eye_R", 5.0, "within 1 mm"),
        ("eye_L", "none", 0.1, "no cameras"),
        ("eye_L", "far", 0.1, "different bodies"),
        ("eye_L", "same", 0.1, "coincide"),
    ],
)
def test_set_pair_baseline_refuses(scene, left, right, baseline, match):
    with pytest.raises(ValueError, match=match):
        set_pair_baseline(scene[0], left, right, baseline)


def test_pair_baseline_needs_its_cameras(scene):
    with pytest.raises(ValueError, match="no cameras"):
        pair_baseline(scene[0], "eye_L", "none")


def test_expand_mount_gives_two_rigid_eyes():
    eyes = expand_mount(CameraMount("w", "base", module="stereolabs/zed-x/2.2mm"))
    assert list(eyes) == ["w_L", "w_R"]
    assert np.linalg.norm(np.subtract(eyes["w_R"].pos, eyes["w_L"].pos)) == pytest.approx(0.12)
    assert {m.body for m in eyes.values()} == {"base"}
    assert expand_mount(CameraMount("w")) == {"w": CameraMount("w")}
