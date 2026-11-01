"""The robot catalog and camera mounts, without robosuite or LIBERO."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from twinrobo.datasets.libero import replay
from twinrobo.datasets.libero import robots as R
from twinrobo.stereo import ModuleRegistry

mujoco = pytest.importorskip("mujoco")


def robot_model():
    world = ET.fromstring(
        """<worldbody>
          <body name="robot0_base" pos="0 0 1">
            <geom type="box" size=".1 .1 .1"/>
            <body name="robot0_right_hand" pos="0.5 0 0.2">
              <geom type="box" size=".03 .03 .03"/>
              <camera name="robot0_eye_in_hand" pos="0.05 0 0" quat="0.5 0.5 0.5 0.5"/>
            </body>
          </body>
        </worldbody>"""
    )
    return SimpleNamespace(naming_prefix="robot0_", worldbody=world)


def compile_(model):
    text = f"<mujoco>{ET.tostring(model.worldbody, encoding='unicode')}</mujoco>"
    m = mujoco.MjModel.from_xml_string(text)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def cam(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)


def test_catalog_lists_every_robot_for_its_simulator():
    assert "Panda" in R.robots_for("libero") and R.ROBOTS == R.robots_for("libero")
    assert R.base_robot("PandaMultiCam") == "Panda"
    assert R.base_robot("Unknown") == "Unknown"


def test_a_multicam_robot_names_its_stereo_eyes():
    r = next(r for r in R.ROBOT_CATALOG if r["name"] == "PandaMultiCam")
    wrist = r["stereo"]["robot0_wrist"]
    assert wrist["left"] in r["cameras"] and wrist["right"] in r["cameras"]
    assert "robot0_mast" in r["cameras"] and "robot0_eye_in_hand" in r["cameras"]
    assert ModuleRegistry().load(wrist["module"]).id == wrist["module"]


def test_quaternion_rotation():
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # 90 deg about z
    assert R._rotate(q, np.array([1.0, 0, 0])) == pytest.approx([0, 1, 0], abs=1e-9)


def test_added_cameras_compile_where_they_should_be():
    model = robot_model()
    R.add_camera_mounts(model)
    m, d = compile_(model)
    left, right = cam(m, "robot0_wrist_L"), cam(m, "robot0_wrist_R")
    module = ModuleRegistry().load(R.MULTICAM_MOUNTS["wrist"]["module"])
    assert np.linalg.norm(d.cam_xpos[right] - d.cam_xpos[left]) == pytest.approx(module.baseline_m)
    ref = cam(m, "robot0_eye_in_hand")
    assert (d.cam_xpos[left] + d.cam_xpos[right]) / 2 == pytest.approx(d.cam_xpos[ref], abs=1e-6)
    assert d.cam_xmat[left] == pytest.approx(d.cam_xmat[ref], abs=1e-4)  # faces the same way


def test_the_mast_camera_looks_at_its_target():
    model = robot_model()
    R.add_camera_mounts(model)
    m, d = compile_(model)
    c = cam(m, "robot0_mast")
    forward = -d.cam_xmat[c].reshape(3, 3)[:, 2]
    base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")]
    target = base + R.MULTICAM_MOUNTS["mast"]["look_at"]
    want = (target - d.cam_xpos[c]) / np.linalg.norm(target - d.cam_xpos[c])
    assert forward == pytest.approx(want, abs=1e-4)


def test_added_hardware_changes_no_dynamics():
    model = robot_model()
    R.add_camera_mounts(model)
    added = [g for g in model.worldbody.iter("geom") if g.get("name", "").startswith("robot0_")]
    assert len(added) >= 3  # the mast's post and housing, the wrist module's housing
    for geom in added:
        assert geom.get("contype") == "0" and geom.get("conaffinity") == "0"


def test_a_missing_body_is_named():
    model = robot_model()
    model.worldbody.remove(model.worldbody[0])
    with pytest.raises(ValueError, match="no body robot0_"):
        R.add_camera_mounts(model)


def test_scene_joints_move_to_another_robot():
    xml = """<mujoco><worldbody>
      <body name="r"><joint name="robot0_j"/><geom size=".1"/></body>
      <body name="o"><joint name="obj" type="free"/><geom size=".1"/></body>
    </worldbody></mujoco>"""
    layout = R.joint_layout(mujoco.MjModel.from_xml_string(xml))
    assert layout == {"robot0_j": (0, 1), "obj": (1, 7)}
    src = np.arange(16.0).reshape(2, 8)
    out, missing = R.remap_scene_qpos(src, layout, layout, np.zeros(8))
    assert not missing and (out[:, 0] == 0).all() and (out[:, 1:] == src[:, 1:]).all()
    out, missing = R.remap_scene_qpos(src, layout, {"robot0_j": (0, 1)}, np.zeros(1))
    assert missing == ["obj"] and out.shape == (2, 1)


def test_static_bodies_of_a_recorded_scene():
    xml = """<mujoco><worldbody>
      <body name="table" pos="1 2 3" quat="0 1 0 0"/><body pos="9 9 9"/>
    </worldbody></mujoco>"""
    poses = replay.static_body_poses(xml)
    assert list(poses) == ["table"]
    assert poses["table"][0] == pytest.approx([1, 2, 3])
    assert poses["table"][1] == pytest.approx([0, 1, 0, 0])
    assert replay.static_body_poses("<mujoco/>") == {}


def test_recorded_fixtures_are_placed_and_the_rest_left_alone():
    live = """<mujoco><worldbody>
      <body name="table"/>
      <body name="moving"><freejoint/><geom size=".1"/></body>
      <body name="robot0_base"/>
    </worldbody></mujoco>"""
    recorded = """<mujoco><worldbody>
      <body name="table" pos="1 2 3"/><body name="moving" pos="4 4 4"/>
      <body name="robot0_base" pos="5 5 5"/><body name="absent" pos="6 6 6"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(live)
    sim = SimpleNamespace(model=SimpleNamespace(_model=model), forward=lambda: None)
    assert replay.apply_fixture_placement(sim, recorded) == ["table"]
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    assert model.body_pos[b] == pytest.approx([1, 2, 3])
