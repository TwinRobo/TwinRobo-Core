"""LiDAR twins in MuJoCo: orientation, ranges, FMCW velocity, reflectivity dropout, noise."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("warp")

from twinrobo.lidar import LidarRegistry, LidarSpec  # noqa: E402
from twinrobo.plugins.mujoco.lidar import MujocoLidar  # noqa: E402

SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.8 0.8 0.8 1" group="1"/>
    <geom name="ahead" type="box" pos="3 0 1" size="0.1 0.5 0.5" rgba="1 1 1 1" group="1"/>
    <geom name="left" type="box" pos="0 4 1" size="0.5 0.1 0.5" rgba="1 1 1 1" group="1"/>
    <geom name="dark_far" type="box" pos="-60 0 1" size="0.1 5 5" rgba=".05 .05 .05 1" group="1"/>
    <body name="mover" pos="0 -5 1">
      <freejoint/>
      <geom type="box" size="0.5 0.1 0.5" rgba="1 1 1 1" group="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def spinning(**kw) -> LidarSpec:
    d = {
        "type": "lidar",
        "id": "test/spin",
        "scan": {"pattern": "spinning", "channels": 9, "vertical_deg": {"min": -20, "max": 20},
                 "columns": 360, "frame_hz": 10},
        "range": {"min_m": 0.1, "max_m": 50.0, "at_reflectivity": 0.1},
        **kw,
    }  # fmt: skip
    return LidarSpec.from_dict(d)


@pytest.fixture()
def scene():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def column(frame, az):
    return int(np.argmin(np.abs(((frame.azimuth_deg - az + 180) % 360) - 180)))


def test_frame_convention_x_forward_y_left(scene):
    model, data = scene
    lidar = MujocoLidar(spinning(), model, data, pos=(0, 0, 1.0))
    f = lidar.scan()
    mid = f.elevation_deg.size // 2  # the level ring (0 deg)
    assert f.elevation_deg[mid] == pytest.approx(0)
    assert f.range[mid, column(f, 0)] == pytest.approx(2.9, abs=0.01)  # the box ahead, +x
    assert f.range[mid, column(f, 90)] == pytest.approx(3.9, abs=0.01)  # the box left, +y
    # points come back in the sensor frame: the box ahead at x ~ 2.9, y ~ 0
    pts = f.points()
    ahead = pts[(np.abs(pts[:, 1]) < 0.05) & (np.abs(pts[:, 2]) < 0.05) & (pts[:, 0] > 0)]
    assert ahead[:, 0].min() == pytest.approx(2.9, abs=0.01)


def test_floor_range_follows_beam_elevation(scene):
    model, data = scene
    f = MujocoLidar(spinning(), model, data, pos=(0, 0, 1.0)).scan()
    low = 0  # rings are top first: the last ring looks most down
    ring = f.elevation_deg.size - 1
    want = 1.0 / np.sin(np.radians(-f.elevation_deg[ring]))  # from 1 m above the floor
    assert f.range[ring, column(f, 180)] == pytest.approx(want, rel=1e-3)
    assert np.isnan(f.range[low, column(f, 180)])  # looking up at nothing: no return


def test_mount_rotation_turns_the_scan(scene):
    model, data = scene
    f = MujocoLidar(spinning(), model, data, pos=(0, 0, 1.0), rpy_deg=(0, 0, 90)).scan()
    mid = f.elevation_deg.size // 2
    # yawed 90 deg left: the box that was on the left (+y) is now straight ahead
    assert f.range[mid, column(f, 0)] == pytest.approx(3.9, abs=0.01)


def test_dim_far_surfaces_drop_out(scene):
    model, data = scene
    f = MujocoLidar(spinning(), model, data, pos=(0, 0, 1.0)).scan()
    mid = f.elevation_deg.size // 2
    # the dark wall 60 m behind is within max range but too dim to return
    assert np.isnan(f.range[mid, column(f, 180)])


def test_fmcw_velocity_sign(scene):
    model, data = scene
    spec = spinning(fmcw=True)
    lidar = MujocoLidar(spec, model, data, pos=(0, 0, 1.0))
    q1 = np.array(data.qpos)
    adr = model.jnt_qposadr[0]  # the mover's free joint: 1 m closer (y -5 -> -4) in 0.5 s
    q1[adr + 1] += 1.0
    f = lidar.scan(next_qpos=q1, dt=0.5)
    mid = f.elevation_deg.size // 2
    assert f.velocity[mid, column(f, -90)] == pytest.approx(-2.0, abs=0.05)  # approaching
    assert f.velocity[mid, column(f, 0)] == pytest.approx(0.0, abs=1e-4)  # static box


def test_noise_is_seeded_by_frame(scene):
    model, data = scene
    lidar = MujocoLidar(spinning(noise={"range_sigma_m": 0.02}), model, data, pos=(0, 0, 1.0))
    a, b, c = lidar.scan(frame=3), lidar.scan(frame=3), lidar.scan(frame=4)
    np.testing.assert_array_equal(a.range, b.range)
    ok = a.valid() & c.valid()
    assert np.abs(a.range[ok] - c.range[ok]).max() > 0


def test_catalog_lidars_load():
    reg = LidarRegistry()
    ids = reg.list()
    assert {"velodyne/vlp-16", "ouster/os1-64", "hesai/xt32"} <= set(ids)
    for i in ids:
        s = reg.load(i)
        assert s.rings >= 1 and s.columns >= 1 and s.max_range_m > s.min_range_m
