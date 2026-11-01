"""The standard twin API: attach, capture, reset, for every kind of twin."""

import shutil

import numpy as np
import pytest

from twinrobo import CameraTwin, DeviceTwin, LiDARTwin, LidarTwin, Observation, StereoTwin
from twinrobo.device import DeviceRegistry, DeviceSpec
from twinrobo.exceptions import SpecError
from twinrobo.geometry import CameraIntrinsics
from twinrobo.lidar import LidarRegistry, LidarSpec
from twinrobo.rig import CameraRig, GroupRig, LidarRig
from twinrobo.stereo import ModuleRegistry

mujoco = pytest.importorskip("mujoco")

XML = """
<mujoco><worldbody>
  <light pos="0 0 3"/>
  <geom type="plane" size="5 5 .1" rgba=".5 .5 .5 1"/>
  <geom type="box" pos="0 2 .5" size=".5 .1 .5" rgba="1 0 0 1"/>
  <camera name="cam_L" pos="0 0 .5" xyaxes="1 0 0 0 0 1"/>
  <camera name="cam_R" pos="0.2 0 .5" xyaxes="1 0 0 0 0 1"/>
</worldbody></mujoco>
"""


class FakeAdapter:
    def __init__(self):
        self.calls, self.resets, self.closed = [], 0, False

    def get_frame(self, timestamp=None, **options):
        self.calls.append(options)
        return options

    def scan(self, frame=0, **options):
        self.calls.append(frame)
        return frame

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeBackend:
    """Like Isaac: one render fills every camera of a frame."""

    capture_options = {"hold_time": True}
    same_frame = {"step": False}

    def __init__(self):
        self.adapters = {}

    def camera(self, twin, camera, **options):
        self.adapters[camera] = FakeAdapter()
        self.adapters[camera].options = options
        return self.adapters[camera]


def small_twin(W=32, H=24):
    fy = H / 2 / np.tan(np.radians(30))
    return CameraTwin(intrinsics=CameraIntrinsics(W, H, fy, fy, (W - 1) / 2, (H - 1) / 2))


def small_stereo():
    return StereoTwin(ModuleRegistry().load("stereolabs/zed-x/2.2mm"), small_twin(), small_twin())


@pytest.fixture
def backend():
    from twinrobo.plugins.mujoco import MujocoRenderer

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    backend = MujocoRenderer(model, data)
    yield backend
    backend.close()


@pytest.mark.gl
def test_camera_captures_a_frame(backend):
    rig = small_twin().attach(backend, "cam_L", device="cpu")
    obs = rig.capture(timestamp=1.5)
    assert isinstance(rig, CameraRig) and list(obs) == ["camera"] and obs.timestamp == 1.5
    assert obs["camera"].rgb.shape == (1, 3, 24, 32) and obs["camera"].timestamp == 1.5
    rig.reset()


@pytest.mark.gl
def test_stereo_eyes_see_their_own_views(backend):
    with pytest.warns(UserWarning, match="mm apart"):
        rig = small_stereo().attach(backend, camera="cam", device="cpu")
    obs = rig.capture()
    assert list(obs) == ["left", "right"]
    assert (obs["left"].rgb - obs["right"].rgb).abs().max() > 0.01  # the box shifts between eyes


@pytest.mark.gl
def test_stereo_warns_when_the_cameras_are_not_at_the_baseline(backend):
    with pytest.warns(UserWarning, match="mm apart"):
        small_stereo().attach(backend, left="cam_L", right="cam_R", device="cpu")


def test_stereo_needs_its_cameras():
    with pytest.raises(ValueError, match="left="):
        small_stereo().attach(FakeBackend(), left="a")


def test_stereo_rectifies_like_its_module():
    be = FakeBackend()
    small_stereo().attach(be, left="a", right="b")
    assert be.adapters["a"].options["rectify"] is True
    small_stereo().attach(be, left="a", right="b", rectify=False)
    assert be.adapters["b"].options["rectify"] is False


def test_stereo_shares_a_twin_between_identical_eyes():
    twin = StereoTwin.from_catalog("intel/realsense-d435", device="cpu", build_psf=False)
    assert twin.left is twin.right and twin.baseline_m == pytest.approx(0.05)


def test_later_parts_share_the_rendered_frame():
    be = FakeBackend()
    rig = small_stereo().attach(be, left="a", right="b")
    rig.capture(force_depth=True)
    assert be.adapters["a"].calls == [{"force_depth": True, "hold_time": True}]
    assert be.adapters["b"].calls == [{"force_depth": True, "hold_time": True, "step": False}]


def test_group_passes_an_option_to_one_part():
    be = FakeBackend()
    rig = small_stereo().attach(be, left="a", right="b")
    rig.capture(left={"force_depth": True})
    assert be.adapters["a"].calls == [{"force_depth": True, "hold_time": True}]
    assert be.adapters["b"].calls == [{"hold_time": True, "step": False}]


def test_reset_and_close_reach_every_part():
    be = FakeBackend()
    rig = small_stereo().attach(be, left="a", right="b")
    rig.reset()
    rig.close()
    assert all(a.resets == 1 and a.closed for a in be.adapters.values())


class FakeSim:
    model, data = object(), object()


def fake_lidar():
    adapter = FakeAdapter()
    adapter.model, adapter.data = FakeSim.model, FakeSim.data
    return adapter


def test_lidar_rig_counts_frames_and_resets():
    rig = LidarRig(fake_lidar, FakeSim())
    assert [rig.capture()["lidar"], rig.capture()["lidar"]] == [0, 1]
    rig.reset()
    assert rig.capture()["lidar"] == 0


def test_lidar_rig_follows_a_rebuilt_simulator():
    sim = FakeSim()
    made = []

    def make():
        made.append(FakeAdapter())
        made[-1].model, made[-1].data = sim.model, sim.data
        return made[-1]

    rig = LidarRig(make, sim)
    rig.capture()
    sim.model = object()
    rig.capture()
    assert len(made) == 2 and made[1].calls == [1]


def test_a_second_eye_never_renders_again():
    be = FakeBackend()
    rig = small_stereo().attach(be, left="a", right="b")
    rig.capture(step=True)
    assert be.adapters["b"].calls[0]["step"] is False


def test_a_part_option_overrides_a_shared_one():
    be = FakeBackend()
    rig = small_stereo().attach(be, left="a", right="b")
    rig.capture(force_depth=True, right={"force_depth": False})
    assert be.adapters["b"].calls[0]["force_depth"] is False


def test_a_failed_isp_switch_changes_neither_eye():
    stereo = StereoTwin(
        ModuleRegistry().load("stereolabs/zed-x/2.2mm"),
        CameraTwin.from_catalog(
            "raspberrypi/camera-module-3/standard", device="cpu", build_psf=False
        ),
        CameraTwin.from_catalog("logitech/c920", device="cpu", build_psf=False),
    )
    before = stereo.left.isp
    with pytest.raises(ValueError, match="no isp.tuning"):
        stereo.set_isp(True)
    assert stereo.left.isp is before


@pytest.mark.gl
def test_reset_finds_the_camera_again_in_a_rebuilt_model(backend):
    rig = small_twin().attach(backend, "cam_R", device="cpu")
    other = XML.replace('name="cam_L"', 'name="first"/><camera name="cam_L"', 1)
    backend.model = mujoco.MjModel.from_xml_string(
        other.replace('"first"/>', '"first" pos="0 0 1"/>')
    )
    backend.data = mujoco.MjData(backend.model)
    mujoco.mj_forward(backend.model, backend.data)
    backend._renderers.clear()
    rig.reset()
    assert rig.adapter._cam_id == 2


@pytest.mark.gl
def test_baseline_of_cameras_on_two_bodies(backend):
    xml = """<mujoco><worldbody>
      <body name="a" pos="0 0 1"><camera name="l"/></body>
      <body name="b" pos="0.12 0 1"><camera name="r"/></body>
    </worldbody></mujoco>"""
    from twinrobo.plugins.mujoco import MujocoRenderer

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert MujocoRenderer(model, data).baseline("l", "r") == pytest.approx(0.12)


def lidar_spec():
    return LidarSpec.from_dict(
        {
            "type": "lidar",
            "id": "test/spin",
            "scan": {
                "pattern": "spinning",
                "channels": 5,
                "vertical_deg": {"min": -10, "max": 10},
                "columns": 90,
                "frame_hz": 10,
            },
            "range": {"min_m": 0.1, "max_m": 50.0, "at_reflectivity": 0.1},
        }
    )


@pytest.mark.gl
def test_lidar_attaches_to_mujoco(backend):
    rig = LidarTwin(lidar_spec(), "cpu").attach(backend, pos=(0, 0, 0.5))
    obs = rig.capture()
    assert list(obs) == ["lidar"] and obs["lidar"].valid().any()


def test_lidar_catalog_and_alias():
    assert LiDARTwin is LidarTwin
    assert LidarTwin.from_catalog("ouster/os1-64", device="cpu").spec.id == "ouster/os1-64"


def test_device_from_the_catalog_names_its_parts():
    assert {"intel/realsense-d435", "intel/realsense-d455"} <= set(DeviceRegistry().list())
    device = DeviceTwin.from_catalog("intel/realsense-d435", device="cpu", build_psf=False)
    assert isinstance(device.parts["rgb"], CameraTwin)
    assert isinstance(device.parts["ir"], StereoTwin)
    assert device.spec.parts["rgb"].translation_m is None  # not published


def test_device_streams_are_named_by_part():
    be = FakeBackend()
    device = DeviceTwin({"rgb": small_twin(), "ir": small_stereo()})
    rig = device.attach(be, rgb="c", ir={"left": "l", "right": "r"})
    obs = rig.capture()
    assert isinstance(rig, GroupRig) and list(obs) == ["rgb", "ir/left", "ir/right"]
    assert isinstance(obs, Observation)
    assert be.adapters["l"].calls == [{"hold_time": True, "step": False}]  # the first part renders


def test_device_needs_every_part_bound():
    device = DeviceTwin({"rgb": small_twin(), "ir": small_stereo()})
    with pytest.raises(ValueError, match="bind every part"):
        device.attach(FakeBackend(), rgb="c")


def test_device_uses_the_catalog_it_came_from(tmp_path):
    src = ModuleRegistry().load("intel/realsense-d435").source_path.parent / "color"
    root = tmp_path / "mine"
    shutil.copytree(src, root / "my" / "cam")
    spec = root / "my" / "cam" / "camera.yaml"
    spec.write_text(spec.read_text().replace("intel/realsense-d435/color", "my/cam"))
    (root / "my" / "dev").mkdir()
    (root / "my" / "dev" / "device.yaml").write_text(
        "type: device\nid: my/dev\nparts: {rgb: {type: camera, id: my/cam}}\n"
    )
    registry = DeviceRegistry([root])
    device = DeviceTwin.from_catalog("my/dev", registry, device="cpu", build_psf=False)
    assert device.parts["rgb"].spec.id == "my/cam"
    device = DeviceTwin.from_catalog(
        "my/dev", registry, device="cpu", build_psf=False, roots=[root]
    )
    assert device.parts["rgb"].spec.id == "my/cam"


def test_lidar_follows_the_catalog_setting(tmp_path, monkeypatch):
    src = LidarRegistry().resolve("ouster/os1-64")
    (tmp_path / "my" / "lidar").mkdir(parents=True)
    path = tmp_path / "my" / "lidar" / "lidar.yaml"
    path.write_text(src.read_text().replace("ouster/os1-64", "my/lidar"))
    monkeypatch.setenv("TWINROBO_CATALOG", str(tmp_path))
    assert LidarTwin.from_catalog("my/lidar", device="cpu").spec.id == "my/lidar"
    assert LidarTwin.from_spec(path, device="cpu").spec.id == "my/lidar"


def test_device_spec_is_checked(tmp_path):
    p = tmp_path / "device.yaml"
    p.write_text("type: device\nparts: {a: {type: radar, id: x/y}}\n")
    with pytest.raises(SpecError, match="needs an id"):
        DeviceSpec.from_yaml(p)
    with pytest.raises(SpecError, match="not found"):
        DeviceRegistry([tmp_path]).load("no/device")
