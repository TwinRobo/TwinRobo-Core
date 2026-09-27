"""MuJoCo adapter on a tiny scene (needs MuJoCo + an offscreen GL backend, e.g. EGL)."""

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported
mujoco = pytest.importorskip("mujoco")

from twinrobo import CameraTwin  # noqa: E402
from twinrobo.geometry import CameraIntrinsics  # noqa: E402
from twinrobo.mujoco import MujocoCameraTwin, MujocoRenderer, to_uint8  # noqa: E402

pytestmark = pytest.mark.gl

# A wall 2 m in front of the camera, a red marker above the axis (top of the image) and a
# green marker at +x (right of the image, for a camera looking down -z with x to the right).
XML = """
<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.3 0.3 0.3"/></visual>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <geom type="box" pos="0 0 -2" size="5 5 0.001" rgba="0.5 0.5 0.5 1"/>
    <geom type="box" pos="0 0.4 -1.9" size="0.1 0.1 0.01" rgba="1 0 0 1"/>
    <geom type="box" pos="0.5 0 -1.9" size="0.1 0.1 0.01" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    backend = MujocoRenderer(model, data)
    yield model, data, backend
    backend.close()


def twin(W=64, H=40, vfov=40.0):
    fy = H / 2 / np.tan(np.radians(vfov) / 2)
    return CameraTwin(intrinsics=CameraIntrinsics(W, H, fy, fy, (W - 1) / 2, (H - 1) / 2))


def test_depth_is_metric_and_image_upright(scene):
    model, data, backend = scene
    cam = MujocoCameraTwin(twin(), backend, camera="cam", device="cpu")
    rgb, depth = cam.render_ideal()
    assert rgb.shape == (40, 64, 3) and depth.shape == (40, 64)
    # Wall at z-depth 2.0 m (corners too: z-depth, not ray distance).
    assert abs(float(depth[0, 0]) - 2.0) < 0.01 and abs(float(depth[20, 5]) - 2.0) < 0.01
    red = (rgb[..., 0].astype(int) - rgb[..., 1]) > 100
    rows = np.nonzero(red.any(1))[0]
    assert rows.size and rows.mean() < 20  # marker above the axis appears in the top half


def test_image_not_mirrored(scene):
    """+x in the camera frame appears on the right: the adapter's images are not mirrored."""
    model, data, backend = scene
    cam = MujocoCameraTwin(twin(), backend, camera="cam", device="cpu")
    rgb, _ = cam.render_ideal()
    green = (rgb[..., 1].astype(int) - rgb[..., 0]) > 100
    cols = np.nonzero(green.any(0))[0]
    assert cols.size and cols.mean() > 32


def test_fovy_is_matched_then_restored(scene):
    model, data, backend = scene
    cam = MujocoCameraTwin(twin(vfov=40.0), backend, camera="cam", device="cpu")
    _, depth = cam.render_ideal()
    assert float(model.cam_fovy[0]) == pytest.approx(60.0)  # restored
    # Marker (y = 0.4 at z = 1.9) projects at fy * 0.4 / 1.9 pixels above center.
    rgb, _ = cam.render_ideal()
    red = (rgb[..., 0].astype(int) - rgb[..., 1]) > 100
    fy = 20 / np.tan(np.radians(20))
    expected_row = 19.5 - fy * 0.4 / 1.9
    assert abs(np.nonzero(red.any(1))[0].mean() - expected_row) < 1.5


def test_identity_twin_reproduces_render(scene):
    model, data, backend = scene
    cam = MujocoCameraTwin(twin(), backend, camera="cam", device="cpu")
    rgb8, _ = cam.render_ideal()
    frame = cam.get_frame()
    out = to_uint8(frame.rgb)
    assert np.abs(out.astype(int) - rgb8).max() <= 1  # sRGB round trip only


def test_to_uint8_center_crop():
    x = torch.zeros(1, 3, 40, 64)
    x[..., :, 12:52] = 1.0  # the centered 40x40 square is white
    out = to_uint8(x, size=(20, 20), crop="center")
    assert out.shape == (20, 20, 3) and out.min() == 255


def test_mount_orientation_convention():
    from twinrobo.mujoco.mounts import CameraMount

    look = lambda m: -m.rotation()[:, 2]  # noqa: E731  (MuJoCo cameras look along -z)
    np.testing.assert_allclose(look(CameraMount("a")), [1, 0, 0], atol=1e-9)  # body +x
    np.testing.assert_allclose(CameraMount("a").rotation()[:, 1], [0, 0, 1], atol=1e-9)  # up +z
    np.testing.assert_allclose(look(CameraMount("a", rpy_deg=(90, 0, 0))), [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(look(CameraMount("a", rpy_deg=(0, 90, 0))), [0, 0, -1], atol=1e-9)
    with pytest.raises(ValueError):
        CameraMount("bad name!")


def test_distortion_twin_renders_the_source_and_warps():
    """A distortion twin renders its wider source pinhole; the warp pulls a marker inward."""
    from twinrobo.optics.distortion import RadialDistortion

    xml = XML.replace(
        '<geom type="box" pos="0.5 0 -1.9"', '<geom type="box" pos="1.5 0 -1.9"'
    )  # the green marker far off-axis (normalized x ~ 0.79)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    backend = MujocoRenderer(model, data)
    try:
        W, H, fy = 160, 100, 86.6
        intr = CameraIntrinsics(W, H, fy, fy, (W - 1) / 2, (H - 1) / 2)
        dist = RadialDistortion(k1=-0.1)
        tw = CameraTwin(intrinsics=intr, distortion=dist)
        cam = MujocoCameraTwin(tw, backend, camera="cam", device="cpu")
        src = tw.render_intrinsics
        rgb, _ = cam.render_ideal()
        assert rgb.shape[:2] == (src.height, src.width) and src.width > W
        fr = cam.get_frame()
        assert fr.rgb.shape[-2:] == (H, W) == fr.rgb_ideal.shape[-2:]

        def green_u(img):
            g = img[0, 1] - img[0, 0]
            cols = torch.nonzero((g > 0.3).any(0)).flatten().float()
            return float(cols.mean())

        x = 1.5 / 1.9
        expect_pin = intr.cx + fy * x
        expect_dist = intr.cx + fy * float(dist.distort(np.array(x), np.array(0.0))[0])
        assert green_u(fr.rgb_ideal) == pytest.approx(expect_pin, abs=1.5)
        assert green_u(fr.rgb) == pytest.approx(expect_dist, abs=1.5)
        assert expect_pin - expect_dist > 3  # a clearly visible barrel shift
    finally:
        backend.close()
