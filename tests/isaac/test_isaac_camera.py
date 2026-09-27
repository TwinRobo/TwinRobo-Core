"""IsaacCameraTwin inside Isaac Sim (run with docker/isaac/run.sh -m pytest tests/isaac).

A synthetic stage with known geometry: a camera at the origin looking along -Z, a
white wall 3 m away and a small red marker at a known 3D point.
"""

import importlib.util

import numpy as np
import pytest
import torch

if importlib.util.find_spec("isaacsim") is None:
    pytest.skip("needs NVIDIA Isaac Sim (docker/isaac/run.sh)", allow_module_level=True)

pytestmark = [pytest.mark.isaac, pytest.mark.gpu]

WALL_Z = -3.0  # wall center; its front face is 1 cm nearer
MARKER = (0.4, 0.25, -2.5)  # marker center; front face 2.5 cm nearer
MARKER_SIZE = 0.05
NEAR = (-0.1, -0.06, -0.3)  # a small blue cube close to the camera


@pytest.fixture(scope="module")
def stage(app):
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Camera.Define(stage, "/World/Cam")  # at the origin, looking along -Z, +Y up

    def box(path, pos, scale, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))

    box("/World/Wall", (0, 0, WALL_Z), (20, 20, 0.02), (0.8, 0.8, 0.8))
    box("/World/Marker", MARKER, (MARKER_SIZE,) * 3, (1, 0, 0))
    box("/World/Near", NEAR, (0.02, 0.02, 0.02), (0, 0, 1))  # 30 cm away, inside USD's 1 m clip
    UsdLux.DomeLight.Define(stage, "/World/Dome").GetIntensityAttr().Set(1000)
    for _ in range(3):
        app.update()
    return stage


@pytest.fixture(scope="module")
def twin():
    from twinrobo import CameraSpec, CameraTwin
    from twinrobo.registry import BUILTIN_CATALOG

    spec = CameraSpec.from_yaml(
        BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml", width=320, height=200
    )
    return CameraTwin.from_spec(spec, device="cuda")


def marker_centroid(rgb):
    img = rgb[0].permute(1, 2, 0).cpu().numpy()
    ys, xs = np.nonzero((img[..., 0] > 0.3) & (img[..., 1] < 0.5 * img[..., 0]))
    assert len(xs) > 3, "marker not visible"
    return xs.mean(), ys.mean()


def render(cam, frames=4):
    for _ in range(frames - 1):  # let RTX settle on the new camera
        cam.render_ideal()
    return cam.render_ideal()


def test_projection_matches_the_twin_intrinsics(stage, twin):
    from twinrobo.isaac import IsaacCameraTwin

    cam = IsaacCameraTwin(twin, "/World/Cam", rt_subframes=4)
    try:
        rgb, depth = render(cam)
        k = twin.render_intrinsics
        assert tuple(rgb.shape) == (1, 3, k.height, k.width)
        z = -(MARKER[2] + MARKER_SIZE / 2)  # distance of the marker's front face
        u = k.cx + k.fx * MARKER[0] / z
        v = k.cy - k.fy * MARKER[1] / z  # image rows go down, +Y goes up
        cu, cv = marker_centroid(rgb)
        assert abs(cu - u) < 1.0 and abs(cv - v) < 1.0, (cu, cv, u, v)
        # metric z-depth: the wall's front face, and the marker's
        wall = depth[0, 0, :20, :20]
        assert torch.allclose(wall, torch.full_like(wall, -WALL_Z - 0.01), rtol=1e-3)
        assert abs(float(depth[0, 0, int(round(cv)), int(round(cu))]) - z) < 1e-2
        # close-up objects are seen (the view's near clip is 1 cm, not USD's 1 unit)
        zn = -(NEAR[2] + 0.01)
        un, vn = round(k.cx + k.fx * NEAR[0] / zn), round(k.cy - k.fy * NEAR[1] / zn)
        assert abs(float(depth[0, 0, vn, un]) - zn) < 1e-2
    finally:
        cam.close()


def test_frame_goes_through_the_optics(stage, twin):
    from twinrobo.isaac import IsaacCameraTwin

    cam = IsaacCameraTwin(twin, "/World/Cam", rt_subframes=4)
    try:
        render(cam)
        fr = cam.get_frame()
        assert tuple(fr.rgb.shape) == tuple(fr.rgb_ideal.shape) == (1, 3, 200, 320)
        assert bool(torch.isfinite(fr.rgb).all())
        mse = torch.mean((fr.rgb.clamp(0, 1) - fr.rgb_ideal.clamp(0, 1)) ** 2).item()
        assert 0 < mse and 10 * np.log10(1 / mse) > 20  # blurred, not a different picture
        assert fr.metadata["camera"] == "/World/Cam"
    finally:
        cam.close()


def test_match_fov_false_keeps_the_cameras_lens_and_close_cleans_up(stage, twin):
    from pxr import UsdGeom

    from twinrobo.isaac import IsaacCameraTwin

    src = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
    src.GetFocalLengthAttr().Set(18.0)
    cam = IsaacCameraTwin(twin, "/World/Cam", match_fov=False, rt_subframes=4)
    try:
        rgb, _ = render(cam)
        W, H = cam.resolution
        f = 18.0 / src.GetHorizontalApertureAttr().Get() * W  # the camera's own fx, in pixels
        z = -(MARKER[2] + MARKER_SIZE / 2)
        cu, cv = marker_centroid(rgb)
        assert abs(cu - ((W - 1) / 2 + f * MARKER[0] / z)) < 1.0
        assert abs(cv - ((H - 1) / 2 - f * MARKER[1] / z)) < 1.0
    finally:
        cam.close()
    assert not stage.GetPrimAtPath("/World/Cam/TwinRoboView").IsValid()
    assert src.GetFocalLengthAttr().Get() == 18.0  # the camera itself is never changed
