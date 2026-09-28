"""Lens-ray rendering (ray cast, pupil raster) in Isaac Sim, as validated for MuJoCo.

Run with docker/isaac/run.sh -m pytest tests/plugins/isaac. A camera at the origin looks
along -Z at a matte wall 3 m away.
"""

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("isaacsim") is None:
    pytest.skip("needs NVIDIA Isaac Sim (docker/isaac/run.sh)", allow_module_level=True)

pytestmark = [pytest.mark.isaac, pytest.mark.gpu]

MARKER = (1.45, 0.0, -2.9)  # right of center (x/z ~ 0.5), on the image's middle row
POST_Z = 0.3  # a thin black post 30 cm in front of the camera (focus: 3 m)


def small_twin(build_psf=False):
    from twinrobo import CameraSpec, CameraTwin
    from twinrobo.registry import BUILTIN_CATALOG

    spec = CameraSpec.from_yaml(
        BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml",
        width=320,
        height=200,
        focus_distance_m=3.0,
    )
    return CameraTwin.from_spec(spec, device="cuda", build_psf=build_psf)


def new_stage(app, *boxes):
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Camera.Define(stage, "/World/Cam")
    for i, (pos, scale, color) in enumerate(
        [((0, 0, -3.0), (20, 20, 0.02), (0.8, 0.8, 0.8)), *boxes]
    ):
        cube = UsdGeom.Cube.Define(stage, f"/World/Box{i}")
        cube.GetSizeAttr().Set(1.0)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdLux.DomeLight.Define(stage, "/World/Dome").GetIntensityAttr().Set(1000)
    for _ in range(3):
        app.update()
    return stage


def frame_of(cam, settle=3):
    for _ in range(settle):
        cam.get_frame()
    return cam.get_frame()


@pytest.mark.parametrize("method", ["raycast", "pupil"])
def test_distortion_places_points_where_the_lens_does(app, method):
    from twinrobo.plugins.isaac import IsaacCameraTwin

    new_stage(app, (MARKER, (0.04, 0.04, 0.002), (1, 0, 0)))  # flat: no side face in view
    twin = small_twin()
    cam = IsaacCameraTwin(twin, "/World/Cam", render=method, rt_subframes=4)
    try:
        fr = frame_of(cam)
        img = fr.rgb[0].permute(1, 2, 0).cpu().numpy()
        ys, xs = np.nonzero((img[..., 0] > 0.3) & (img[..., 1] < 0.5 * img[..., 0]))
        assert len(xs) > 3
        rays = cam.lens_renderer.rays
        K = rays.intrinsics
        ct = rays.chief_tan[int(round(ys.mean()))].cpu().numpy()[:, 0]
        z = -(MARKER[2] + 0.001)
        expected = np.interp(MARKER[0] / z, ct, np.arange(ct.size))
        print(f"{method}: marker at {xs.mean():.2f} px, lens chief ray {expected:.2f} px")
        assert abs(xs.mean() - expected) < 0.8, (xs.mean(), expected)
        assert K.cx + MARKER[0] / z * K.fx - expected > 3  # a pinhole would be well off
        assert fr.metadata["render"] == method and fr.metadata["views"] == 7
    finally:
        cam.close()


def test_raycast_defocused_occluder_edges_match_the_traced_rays(app):
    """A 6 mm post 0.3 m away, far in front of the 3 m focus: its blurred edges are exact."""
    from twinrobo.plugins.isaac import IsaacCameraTwin

    half = 0.003
    new_stage(app, ((0, 0, -POST_Z), (2 * half, 2.0, 0.002), (0, 0, 0)))
    twin = small_twin()
    cam = IsaacCameraTwin(twin, "/World/Cam", render="raycast", rt_subframes=4, view_oversample=2)
    try:
        fr = frame_of(cam)
        rays = cam.lens_renderer.rays
        o, t, w = rays.rays(1, slice(100, 101))
        x = o[0, :, :, 0] + t[0, :, :, 0] * POST_Z  # every ray's x at the post
        exact = (((x.abs() > half).float() * w[0]).sum(-1) / w[0].sum(-1)).cpu().numpy()
        cols = slice(140, 180)
        assert ((exact[cols] > 0.05) & (exact[cols] < 0.95)).any()  # partially covered pixels
        row = fr.rgb[0, 1, 100].cpu().numpy()
        wall = np.median(row[20:60])  # the wall's own brightness, away from the post
        err_ray = np.abs(row[cols] / wall - exact[cols]).max()
    finally:
        cam.close()
    psf = IsaacCameraTwin(small_twin(build_psf=True), "/World/Cam", rt_subframes=4)
    try:
        prow = frame_of(psf).rgb[0, 1, 100].cpu().numpy()
        err_psf = np.abs(prow[cols] / np.median(prow[20:60]) - exact[cols]).max()
    finally:
        psf.close()
    print(f"defocused edge error: raycast {err_ray:.3f}, pinhole + PSF {err_psf:.3f}")
    assert err_ray < 0.12, (err_ray, err_psf)  # MuJoCo measures ~0.05
    assert err_psf > 2 * err_ray, (err_ray, err_psf)


def test_close_removes_the_pupil_views(app):
    from twinrobo.plugins.isaac import IsaacCameraTwin

    stage = new_stage(app)
    cam = IsaacCameraTwin(small_twin(), "/World/Cam", render="raycast", pupil_views=3)
    frame_of(cam, settle=0)
    assert stage.GetPrimAtPath("/World/Cam/TwinRoboPupil2").IsValid()
    cam.close()
    kids = [p.GetName() for p in stage.GetPrimAtPath("/World/Cam").GetChildren()]
    assert kids == [], kids
