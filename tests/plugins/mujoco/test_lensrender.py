"""Lens-ray rendering (pupil raster, ray cast): geometry, occlusion, shading, agreement."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("deeplens")

from twinrobo import CameraSpec, CameraTwin  # noqa: E402

SPEC = Path(__file__).resolve().parent.parent.parent / "data" / "cellphone80deg" / "camera.yaml"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def small_twin(w=320, h=200, focus=3.0):
    spec = CameraSpec.from_yaml(SPEC, width=w, height=h, focus_distance_m=focus)
    return CameraTwin.from_spec(spec, device=DEV, build_psf=False)


@pytest.fixture(scope="module")
def rays():
    from twinrobo.plugins.mujoco.lensrender import lens_rays

    return lens_rays(small_twin().reference, 32)


def test_lens_rays_orientation_distortion_illumination(rays):
    K = rays.intrinsics
    W, H = rays.resolution
    ct = rays.chief_tan
    # upright image: top-right pixel looks right (+x) and up (+y)
    assert ct[5, W - 5, 0] > 0 and ct[5, W - 5, 1] > 0
    assert ct[H - 5, 5, 0] < 0 and ct[H - 5, 5, 1] < 0
    # barrel distortion of this lens: the field edge sees wider than the paraxial pinhole
    edge = (W - 1 - K.cx) / K.fx
    assert float(ct[int(K.cy), W - 1, 0]) > 1.05 * edge
    # relative illumination: 1 on axis, falls off to the corners, for every wavelength
    illum = rays.illumination
    assert torch.allclose(
        illum[:, int(K.cy), int(K.cx)], torch.ones(3, device=illum.device), atol=0.05
    )
    assert bool((illum[:, 0, 0] < 0.5).all())
    # rays leave from within the entrance pupil (the stop, r = 1.09 mm)
    assert 1.0e-3 < rays.pupil_radius_m < 1.2e-3


def test_rectify_grid_inverts_the_chief_ray_map(rays):
    import torch.nn.functional as F

    grid = rays.rectify_grid()
    chief = rays.chief_tan.permute(2, 0, 1)[None]
    got = F.grid_sample(chief, grid, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    want = rays.pinhole_tan()
    inner = (slice(20, -20), slice(30, -30))  # away from the border (outside the lens field)
    err = (got - want)[inner].abs().max().item()
    assert err < 2e-3, err  # < 0.5 px at fx ~ 254


def test_with_geometry_takes_the_calibration_and_keeps_the_blur(rays):
    from twinrobo.geometry import CameraIntrinsics
    from twinrobo.optics.distortion import RadialDistortion

    K0 = rays.intrinsics
    K = CameraIntrinsics(K0.width, K0.height, K0.fx * 1.1, K0.fy * 1.1, K0.cx + 2, K0.cy - 1)
    dist = RadialDistortion(k1=-0.05)
    cal = rays.with_geometry(K, dist)
    assert cal.intrinsics == K and cal.metadata["geometry"] == "calibration"
    # chief rays follow the calibration: projecting them lands on their own pixels
    t = cal.chief_tan
    xd, yd = dist.distort(t[..., 0], -t[..., 1])
    vv, uu = torch.meshgrid(
        torch.arange(K.height, device=t.device), torch.arange(K.width, device=t.device),
        indexing="ij",
    )  # fmt: skip
    assert float((xd * K.fx + K.cx - uu).abs().max()) < 0.05
    assert float((yd * K.fy + K.cy - vv).abs().max()) < 0.05
    # every ray of a pixel turned by the same angle: the spread around the chief ray is kept
    row = slice(40, 60)
    _, t0, w0 = rays.rays(1, row)
    _, t1, _ = cal.rays(1, row)
    ok = (w0 > 0)[..., None]
    spread0 = (t0 - rays.chief_tan[row][:, :, None]).where(ok, 0)
    spread1 = (t1 - cal.chief_tan[row][:, :, None]).where(ok, 0)
    assert float((spread0 - spread1).abs().max()) < 2e-3  # fp16 residuals


def test_catalog_surrogate_lenses_take_the_catalog_geometry():
    twin = CameraTwin.from_catalog("intel/realsense-d455/color", device=DEV, build_psf=False)
    geo = twin.lens_geometry()
    assert geo["intrinsics"] == twin.intrinsics
    assert small_twin().lens_geometry() == {}  # physical distortion: the lens' own


# -- rendering (needs MuJoCo + GL) -----------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")

SCENE = """
<mujoco>
  <visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
  <asset>
    <material name="white" emission="1" rgba="1 1 1 1" specular="0"/>
    <material name="black" rgba="0 0 0 1" specular="0"/>
    <material name="red" emission="1" rgba="1 0 0 1" specular="0"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="1 0 0 0 1 0"/>
    <geom type="box" pos="0 0 -3" size="5 5 0.001" material="white"/>
    {extra}
  </worldbody>
</mujoco>
"""
MARKER = '<geom type="box" pos="1.5 0 -2.99" size="0.01 0.01 0.001" material="red"/>'


def backend(extra):
    from twinrobo.plugins.mujoco import MujocoRenderer

    m = mujoco.MjModel.from_xml_string(SCENE.format(extra=extra))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return MujocoRenderer(m, d)


@pytest.mark.gl
@pytest.mark.parametrize("method", ["pupil", "raycast"])
def test_distortion_places_points_where_the_lens_does(method):
    from twinrobo.plugins.mujoco import MujocoCameraTwin

    twin = small_twin()
    be = backend(MARKER)
    cam = MujocoCameraTwin(twin, be, camera="cam", render=method, rays_per_pixel=8)
    img = cam.get_frame().rgb[0].cpu().numpy()
    ys, xs = np.nonzero(img[0] - img[1] > 0.3)
    ct = cam.lens_renderer.rays.chief_tan[100].cpu().numpy()[:, 0]
    expected = np.interp(1.5 / 2.99, ct, np.arange(ct.size))
    assert abs(xs.mean() - expected) < 0.8
    # the paraxial pinhole (PSF path without a distortion model) would put it ~5 px further out
    K = cam.lens_renderer.rays.intrinsics
    assert K.cx + (1.5 / 2.99) * K.fx - expected > 3


@pytest.mark.gl
@pytest.mark.parametrize("half_width", [0.0015, 0.003])  # 3 mm and 6 mm posts (2.5 / 5 px)
def test_defocused_occluder_edges_match_traced_rays(rays, half_width):
    """A post in front of the focus plane, blurred by the aperture: its partially covered edges.

    The exact answer comes from the traced rays alone (which pass beside the post);
    the renderers must reproduce it, and must beat the pinhole + PSF approximation.
    Occluders narrower than a sensor pixel are a known limit (MuJoCo's multisampled
    silhouettes blend them into the background in every view); see lensrender.
    """
    from twinrobo.plugins.mujoco import MujocoCameraTwin
    from twinrobo.plugins.mujoco.lensrender import LensRayRenderer

    post = f'<geom type="box" pos="0 0 -0.3" size="{half_width} 1 0.001" material="black"/>'
    be = backend(post)
    o, t, w = rays.rays(1, slice(100, 101))
    x = o[0, :, :, 0] + t[0, :, :, 0] * 0.3  # every ray's x at the post depth
    exact = (((x.abs() > half_width).float() * w[0]).sum(-1) / w[0].sum(-1)).cpu().numpy()
    cols = slice(140, 180)
    assert ((exact[cols] > 0.05) & (exact[cols] < 0.95)).any()  # partially covered edge pixels
    errs = {}
    for method in ("raycast", "pupil"):
        img, _ = LensRayRenderer(rays, method, 7, oversample=2).render(be, "cam", 0)
        errs[method] = np.abs(img[0, 1, 100].cpu().numpy()[cols] - exact[cols]).max()
    twin = small_twin()
    twin_psf = CameraTwin.from_spec(twin.spec, device=DEV)  # same camera, PSF renderer
    psf = MujocoCameraTwin(twin_psf, be, camera="cam", render="psf").get_frame().rgb
    errs["psf"] = np.abs(psf[0, 1, 100].cpu().numpy()[cols] - exact[cols]).max()
    # measured ~0.05 (residual: MuJoCo's multisampled silhouette colors); PSF ~0.2-0.4
    assert errs["raycast"] < 0.08, errs
    assert errs["pupil"] < 0.15, errs
    assert errs["psf"] > 2 * errs["raycast"], errs


@pytest.mark.gl
def test_methods_agree_and_shading_modes(rays):
    from twinrobo.plugins.mujoco.lensrender import LensRayRenderer

    be = backend(MARKER)
    a, _ = LensRayRenderer(rays, "pupil", 7).render(be, "cam", 0)
    b, depth = LensRayRenderer(rays, "raycast", 7).render(be, "cam", 0)
    mse = torch.mean((a - b) ** 2).item()
    assert 10 * np.log10(1 / mse) > 40
    # uniform emissive wall: corrected shading is flat, raw shows relative illumination
    assert abs(float(b[0, 1, 100, 100]) - 1) < 0.02
    raw, _ = LensRayRenderer(rays, "raycast", 7, shading="raw").render(be, "cam", 0)
    assert float(raw[0, 1, 2, 2]) < 0.5 * float(raw[0, 1, 100, 160])
    # chief-ray depth: the wall at 3 m
    assert abs(float(depth[0, 0, 100, 160]) - 3.0) < 0.02


@pytest.mark.gl
def test_camera_offset_is_restored():
    from twinrobo.plugins.mujoco.lensrender import _camera_offset

    be = backend("")
    before = np.array(be.data.cam_xpos[0])
    with _camera_offset(be.model, be.data, 0, (0.001, 0, 0)):
        assert not np.allclose(be.data.cam_xpos[0], before)
    assert np.allclose(be.data.cam_xpos[0], before)
