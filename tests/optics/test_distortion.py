import math

import numpy as np
import pytest
import torch

from twinrobo import CameraTwin
from twinrobo.geometry import CameraIntrinsics
from twinrobo.optics.distortion import DistortionWarp, RadialDistortion, field_of_view
from twinrobo.registry import CatalogRegistry

INTR = CameraIntrinsics(160, 100, 80.0, 80.0, 79.5, 49.5)
BARREL = RadialDistortion(k1=-0.06, k2=0.007, p1=0.001, p2=-0.0005)


def test_known_radial_point():
    d = RadialDistortion(k1=-0.1)
    xd, yd = d.distort(np.array(0.5), np.array(0.0))
    assert xd == pytest.approx(0.5 * (1 - 0.1 * 0.25)) and yd == 0


def test_distort_undistort_round_trip():
    x, y = np.meshgrid(np.linspace(-1.4, 1.4, 41), np.linspace(-0.9, 0.9, 31))
    xu, yu = BARREL.undistort(*BARREL.distort(x, y))
    assert np.abs(xu - x).max() < 1e-9 and np.abs(yu - y).max() < 1e-9


def test_from_dict_only_parametric_models():
    assert RadialDistortion.from_dict({"model": "physical"}) is None
    assert RadialDistortion.from_dict(None) is None
    d = RadialDistortion.from_dict({"model": "opencv", "k1": -0.1, "k3": None})
    assert d == RadialDistortion(k1=-0.1)


def test_zero_coefficients_keep_the_pinhole_path():
    twin = CameraTwin(intrinsics=INTR, distortion=RadialDistortion())
    assert twin.warp is None and twin.render_intrinsics == INTR
    rgb = torch.rand(1, 3, INTR.height, INTR.width)
    depth = torch.rand(1, 1, INTR.height, INTR.width) + 0.5
    fr = twin.process(rgb, depth)
    assert torch.equal(fr.rgb, rgb) and torch.equal(fr.rgb_ideal, rgb)


def _pattern(x, y):
    """A smooth scene defined on undistorted normalized coordinates."""
    return 0.5 + 0.25 * torch.sin(7 * x) * torch.cos(5 * y)


def test_warp_matches_analytic_scene():
    warp = DistortionWarp(INTR, BARREL)
    src = warp.source
    assert src.width > INTR.width and src.height > INTR.height and src.fy == INTR.fy
    v, u = torch.meshgrid(
        torch.arange(src.height, dtype=torch.float64),
        torch.arange(src.width, dtype=torch.float64),
        indexing="ij",
    )
    img = _pattern((u - src.cx) / src.fx, (v - src.cy) / src.fy).float()[None, None]
    out, _ = warp(img.expand(1, 3, -1, -1), torch.ones(1, 1, src.height, src.width))
    v, u = torch.meshgrid(
        torch.arange(INTR.height, dtype=torch.float64),
        torch.arange(INTR.width, dtype=torch.float64),
        indexing="ij",
    )
    x, y = BARREL.undistort((u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy)
    ref = _pattern(x, y).float()
    assert (out[0, 0] - ref).abs().max() < 5e-3  # bilinear sampling error only
    # the pinhole view is an exact crop when the source shares the focal length
    ph = warp.pinhole(img)
    x0, y0 = int(src.cx - INTR.cx), int(src.cy - INTR.cy)
    assert torch.allclose(ph, img[..., y0 : y0 + INTR.height, x0 : x0 + INTR.width], atol=1e-6)


def test_depth_is_warped_nearest():
    warp = DistortionWarp(INTR, BARREL)
    src = warp.source
    depth = torch.ones(1, 1, src.height, src.width)
    depth[..., :, src.width // 2 :] = 3.0  # a depth edge
    _, d = warp(torch.zeros(1, 3, src.height, src.width), depth)
    assert set(d.unique().tolist()) <= {1.0, 3.0}  # no interpolated mid-depths


def test_twin_warps_source_sized_input_only():
    twin = CameraTwin(intrinsics=INTR, distortion=BARREL)
    src = twin.render_intrinsics
    fr = twin.process(
        torch.rand(1, 3, src.height, src.width), torch.ones(1, 1, src.height, src.width)
    )
    assert fr.rgb.shape[-2:] == (INTR.height, INTR.width) == fr.rgb_ideal.shape[-2:]
    assert fr.depth.shape[-2:] == (INTR.height, INTR.width)
    # sensor-sized input (e.g. the scene camera's own FoV): blur only, no warp
    fr = twin.process(
        torch.rand(1, 3, INTR.height, INTR.width), torch.ones(1, 1, INTR.height, INTR.width)
    )
    assert fr.rgb.shape[-2:] == (INTR.height, INTR.width)


@pytest.mark.parametrize(
    "cid, vendor",
    [("stereolabs/zed-x/2.2mm", (110, 80, 120)), ("stereolabs/zed-x/4mm", (75, 50, 83))],
)
def test_zed_x_fov_matches_datasheet(cid, vendor):
    spec = CatalogRegistry(use_env=False).load(cid)
    i = spec.calibration["intrinsic"]
    res = spec.sensor.resolution
    intr = CameraIntrinsics(res.width, res.height, i["fx"], i["fy"], i["cx"], i["cy"])
    fov = field_of_view(intr, RadialDistortion.from_dict(spec.calibration["distortion"]))
    for got, want in zip((fov["h"], fov["v"], fov["d"]), vendor, strict=True):
        assert got == pytest.approx(want, abs=1.0)  # datasheet rounds to whole degrees
    assert spec.lens.focal_length_mm * 1000 / spec.sensor.pixel_pitch_um == pytest.approx(
        i["fx"], rel=0.06
    )  # nominal focal length vs the fitted one


def test_resolution_override_keeps_the_field_of_view():
    from twinrobo import CameraSpec, CatalogRegistry

    path = CatalogRegistry().resolve("stereolabs/zed-x/2.2mm")
    spec = CameraSpec.from_yaml(path, width=960, height=600)
    i = spec.calibration["intrinsic"]
    intr = CameraIntrinsics(960, 600, i["fx"], i["fy"], i["cx"], i["cy"])
    fov = field_of_view(intr, RadialDistortion.from_dict(spec.calibration["distortion"]))
    assert fov["d"] == pytest.approx(120, abs=0.2) and i["cx"] == pytest.approx(479.5)
    assert math.isclose(spec.sensor.pixel_pitch_um, 6.0)


def test_non_invertible_distortion_is_rejected():
    with pytest.raises(ValueError, match="not invertible"):
        DistortionWarp(INTR, RadialDistortion(k1=-0.3))  # radial curve peaks inside the sensor
