"""DeepLens adapter: loading (Phase 0 exit), PSF banks, cache, and agreement with the reference."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("deeplens")

import yaml  # noqa: E402

from twinrobo import CameraSpec, CameraTwin  # noqa: E402
from twinrobo.optics import DenseDepthRenderer, PSFCache  # noqa: E402
from twinrobo.optics.deeplens import DeepLensOptics  # noqa: E402
from twinrobo.utils.synthetic import slanted_plane  # noqa: E402
from twinrobo.validation.optics import compare_to_reference  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def optics(request):

    return DeepLensOptics(
        Path(__file__).resolve().parent.parent / "data" / "cellphone80deg" / "lens.json",
        sensor_resolution=(320, 200),
        focus_distance_m=2.0,
        device=DEVICE,
        psf_ks=33,
    )


def test_loads(optics):
    assert optics.sensor_resolution == (320, 200)
    # DeepLens paraxial EFL; the lens-file header (4.35) is the distorted edge-of-field value
    assert 5.0 < optics.focal_length_mm < 5.3
    assert 2.2 < optics.f_number < 2.5


def test_psf_bank(optics):
    psf = optics.generate_psf([[0.0, 0.0], [0.7, -0.3]], [1.0, 10.0], ks=32, spp=1024)
    assert psf.shape == (2, 2, 3, 32, 32)  # explicit even ks still honored
    assert psf.device.type == DEVICE
    torch.testing.assert_close(
        psf.sum((-1, -2)), torch.ones(2, 2, 3, device=psf.device), atol=1e-3, rtol=0
    )


def test_psf_rejects_bad_input(optics):
    with pytest.raises(ValueError):
        optics.generate_psf([[1.5, 0.0]], [1.0])
    with pytest.raises(ValueError):
        optics.generate_psf([[0.0, 0.0]], [0.0])


def test_default_psf_is_centered(optics):
    """On-axis PSF centroid sits on index ks // 2 (odd ks), i.e. no half-pixel image shift."""
    p = optics.generate_psf([[0.0, 0.0]], [2.0], spp=8192)[0, 0, 1]
    ks = p.shape[-1]
    assert ks % 2 == 1
    idx = torch.arange(ks, device=p.device, dtype=p.dtype)
    assert abs(float((p.sum(0) * idx).sum()) - ks // 2) < 0.1
    assert abs(float((p.sum(1) * idx).sum()) - ks // 2) < 0.1


def test_sample_depths_includes_focus(optics):
    d = optics.sample_depths(0.3, 20.0, 8)
    assert d[0] == pytest.approx(20.0) and d[-1] == pytest.approx(0.3, rel=1e-5)
    assert (d[1:] < d[:-1]).all()  # far -> near
    assert torch.isclose(d, torch.tensor(2.0)).any()


def test_build_psf_bank(optics):
    bank = optics.build_psf_bank(grid=(3, 2), near_m=0.5, far_m=10.0, num_depths=3, spp=1024)
    assert bank.psfs.shape == (3, 2, 3, 3, 33, 33)
    assert bank.sensor_resolution == (320, 200)
    assert bank.metadata["lens_sha256"] and bank.metadata["focus"] == 2.0


def _small_spec(example_spec_path):
    raw = yaml.safe_load(example_spec_path.read_text())
    raw["sensor"]["resolution"] = {"width": 160, "height": 100}
    raw["calibration"]["psf"].update(
        {"field_samples": [2, 2], "depth_samples": 3, "ks": 17, "spp": 1024}
    )
    return CameraSpec.from_dict(raw, base_dir=example_spec_path.parent)


def test_from_spec_builds_then_reuses_cached_bank(example_spec_path, tmp_path):
    spec = _small_spec(example_spec_path)
    cache = PSFCache(tmp_path)
    cam = CameraTwin.from_spec(spec, device=DEVICE, cache=cache)
    assert isinstance(cam.optics, DenseDepthRenderer)
    assert cam.optics.bank.sensor_resolution == (160, 100)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    cam2 = CameraTwin.from_spec(spec, device=DEVICE, cache=cache)
    torch.testing.assert_close(cam2.optics.bank.psfs, cam.optics.bank.psfs)  # loaded, not rebuilt
    frame = cam.process(torch.rand(1, 3, 100, 160), torch.full((1, 1, 100, 160), 2.0))
    assert frame.rgb.shape == (1, 3, 100, 160)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_render_reference_handles_invalid_depth(optics):
    W, H = optics.sensor_resolution
    rgb = torch.rand(1, 3, H, W, device=DEVICE)
    depth = torch.full((1, 1, H, W), 2.0, device=DEVICE)
    depth[..., :, : W // 2] = float("inf")  # simulator background
    out = optics.render_reference(rgb, depth, psf_grid=(4, 4), num_layers=3)
    assert out.shape == rgb.shape and torch.isfinite(out).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runtime_renderer_matches_reference(optics):
    """Phase 1 exit check: the runtime renderer reproduces the DeepLens reference."""
    bank = optics.build_psf_bank(grid=(4, 3), near_m=0.3, far_m=20.0, num_depths=8)
    W, H = optics.sensor_resolution
    rgb, depth = slanted_plane(H, W, 0.4, 15.0, DEVICE)
    cmp = compare_to_reference(DenseDepthRenderer(bank), optics, rgb, depth)
    assert cmp["psnr_vs_reference"] > 35.0
    assert cmp["psnr_vs_reference"] > cmp["psnr_ideal_vs_reference"] + 8.0


def test_infinity_focus(optics):

    inf = DeepLensOptics(
        Path(__file__).resolve().parent.parent / "data" / "cellphone80deg" / "lens.json",
        sensor_resolution=(320, 200),
        focus_distance_m=float("inf"),
        device=DEVICE,
    )
    d = inf.sample_depths(0.3, 20.0, 1)
    assert torch.isfinite(d).all() and d.item() == pytest.approx(20.0)
    assert torch.isfinite(inf.sample_depths(0.3, 20.0, 5)).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_normalization_stress_real_bank(optics):
    """Blocky random depth across all layers + random RGB: output stays finite and bounded."""
    bank = optics.build_psf_bank(grid=(5, 4), near_m=0.3, far_m=20.0, num_depths=8)
    W, H = optics.sensor_resolution
    g = torch.Generator().manual_seed(0)
    blocks = torch.rand(1, 1, H // 10, W // 10, generator=g)
    depth = torch.nn.functional.interpolate(0.3 + blocks * 25.0, size=(H, W), mode="nearest")
    depth[..., :20, :20] = float("inf")
    rgb = torch.rand(1, 3, H, W, generator=g)
    out = DenseDepthRenderer(bank).render(rgb.to(DEVICE), depth.to(DEVICE))
    assert torch.isfinite(out).all()
    assert out.min() > -1e-3 and out.max() < 1.02 * rgb.max()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_point_response_matches_direct_trace(optics):
    """Rendered point sources between nodes/layers vs PSFs traced directly by DeepLens."""
    from twinrobo.validation.optics import point_response_check

    bank = optics.build_psf_bank(grid=(9, 6), near_m=0.3, far_m=20.0, num_depths=16)
    res = point_response_check(
        DenseDepthRenderer(bank), optics, [(100, 160), (60, 50), (190, 310)], [10.0, 1.2, 0.33]
    )
    l1 = torch.tensor([r["l1"] for r in res])
    nearest = torch.tensor([r["l1_nearest"] for r in res])
    assert l1.max() < 0.4
    assert l1.mean() <= nearest.mean()  # interpolation beats nearest-node lookup
