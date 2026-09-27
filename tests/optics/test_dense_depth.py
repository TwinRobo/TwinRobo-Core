"""DenseDepthRenderer tests with synthetic PSF banks (no DeepLens needed)."""

import pytest
import torch

from twinrobo.optics.dense_depth import DenseDepthRenderer, _axis_weights
from twinrobo.optics.psf import PSFBank

KS = 16
C0 = KS // 2  # PSF center index


def delta(ks=KS):
    p = torch.zeros(ks, ks)
    p[C0, C0] = 1
    return p


def gauss(sigma, ks=KS):
    y, x = torch.meshgrid(torch.arange(ks) - C0, torch.arange(ks) - C0, indexing="ij")
    p = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return p / p.sum()


def box(r, ks=KS):
    p = torch.zeros(ks, ks)
    p[C0 - r : C0 + r + 1, C0 - r : C0 + r + 1] = 1
    return p / p.sum()


def make_bank(per_depth, depths_m, W, H, grid=(1, 1)):
    """per_depth: list of [ks, ks] PSFs (same for all nodes/channels) or [Gh, Gw, ks, ks]."""
    gw, gh = grid
    psfs = []
    for p in per_depth:
        if p.ndim == 2:
            p = p.expand(gh, gw, KS, KS)
        psfs.append(p[:, :, None].expand(gh, gw, 3, KS, KS))
    return PSFBank(
        psfs=torch.stack(psfs).contiguous(),
        depths_m=torch.tensor(depths_m, dtype=torch.float32),
        sensor_resolution=(W, H),
    )


def test_axis_weights_partition_of_unity():
    for n in (1, 2, 3, 7):
        w = _axis_weights(101, n, 7, 8, "cpu")
        torch.testing.assert_close(w.sum(0), torch.ones(w.shape[1]))
        assert (w >= 0).all()


@pytest.mark.parametrize("grid", [(1, 1), (3, 2), (5, 4)])
def test_delta_bank_is_identity(grid):
    H, W = 37, 53
    bank = make_bank([delta(), delta()], [10.0, 1.0], W, H, grid)
    rgb = torch.rand(2, 3, H, W)
    depth = torch.rand(2, 1, H, W) * 9 + 1
    out = DenseDepthRenderer(bank).render(rgb, depth)
    # A layer share below eps (1e-4) may be dropped; that bounds the error.
    torch.testing.assert_close(out, rgb, atol=1e-4, rtol=0)


@pytest.mark.parametrize("grid", [(1, 1), (4, 3)])
def test_uniform_image_stays_uniform(grid):
    H, W = 40, 60
    bank = make_bank([gauss(3.0), gauss(1.0)], [10.0, 1.0], W, H, grid)
    rgb = torch.full((1, 3, H, W), 0.7)
    depth = torch.full((1, 1, H, W), 3.0)
    out = DenseDepthRenderer(bank).render(rgb, depth)
    torch.testing.assert_close(out, rgb, atol=1e-5, rtol=0)


def test_matches_deeplens_conv_for_uniform_bank():
    imgsim = pytest.importorskip("deeplens.imgsim")
    H, W = 48, 64
    # Asymmetric PSF to catch flips / off-by-one centering.
    p = torch.zeros(KS, KS)
    p[C0, C0] = 0.5
    p[C0 - 3, C0 + 1] = 0.3
    p[C0 + 2, C0 - 4] = 0.2
    bank = make_bank([p], [5.0], W, H)
    rgb = torch.rand(1, 3, H, W)
    out = DenseDepthRenderer(bank).render(rgb, torch.full((1, 1, H, W), 5.0))
    ref = imgsim.conv_psf(rgb, p.expand(3, KS, KS).contiguous())
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)


def test_point_source_reproduces_node_psf_orientation():
    """A point at a node center yields that node's PSF, unflipped, in image orientation."""
    H, W = 60, 60
    gw = gh = 3
    per_node = torch.stack([torch.stack([delta() for _ in range(gw)]) for _ in range(gh)])
    p = torch.zeros(KS, KS)
    p[C0, C0] = 0.6
    p[C0 - 5, C0 + 2] = 0.4  # energy up (-row) and right (+col) of center
    per_node[1, 1] = p  # center node
    bank = make_bank([per_node], [5.0], W, H, (gw, gh))
    rgb = torch.zeros(1, 3, H, W)
    r, c = 30, 30  # center node sits at pixel-index 29.5 on both axes
    rgb[..., r, c] = 1
    out = DenseDepthRenderer(bank).render(rgb, torch.full((1, 1, H, W), 5.0))[0, 0]
    # Next to the node its weight is ~0.97 per axis: the response is that node's PSF.
    assert out[r, c] > 0.55
    assert out[r - 5, c + 2] > 0.3
    assert out[r + 5, c - 2] < 1e-4  # a flipped PSF would put energy here
    torch.testing.assert_close(out.sum(), torch.tensor(1.0), atol=1e-4, rtol=0)


def test_sharp_near_occluder_has_no_background_bleed():
    H, W = 64, 64
    bank = make_bank([box(4), delta()], [10.0, 1.0], W, H)  # far blurred, near sharp
    rgb = torch.zeros(1, 3, H, W)
    depth = torch.full((1, 1, H, W), 10.0)
    rgb[..., 20:44, 20:44] = 1.0
    depth[..., 20:44, 20:44] = 1.0
    rgb[..., :, :8] = 0.5  # bright far content near but outside the square
    out = DenseDepthRenderer(bank).render(rgb, depth)
    # Near square stays exactly as is (sharp, fully opaque over the background).
    torch.testing.assert_close(out[..., 20:44, 20:44], rgb[..., 20:44, 20:44], atol=1e-5, rtol=0)
    # Far background right next to the square is not darkened by a halo.
    assert out[..., 30, 12].min() > -1e-5
    assert torch.isfinite(out).all()


def test_blurred_near_occluder_spreads_over_sharp_background():
    H, W = 64, 64
    bank = make_bank([delta(), box(4)], [10.0, 1.0], W, H)  # far sharp, near blurred
    rgb = torch.zeros(1, 3, H, W)
    depth = torch.full((1, 1, H, W), 10.0)
    rgb[..., 24:40, 24:40] = 1.0
    depth[..., 24:40, 24:40] = 1.0
    out = DenseDepthRenderer(bank).render(rgb, depth)
    # Light from the near square spreads outside its footprint over the background.
    assert out[0, 0, 32, 42] > 0.05
    # Background far away is untouched.
    assert out[0, 0, 5, 5].abs() < 1e-5
    # Energy of the near square is conserved (black background).
    torch.testing.assert_close(out.sum(), rgb.sum(), rtol=1e-4, atol=1e-3)


def test_flat_depth_between_layers_conserves_energy():
    H, W = 48, 48
    bank = make_bank([gauss(4.0), gauss(0.8)], [10.0, 1.0], W, H)
    rgb = torch.rand(1, 3, H, W)
    depth = torch.full((1, 1, H, W), 2.0)  # between the two layers in disparity
    out = DenseDepthRenderer(bank).render(rgb, depth)
    torch.testing.assert_close(out.mean(), rgb.mean(), atol=2e-3, rtol=0)


def test_invalid_depth_goes_to_far_layer():
    H, W = 32, 32
    bank = make_bank([box(3), delta()], [10.0, 1.0], W, H)
    rgb = torch.rand(1, 3, H, W)
    r = DenseDepthRenderer(bank)
    bad = torch.full((1, 1, H, W), float("inf"))
    bad[..., :4, :] = float("nan")
    bad[..., 4:8, :] = -1.0
    far = torch.full((1, 1, H, W), 10.0)
    torch.testing.assert_close(r.render(rgb, bad), r.render(rgb, far))


def test_rejects_resolution_mismatch():
    bank = make_bank([delta()], [5.0], 32, 32)
    with pytest.raises(ValueError, match="resolution"):
        DenseDepthRenderer(bank).render(torch.rand(1, 3, 16, 16), torch.ones(1, 1, 16, 16))


def test_bank_roundtrip(tmp_path):
    bank = make_bank([gauss(2.0), delta()], [10.0, 1.0], 32, 24, (2, 2))
    bank.metadata["lens"] = "x"
    bank.save(tmp_path / "b.pt")
    b2 = PSFBank.load(tmp_path / "b.pt")
    torch.testing.assert_close(b2.psfs, bank.psfs)
    assert b2.sensor_resolution == (32, 24) and b2.metadata == {"lens": "x"}


def test_bank_rejects_unsorted_depths():
    with pytest.raises(ValueError, match="far -> near"):
        make_bank([delta(), delta()], [1.0, 10.0], 8, 8)


def test_gradients_flow_to_rgb():
    H, W = 24, 32
    bank = make_bank([gauss(2.0), gauss(0.8)], [10.0, 1.0], W, H, (3, 2))
    rgb = torch.rand(1, 3, H, W, requires_grad=True)
    depth = torch.rand(1, 1, H, W) * 9 + 1
    DenseDepthRenderer(bank).render_differentiable(rgb, depth).square().sum().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0


def test_cache_tracks_users_and_releases_unused_banks(tmp_path):
    from twinrobo.optics.cache import PSFCache

    cache = PSFCache(tmp_path)
    bank = make_bank([delta()], [5.0], 8, 8)
    key_a, key_shared = {"k": "a"}, {"k": "shared"}
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    cache.put(key_a, bank, source=a)
    cache.put(key_shared, bank, source=a)
    cache.get_or_build(key_shared, lambda: bank, source=b)  # second preset reuses the bank
    legacy = cache.path_for({"k": "legacy"})
    bank.save(legacy)
    legacy.with_suffix(".json").write_text('{"k": "legacy"}')  # pre-used_by sidecar

    removed = cache.release(a)
    assert removed == [cache.path_for(key_a)]
    assert (
        not cache.path_for(key_a).exists()
        and not cache.path_for(key_a).with_suffix(".json").exists()
    )
    assert cache.path_for(key_shared).exists()  # still used by b
    assert cache.release(b) == [cache.path_for(key_shared)]
    assert legacy.exists()  # unknown users: never removed automatically
    assert cache.release(a) == []
