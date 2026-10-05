"""PSF bank defaults and even-kernel centering, with a fake DeepLens lens (no DeepLens needed)."""

import math
from types import SimpleNamespace

import pytest
import torch

from twinrobo.camera import psf_bank_params
from twinrobo.optics.deeplens import DEFAULT_FAR_M, M_TO_MM, DeepLensOptics
from twinrobo.registry import CatalogRegistry


def fake_optics(focus=None):
    seen = []

    def psf(points, wvln, ks, spp=None):
        seen.append((points[:, 2], ks))
        out = torch.zeros(points.shape[0], ks, ks)
        out[:, ks // 2, ks // 2] = 1.0  # DeepLens centers odd kernels on index ks // 2
        return out

    o = object.__new__(DeepLensOptics)
    o._lens = SimpleNamespace(psf=psf, device=torch.device("cpu"), dtype=torch.float32,
                              wvln_rgb=[0.65, 0.55, 0.45])  # fmt: skip
    o.psf_ks, o.focus_distance_m = 65, focus
    return o, seen


def test_even_ks_is_centered_and_normalized():
    o, seen = fake_optics()
    p = o.generate_psf([[0.0, 0.0]], [2.0, math.inf], ks=64)
    assert p.shape == (2, 1, 3, 64, 64)
    assert seen[0][1] == 65  # traced odd, then cropped
    assert (p.flatten(-2).argmax(-1) == 32 * 64 + 32).all()  # center at ks // 2
    torch.testing.assert_close(p.sum((-1, -2)), torch.ones(2, 1, 3))


def test_infinite_depth_is_traced_at_deeplens_infinity():
    o, seen = fake_optics()
    o.generate_psf([[0.0, 0.0]], [math.inf])
    assert seen[0][0].item() == pytest.approx(-DEFAULT_FAR_M * M_TO_MM)
    with pytest.raises(ValueError):
        o.generate_psf([[0.0, 0.0]], [float("nan")])


def test_depth_layers_to_infinity_uniform_in_disparity():
    o, _ = fake_optics()
    d = o.sample_depths(0.1, math.inf, 16)
    assert d[0].isinf() and d[-1].item() == pytest.approx(0.1)
    disp = 1 / d
    torch.testing.assert_close(disp.diff(), torch.full((15,), 10.0 / 15), rtol=1e-4, atol=0)


def test_default_bank_params():
    spec = CatalogRegistry().load("stereolabs/zed-x/2.2mm")  # 1920x1200, sets no psf values
    p = psf_bank_params(spec)
    assert p["grid"] == (16, 10) and p["num_depths"] == 16
    assert p["near_m"] == 0.1 and p["far_m"] == math.inf
