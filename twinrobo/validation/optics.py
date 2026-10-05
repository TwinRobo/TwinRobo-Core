"""Optical validation: compare the runtime renderer against the DeepLens reference.

Phase 1 exit check: ``renderer.render(rgb, depth)`` must agree
with the physical reference, and be fast. Real-camera PSF/MTF comparison comes
with measured data (Phase 3/4).
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor


def psnr(a: Tensor, b: Tensor, peak: float = 1.0) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(peak**2 / mse)).item()


def timed(fn, *args, repeats: int = 5, warmup: int = 1, **kwargs) -> tuple[Any, list[float]]:
    """Run ``fn`` and return ``(result, wall-clock seconds per repeat)``, CUDA-synchronized."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    result = None
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        sync()
        times.append(time.perf_counter() - t0)
    return result, times


def compare_to_reference(renderer, reference, rgb: Tensor, depth: Tensor, psf_grid=None) -> dict:
    """Render with the runtime renderer and the DeepLens reference and compare.

    ``reference`` is a `DeepLensOptics`. The reference uses the renderer bank's
    depth layers and far plane, so only the rendering method differs.
    """
    bank = renderer.bank
    out = renderer.render(rgb, depth)
    ref = reference.render_reference(
        rgb,
        depth,
        psf_grid=psf_grid or bank.grid,
        depths_m=bank.depths_m,
        far_m=float(bank.depths_m[0]),  # inf is mapped to DEFAULT_FAR_M
    )
    return {
        "psnr_vs_reference": psnr(out, ref),
        "psnr_ideal_vs_reference": psnr(rgb, ref),
        "max_abs_err": (out - ref).abs().max().item(),
        "mean_in": rgb.mean().item(),
        "mean_out": out.mean().item(),
        "mean_ref": ref.mean().item(),
        "out": out,
        "ref": ref,
    }


def point_response_check(renderer, reference, pixels, depths_m, channel: int = 1) -> list[dict]:
    """Compare rendered point-source responses with PSFs traced directly by DeepLens.

    For each pixel ``(row, col)`` and depth, a single bright pixel on black is
    rendered through ``renderer``, which interpolates the bank in field and depth.
    The ``ks x ks`` crop around it is compared with ``reference.generate_psf``
    at that exact field position and depth. This measures PSF-bank interpolation
    error against physics.

    Returns per case: ``l1`` = sum |rendered - traced| (0 = identical, 2 = disjoint),
    ``l1_mc_floor`` = the same between two independent traces, ``l1_nearest`` =
    the nearest bank PSF (node and layer) vs traced, and ``peak_ratio`` =
    rendered peak / traced peak.
    """
    bank = renderer.bank
    W, H = bank.sensor_resolution
    ks, c = bank.ks, bank.ks // 2
    gw, gh = bank.grid
    dev = bank.device
    results = []
    for r, col in pixels:
        fx = 2 * (col + 0.5) / W - 1
        fy = 1 - 2 * (r + 0.5) / H
        for d in depths_m:
            rgb = torch.zeros(1, 3, H, W, device=dev)
            rgb[..., r, col] = 1.0
            out = renderer.render(rgb, torch.full((1, 1, H, W), float(d), device=dev))
            pad = torch.nn.functional.pad(out, (c, c, c, c))
            crop = pad[0, channel, r : r + ks, col : col + ks]
            traced = reference.generate_psf([[fx, fy]], [d], ks=ks)[0, 0, channel]
            traced2 = reference.generate_psf([[fx, fy]], [d], ks=ks)[0, 0, channel]
            i = round((r + 0.5) * (gh - 1) / H) if gh > 1 else 0
            j = round((col + 0.5) * (gw - 1) / W) if gw > 1 else 0
            k = int(torch.argmin((1 / bank.depths_m - 1 / d).abs()))
            nearest = bank.psfs[k, i, j, channel]
            results.append(
                {
                    "pixel": (r, col),
                    "field": (round(fx, 3), round(fy, 3)),
                    "depth_m": d,
                    "l1": (crop - traced).abs().sum().item(),
                    "l1_mc_floor": (traced2 - traced).abs().sum().item(),
                    "l1_nearest": (nearest - traced).abs().sum().item(),
                    "peak_ratio": (crop.max() / traced.max()).item(),
                }
            )
    return results
