"""Phase 1 benchmark: DenseDepthRenderer speed/memory and agreement with the DeepLens reference.

python tools/validate/benchmark_optics.py --out docs/benchmarks/phase1_optics.md
"""

import argparse
import platform
import time
from pathlib import Path

import torch

from twinrobo.optics import DenseDepthRenderer
from twinrobo.optics.deeplens import DeepLensOptics, deeplens_version
from twinrobo.utils.synthetic import fronto_plane, occluder_scene, slanted_plane
from twinrobo.validation.optics import compare_to_reference, point_response_check, psnr, timed

ROOT = Path(__file__).resolve().parents[2]
LENS = ROOT / "tests" / "data" / "cellphone80deg" / "lens.json"  # the reference lens (DeepLens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--grid", type=int, nargs=2, default=[17, 11], metavar=("GW", "GH"))
    ap.add_argument("--depths", type=int, default=16)
    ap.add_argument("--near", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=20.0)
    ap.add_argument("--focus", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None, help="write a Markdown report here")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires CUDA")

    W, H = args.width, args.height
    grid = tuple(args.grid)
    optics = DeepLensOptics(
        LENS, sensor_resolution=(W, H), focus_distance_m=args.focus, device="cuda"
    )

    t0 = time.perf_counter()
    bank = optics.build_psf_bank(
        grid=grid, near_m=args.near, far_m=args.far, num_depths=args.depths
    )
    torch.cuda.synchronize()
    t_bank = time.perf_counter() - t0
    renderer = DenseDepthRenderer(bank)

    scenes = {
        "slanted plane 0.4-15 m": slanted_plane(H, W, 0.4, 15.0, "cuda"),
        "fronto plane @ focus (2 m)": fronto_plane(H, W, args.focus, "cuda"),
        "fronto plane @ 0.5 m": fronto_plane(H, W, 0.5, "cuda"),
        "occluder 0.5 m over inf": occluder_scene(H, W, 0.5, args.far, "cuda"),
    }
    rows = []
    for name, (rgb, depth) in scenes.items():
        torch.cuda.reset_peak_memory_stats()
        _, times = timed(renderer.render, rgb, depth, repeats=5)
        peak = torch.cuda.max_memory_allocated() / 1e9
        t0 = time.perf_counter()
        cmp = compare_to_reference(renderer, optics, rgb, depth)
        torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0 - min(times)
        rows.append(
            {
                "scene": name,
                "ms": 1000 * sorted(times)[len(times) // 2],
                "peak_gb": peak,
                "ref_s": t_ref,
                "psnr": cmp["psnr_vs_reference"],
                "psnr_ideal": cmp["psnr_ideal_vs_reference"],
                "energy": cmp["mean_out"] / cmp["mean_in"],
                "energy_ref": cmp["mean_ref"] / cmp["mean_in"],
            }
        )
        print(rows[-1])

    # Monte Carlo floor: two independently sampled banks rendered by the same renderer.
    bank2 = optics.build_psf_bank(
        grid=grid, near_m=args.near, far_m=args.far, num_depths=args.depths
    )
    rgb, depth = scenes["slanted plane 0.4-15 m"]
    mc_floor = psnr(renderer.render(rgb, depth), DenseDepthRenderer(bank2).render(rgb, depth))

    # Physics check: rendered point sources vs PSFs traced directly by DeepLens.
    pixels = [(H // 2, W // 2), (H // 4 + 7, W // 6 + 11), (H - 50, W - 20), (10, 15)]
    points = point_response_check(renderer, optics, pixels, [10.0, 1.2, 0.33])

    gw, gh = grid
    lines = [
        "# Phase 1 optics benchmark",
        "",
        f"- GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; "
        f"DeepLens {deeplens_version()}; Python {platform.python_version()}",
        f"- Lens: `{LENS.name}`, focus {args.focus} m; image {W}x{H}; PSF grid {gw}x{gh}; "
        f"{bank.num_depths} depth layers in [{args.near}, {args.far}] m; ks {bank.ks}",
        f"- PSF bank build (DeepLens, one-time, cached): {t_bank:.1f} s",
        f"- Monte Carlo floor (two independently sampled banks, slanted plane): {mc_floor:.1f} dB",
        "",
        "| Scene | Runtime render (ms) | Peak GPU mem (GB) | DeepLens reference (s) "
        "| PSNR vs reference (dB) | Ideal vs reference (dB) | Energy out/in | Reference out/in |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scene']} | {r['ms']:.0f} | {r['peak_gb']:.2f} | {r['ref_s']:.1f} "
            f"| {r['psnr']:.1f} | {r['psnr_ideal']:.1f} "
            f"| {r['energy']:.4f} | {r['energy_ref']:.4f} |"
        )
    lines += [
        "",
        "## Point responses vs direct DeepLens traces",
        "",
        "A single bright pixel is rendered at an arbitrary field position and depth (between",
        "PSF nodes and depth layers). Its green response is compared with a PSF traced",
        "directly by DeepLens there. L1 = sum|a - b| (0 identical, 2 disjoint). The MC floor",
        "is two independent traces of the same point. Nearest is the closest bank PSF.",
        "",
        "| Pixel (row, col) | Field (x, y) | Depth (m) | L1 rendered | L1 MC floor "
        "| L1 nearest | Peak ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p['pixel']} | {p['field']} | {p['depth_m']} | {p['l1']:.3f} "
            f"| {p['l1_mc_floor']:.3f} | {p['l1_nearest']:.3f} | {p['peak_ratio']:.2f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "The reference is DeepLens `psf_map_rgb` + `conv_psf_map_depth_interp` at the same",
        "depth layers. It uses one PSF per block and blends by output-pixel depth, with no",
        "occlusion handling, so it is not ground truth at depth edges. Distortion and",
        "vignetting are not modeled by either path.",
        "",
        "Energy out/in < 1 for the runtime renderer is expected. DeepLens PSFs are",
        "recentered on the chief ray, but coma moves their energy centroid radially",
        "(about 1 px at the field edge), and the scatter formulation moves light with it, so a",
        "little light leaves the frame. The gather-style reference normalizes per output pixel.",
        "",
        "Render time scales with the number of occupied depth layers. A single plane",
        "occupies one layer. The slanted plane occupies all layers.",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)


if __name__ == "__main__":
    main()
