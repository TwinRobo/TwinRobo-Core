"""Dense-depth optics: render a synthetic RGB-D scene through the example camera.

Builds (first run, ~20 s) or loads the cached PSF bank for the example spec,
renders an occlusion scene and a slanted plane with the runtime renderer and
the DeepLens reference, and writes side-by-side PNGs to outputs/03_dense_depth/.

    python examples/03_dense_depth_optics.py
"""

from pathlib import Path

import torch
from torchvision.utils import save_image

from twinrobo import CameraTwin
from twinrobo.camera import build_reference_optics
from twinrobo.registry import BUILTIN_CATALOG
from twinrobo.utils.synthetic import occluder_scene, slanted_plane
from twinrobo.validation.optics import compare_to_reference, timed

ROOT = Path(__file__).resolve().parents[1]
SPEC = BUILTIN_CATALOG / "stereolabs" / "zed-x" / "2.2mm" / "camera.yaml"  # a catalog camera
OUT = ROOT / "outputs/03_dense_depth"


def to_display(x):
    """Linear -> approximate sRGB for viewing."""
    return x.clamp(0, 1) ** (1 / 2.2)


def main():
    camera = CameraTwin.from_spec(SPEC)
    renderer = camera.optics
    reference = build_reference_optics(camera.spec)
    print(
        renderer.describe(), "depths [m]:", [round(d, 2) for d in renderer.bank.depths_m.tolist()]
    )

    W, H = renderer.bank.sensor_resolution
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = {
        "occluder": occluder_scene(H, W, near_m=0.5, far_m=20.0, device="cuda"),
        "slanted": slanted_plane(H, W, near_m=0.4, far_m=15.0, device="cuda"),
    }
    for name, (rgb, depth) in scenes.items():
        frame, times = timed(camera.process, rgb, depth, repeats=3)
        cmp = compare_to_reference(renderer, reference, rgb, depth)
        print(
            f"{name:9s} runtime {1000 * min(times):6.1f} ms | PSNR vs DeepLens reference "
            f"{cmp['psnr_vs_reference']:.1f} dB (ideal: {cmp['psnr_ideal_vs_reference']:.1f} dB)"
        )
        # ideal | CameraTwin | DeepLens reference | 10x |CameraTwin - ideal|
        diff = (frame.rgb_optical - rgb).abs() * 10
        row = torch.cat([rgb, frame.rgb_optical, cmp["ref"], diff], dim=-1)
        save_image(to_display(row), OUT / f"{name}.png")
        crop = (slice(H // 4 - 60, H // 4 + 60), slice(W // 3 - 90, W // 3 + 90))
        save_image(
            to_display(
                torch.cat(
                    [rgb[..., crop[0], crop[1]], frame.rgb_optical[..., crop[0], crop[1]]], -1
                )
            ),
            OUT / f"{name}_edge_crop.png",
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
