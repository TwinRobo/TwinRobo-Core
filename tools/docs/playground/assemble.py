"""Turn the renderers' outputs into the playground's site assets.

    python tools/docs/playground/assemble.py   # -> docs/assets/playground/

Reads ``outputs/playground/<scene>`` (MuJoCo) and ``outputs/isaac/playground/<scene>``
(Isaac Sim), writes two WebPs per camera and method (the raw sensor image and the
calibrated, undistorted one), the pinhole view, a colour-mapped
depth image (only for cameras whose depth output is simulated) and ``manifest.json``
(scenes, cameras, what the page shows about them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from twinrobo import CatalogRegistry

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [ROOT / "outputs/playground", ROOT / "outputs/isaac/playground"]
METHODS = {
    "psf": "PSF bank",
    "pupil": "Pupil views",
    "raycast": "Ray cast",
}
OUTPUTS = {"raw": "Raw sensor", "cal": "Calibrated"}  # <method>.webp, <method>-cal.webp
QUALITY = 82


def slug(camera_id: str) -> str:
    return camera_id.replace("/", "--")


def outputs_depth(camera_id: str) -> bool:
    """Whether the real camera delivers depth (its spec's ``outputs.depth``)."""
    return CatalogRegistry().load(camera_id).outputs.depth


def to_mono(srgb: np.ndarray) -> np.ndarray:
    """A mono sensor's view: Rec. 709 luminance in linear light (as `CameraTwin.sensor_color`)."""
    x = srgb.astype(np.float64) / 255
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    out = np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)
    return np.repeat((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)[..., None], 3, axis=2)


def depth_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    from matplotlib import colormaps

    d = np.where(np.isfinite(depth) & (depth > 0), depth, far)
    t = np.clip((np.log(d) - np.log(near)) / (np.log(far) - np.log(near)), 0, 1)
    rgb = colormaps["turbo"](1 - t)[..., :3]  # near = warm
    return (rgb * 255 + 0.5).astype(np.uint8)


def webp(src: Path | np.ndarray, dst: Path) -> None:
    img = Image.open(src).convert("RGB") if isinstance(src, Path) else Image.fromarray(src)
    img.save(dst, "WEBP", quality=QUALITY, method=6)


def assemble(out: Path) -> dict:
    scenes = []
    for base in SOURCES:
        for meta_path in sorted(base.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text())
            sdir = meta_path.parent
            scene = dict(meta["scene"])
            depths = [np.load(sdir / c["id"] / "depth.npy") for c in meta["cameras"]]
            valid = np.concatenate([d[np.isfinite(d) & (d > 0)].ravel() for d in depths])
            # from the 10th percentile: past the gripper fingers right at the lens
            near, far = (float(v) for v in np.percentile(valid, [10, 99]))
            scene["depth_m"] = [round(near, 3), round(far, 3)]
            cams = []
            for cam, depth in zip(meta["cameras"], depths, strict=True):
                mono = CatalogRegistry().load(cam["id"]).sensor.color == "mono"
                dst = out / scene["id"] / slug(cam["id"])
                dst.mkdir(parents=True, exist_ok=True)
                for name in ["pinhole", *METHODS, *(f"{m}-cal" for m in METHODS)]:
                    img = np.asarray(Image.open(sdir / cam["id"] / f"{name}.png").convert("RGB"))
                    webp(to_mono(img) if mono else img, dst / f"{name}.webp")
                cam = {"outputs_depth": outputs_depth(cam["id"]), **cam}
                if cam["outputs_depth"]:  # only depth cameras have a Depth view
                    webp(depth_rgb(depth, near, far), dst / "depth.webp")
                cams.append({**cam, "dir": f"{scene['id']}/{slug(cam['id'])}"})
            scene["cameras"] = cams
            scenes.append(scene)
            print(f"{scene['id']}: {len(cams)} cameras, depth {near:.2f}–{far:.2f} m")
    scenes.sort(key=lambda s: s["simulator"] != "MuJoCo")
    manifest = {"methods": METHODS, "outputs": OUTPUTS, "scenes": scenes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/assets/playground"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    assemble(out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"-> {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
