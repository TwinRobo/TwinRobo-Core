"""Turn the renderers' outputs into the playground's site assets.

    python tools/docs/playground/assemble.py   # -> docs/assets/playground/

Reads ``outputs/playground/<scene>`` (MuJoCo) and ``outputs/isaac/playground/<scene>``
(Isaac Sim), writes one WebP per camera and method, the pinhole view, a colour-mapped
depth image and ``manifest.json`` (scenes, cameras, what the page shows about them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [ROOT / "outputs/playground", ROOT / "outputs/isaac/playground"]
METHODS = {
    "psf": "PSF bank",
    "pupil": "Pupil views",
    "raycast": "Ray cast",
}
QUALITY = 82


def slug(camera_id: str) -> str:
    return camera_id.replace("/", "--")


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
                dst = out / scene["id"] / slug(cam["id"])
                dst.mkdir(parents=True, exist_ok=True)
                for name in ["pinhole", *METHODS]:
                    webp(sdir / cam["id"] / f"{name}.png", dst / f"{name}.webp")
                webp(depth_rgb(depth, near, far), dst / "depth.webp")
                cams.append({**cam, "dir": f"{scene['id']}/{slug(cam['id'])}"})
            scene["cameras"] = cams
            scenes.append(scene)
            print(f"{scene['id']}: {len(cams)} cameras, depth {near:.2f}–{far:.2f} m")
    scenes.sort(key=lambda s: s["simulator"] != "MuJoCo")
    manifest = {"methods": METHODS, "scenes": scenes}
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
