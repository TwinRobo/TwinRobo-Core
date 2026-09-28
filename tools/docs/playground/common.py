"""Shared by the playground renderers: which cameras, at which size, and how outputs are saved.

Each renderer writes, per scene, into ``<staging>/<scene>/``:
``<camera>/<method>.png`` (the camera's image, sRGB), ``<camera>/pinhole.png`` (the
simulator's pinhole at the same field of view), ``<camera>/depth.npy`` (metric z-depth)
and ``meta.json``. `assemble.py` turns that into the site's assets and manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

WIDTH = 640  # playground images: every camera read out at this width (its own aspect)
METHODS = ("psf", "pupil", "raycast")
SKIP = {"examples/cellphone80deg"}  # the teaching example is not a product


def cameras() -> list[str]:
    from twinrobo import CatalogRegistry

    reg = CatalogRegistry()
    ids = sorted(e.id if hasattr(e, "id") else e for e in reg.list())
    return [i for i in ids if i not in SKIP]


def spec_for(camera_id: str):
    """The catalog camera read out at `WIDTH` px (same field of view, same lens)."""
    from twinrobo import CameraSpec, CatalogRegistry

    path = CatalogRegistry().resolve(camera_id)
    spec = CameraSpec.from_yaml(path)
    W, H = spec.sensor.resolution.width, spec.sensor.resolution.height
    h = int(round(WIDTH * H / W / 2) * 2)
    return CameraSpec.from_yaml(path, width=WIDTH, height=h), spec


def camera_meta(camera_id: str, full_spec, twin) -> dict:
    """What the page shows about a camera (from its full-resolution catalog spec)."""
    from twinrobo.optics.distortion import field_of_view

    fov = field_of_view(twin.intrinsics, twin.distortion)
    s, lens = full_spec.sensor, full_spec.lens
    return {
        "id": camera_id,
        "label": f"{full_spec.manufacturer} {full_spec.product}".strip(),
        "sensor": f"{s.resolution.width}×{s.resolution.height}"
        + (f", {s.pixel_pitch_um} µm" if s.pixel_pitch_um else ""),
        "lens": f"{lens.focal_length_mm} mm f/{lens.f_number}",
        "focus_m": lens.focus_distance_m,
        "fov_deg": {k: round(v, 1) for k, v in fov.items()},
        "status": full_spec.validation.status,
        "size": [twin.intrinsics.width, twin.intrinsics.height],
    }


def to_srgb8(rgb) -> np.ndarray:
    """Linear ``[1, 3, H, W]`` -> sRGB uint8 ``[H, W, 3]``."""
    from twinrobo.isp.color import linear_to_srgb

    x = linear_to_srgb(rgb[0].detach().clamp(0, 1)).permute(1, 2, 0).cpu().numpy()
    return (x * 255 + 0.5).astype(np.uint8)


def save_frame(out: Path, method: str, frame, pinhole: bool) -> None:
    import imageio.v3 as iio

    out.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out / f"{method}.png", to_srgb8(frame.rgb))
    if pinhole:
        iio.imwrite(out / "pinhole.png", to_srgb8(frame.rgb_ideal))
        np.save(out / "depth.npy", frame.depth[0, 0].detach().float().cpu().numpy())


def write_meta(scene_dir: Path, scene: dict, cams: list[dict]) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "meta.json").write_text(json.dumps({"scene": scene, "cameras": cams}, indent=2))


def finite(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else v
