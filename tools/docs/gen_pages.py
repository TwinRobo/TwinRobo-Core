"""Generate site pages at build time (mkdocs-gen-files): one source for each page.

- ``contributing.md`` from ``CONTRIBUTING.md`` (its repo-relative links are rewritten
  by ``tools/docs/hooks.py``); the home page is the hand-made ``docs/index.md``;
- ``catalog.md``: every camera and stereo module in ``twinrobo/catalog``, read
  from their YAML files, so a contributed camera appears without editing docs.
"""

import math
from pathlib import Path

import mkdocs_gen_files
import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "twinrobo" / "catalog"
REPO = "https://github.com/TwinRobo/TwinRobo-Core"
STATUS = {
    "estimated": "estimated (datasheet)",
    "measured": "measured (real unit)",
    "verified": "verified",
    "manufacturer_verified": "manufacturer verified",
}

for src, dst in (("CONTRIBUTING.md", "contributing.md"),):
    with mkdocs_gen_files.open(dst, "w") as f:
        f.write((ROOT / src).read_text())
    mkdocs_gen_files.set_edit_path(dst, f"../{src}")


def pinhole_fov(spec: dict) -> tuple[float, float] | None:
    """Horizontal and vertical FoV (deg) of the camera's pinhole (before distortion)."""
    res = spec["sensor"]["resolution"]
    w, h = res["width"], res["height"]
    intr = (spec.get("calibration") or {}).get("intrinsic") or {}
    if intr.get("fx") and intr.get("fy"):
        fx, fy = intr["fx"], intr["fy"]
    else:
        pitch, f = spec["sensor"].get("pixel_pitch_um"), spec["lens"].get("focal_length_mm")
        if not pitch or not f:
            return None
        fx = fy = f / (pitch / 1000)
    return math.degrees(2 * math.atan(w / 2 / fx)), math.degrees(2 * math.atan(h / 2 / fy))


cameras, modules = [], []
for path in sorted(CATALOG.rglob("camera.yaml")):
    spec = yaml.safe_load(path.read_text())
    cameras.append((path, spec))
for path in sorted(CATALOG.rglob("module.yaml")):
    modules.append((path, yaml.safe_load(path.read_text())))

lines = [
    "# Camera catalog",
    "",
    "Every camera below works in every simulator and rendering method: load it with",
    '`CameraTwin.from_catalog("<id>")`. This page is generated from the catalog files',
    "in [`twinrobo/catalog`](" + REPO + "/tree/main/twinrobo/catalog), so it always",
    "lists what the installed package ships.",
    "",
    '!!! tip "Your camera is missing?"',
    "    [Add it](contributing.md#adding-a-camera): an entry is a `camera.yaml` and a lens",
    "    file. Camera makers can [partner with us](partners.md) for verified entries.",
    "",
    "## Cameras",
    "",
    "| ID | Camera | Sensor | Lens | Pinhole FoV (H × V) | Status |",
    "|---|---|---|---|---|---|",
]
for path, spec in cameras:
    rel = path.relative_to(ROOT).as_posix()
    s, lens = spec["sensor"], spec["lens"]
    res = s["resolution"]
    shutter = (s.get("shutter") or {}).get("type") or ""
    pitch = s.get("pixel_pitch_um")
    sensor = ", ".join(
        x for x in (f"{res['width']}×{res['height']}", f"{pitch} µm" if pitch else "", shutter) if x
    )
    fnum = lens.get("f_number")
    lens_txt = f"{lens.get('focal_length_mm', '?')} mm" + (f" f/{fnum}" if fnum else "")
    fov = pinhole_fov(spec)
    fov_txt = f"{fov[0]:.0f}° × {fov[1]:.0f}°" if fov else "from the lens file"
    status = STATUS.get((spec.get("validation") or {}).get("status"), "–")
    name = f"{spec.get('manufacturer', '')} {spec.get('product', '')}".strip()
    lines.append(
        f"| [`{spec['id']}`]({REPO}/blob/main/{rel}) | {name} | {sensor} | {lens_txt} "
        f"| {fov_txt} | {status} |"
    )
lines += [
    "",
    "Pinhole FoV is the undistorted field of view from the focal length; wide lenses",
    "see more at the edges (their distortion is modeled). *Estimated* entries follow",
    "the datasheet with a surrogate lens design; *measured* entries are fitted to",
    "captures of a real unit.",
    "",
    "## Stereo modules",
    "",
    "| ID | Product | Baseline | Eyes | Output |",
    "|---|---|---|---|---|",
]
for path, mod in modules:
    rel = path.relative_to(ROOT).as_posix()
    eyes = mod.get("eyes") or {}
    t = (eyes.get("right") or {}).get("translation_m") or [0, 0, 0]
    baseline = math.dist(t, (eyes.get("left") or {}).get("translation_m") or [0, 0, 0]) * 1000
    eye = (eyes.get("left") or {}).get("spec", "")
    eye_id = (path.parent / eye).resolve().relative_to(CATALOG).parent.as_posix() if eye else "–"
    name = f"{mod.get('manufacturer', '')} {mod.get('product', '')}".strip()
    lines.append(
        f"| [`{mod['id']}`]({REPO}/blob/main/{rel}) | {name} | {baseline:.0f} mm "
        f"| `{eye_id}` | {mod.get('output', 'rectified')} |"
    )
lines += [
    "",
    f"{len(cameras)} cameras and {len(modules)} stereo modules.",
    "",
]
with mkdocs_gen_files.open("catalog.md", "w") as f:
    f.write("\n".join(lines))
