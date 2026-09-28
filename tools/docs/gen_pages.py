"""Generate site pages at build time (mkdocs-gen-files): one source for each page.

- ``contributing.md`` from ``CONTRIBUTING.md`` and ``cla.md`` from ``CLA.md`` (their
  repo-relative links are rewritten by ``tools/docs/hooks.py``); ``license.md``: the
  licenses at a glance and ``COMMERCIAL.md``; the home page is ``docs/index.md``;
- ``catalog.md`` and ``assets/catalog.json``: every camera and stereo module in
  ``twinrobo/catalog``, read from their YAML files (the page's camera browser loads
  the JSON), so a contributed camera appears without editing docs.
"""

import json
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

for src, dst in (("CONTRIBUTING.md", "contributing.md"), ("CLA.md", "cla.md")):
    with mkdocs_gen_files.open(dst, "w") as f:
        f.write((ROOT / src).read_text())
    mkdocs_gen_files.set_edit_path(dst, f"../{src}")

# license.md: the licensing at a glance, then COMMERCIAL.md (one source for the terms)
commercial = (
    (ROOT / "COMMERCIAL.md").read_text().replace("# Commercial license", "## Commercial license", 1)
)
with mkdocs_gen_files.open("license.md", "w") as f:
    f.write((Path(__file__).parent / "license_head.md").read_text().format(repo=REPO) + commercial)
mkdocs_gen_files.set_edit_path("license.md", "../COMMERCIAL.md")


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
    cameras.append((path, yaml.safe_load(path.read_text())))
for path in sorted(CATALOG.rglob("module.yaml")):
    modules.append((path, yaml.safe_load(path.read_text())))

# Pre-rendered views of the cameras (tools/docs/playground): which catalog ids have them.
VIEWS = ROOT / "docs" / "assets" / "playground" / "manifest.json"
viewable = set()
if VIEWS.exists():
    for sc in json.loads(VIEWS.read_text())["scenes"]:
        viewable |= {c["id"] for c in sc["cameras"]}


def module_entry(path: Path, mod: dict) -> dict:
    eyes = mod.get("eyes") or {}
    t = (eyes.get("right") or {}).get("translation_m") or [0, 0, 0]
    baseline = math.dist(t, (eyes.get("left") or {}).get("translation_m") or [0, 0, 0]) * 1000
    eye_ids = []
    for side in ("left", "right"):
        eye = (eyes.get(side) or {}).get("spec")
        if eye:
            eye_ids.append((path.parent / eye).resolve().relative_to(CATALOG).parent.as_posix())
    return {
        "id": mod["id"],
        "name": f"{mod.get('manufacturer', '')} {mod.get('product', '')}".strip(),
        "baseline_mm": round(baseline, 1),
        "eyes": sorted(set(eye_ids)),
        "output": mod.get("output", "rectified"),
        "yaml": f"{REPO}/blob/main/{path.relative_to(ROOT).as_posix()}",
    }


mods = [module_entry(p, m) for p, m in modules]


def camera_entry(path: Path, spec: dict) -> dict:
    s, lens = spec["sensor"], spec["lens"]
    res = s["resolution"]
    fov = pinhole_fov(spec)
    calib = spec.get("calibration") or {}
    return {
        "id": spec["id"],
        "maker": spec.get("manufacturer") or "",
        "product": spec.get("product") or "",
        "sensor": s.get("model") or "",
        "width": res["width"],
        "height": res["height"],
        "pitch_um": s.get("pixel_pitch_um"),
        "shutter": (s.get("shutter") or {}).get("type") or "",
        "mono": s.get("color") == "mono",
        "focal_mm": lens.get("focal_length_mm"),
        "f_number": lens.get("f_number"),
        "focus_m": lens.get("focus_distance_m"),
        "fov": [round(fov[0], 1), round(fov[1], 1)] if fov else None,
        "distortion": bool(calib.get("distortion")),
        "status": (spec.get("validation") or {}).get("status") or "estimated",
        "yaml": f"{REPO}/blob/main/{path.relative_to(ROOT).as_posix()}",
        "modules": [m["id"] for m in mods if spec["id"] in m["eyes"]],
        "views": spec["id"] in viewable,
    }


# Near-infrared cameras (e.g. RealSense IR imagers) are left out: TwinRobo renders visible
# light, so their images would not show what they see (ROADMAP.md).
shown = [(p, c) for p, c in cameras if (c["sensor"].get("spectrum") or "visible") != "nir"]
cams = [camera_entry(p, c) for p, c in shown]
listed = {c["id"] for c in cams}
with mkdocs_gen_files.open("assets/catalog.json", "w") as f:
    json.dump({"cameras": cams, "modules": mods, "status": STATUS}, f, indent=1)

BROWSER = (Path(__file__).parent / "catalog_browser.html").read_text()

lines = [
    "---",
    "hide:",
    "  - navigation",
    "  - toc",
    "nav_icon: material/camera-iris",
    "---",
    "",
    # a title for screen readers and search only (Material adds a visible one otherwise)
    '<h1 class="tr-sr-only">Camera catalog</h1>',
    "",
    BROWSER,
    "",
    "## Cameras",
    "",
    "| Camera | ID | Sensor | Lens | FoV (H × V) | Status | |",
    "|---|---|---|---|---|---|---|",
]
SHOW = '<a class="trc-show" href="#cam={id}" data-camera="{id}">Show in browser</a>'
SHOW_OFF = '<span class="trc-show is-off" aria-disabled="true" title="{tip}">Show in browser</span>'
TIP_NIR = (
    "Not viewable in the browser yet: a near-infrared camera, and TwinRobo renders visible "
    "light (see the roadmap)"
)
TIP_NONE = "No in-browser preview yet: this camera's views have not been rendered for the docs"
# the table lists every camera (the browser leaves near-infrared ones out)
table = [
    {**camera_entry(p, c), "nir": (c["sensor"].get("spectrum") or "visible") == "nir"}
    for p, c in cameras
]
for c in sorted(table, key=lambda c: (c["maker"], c["product"])):
    sensor = ", ".join(
        x
        for x in (
            f"{c['width']}×{c['height']}",
            f"{c['pitch_um']} µm" if c["pitch_um"] else "",
            c["shutter"],
            "mono" if c["mono"] else "",
        )
        if x
    )
    lens = f"{c['focal_mm']} mm" + (f" f/{c['f_number']}" if c["f_number"] else "")
    fov = f"{c['fov'][0]:.0f}° × {c['fov'][1]:.0f}°" if c["fov"] else "from the lens file"
    if c["views"] and not c["nir"]:  # the page script loads it into the viewer above
        show = SHOW.format(id=c["id"])
    else:
        show = SHOW_OFF.format(tip=TIP_NIR if c["nir"] else TIP_NONE)
    lines.append(
        f"| {c['maker']} {c['product']} | [`{c['id']}`]({c['yaml']}) | {sensor} | {lens} | {fov} "
        f"| {STATUS.get(c['status'], c['status'])} | {show} |"
    )
lines += [
    "",
    "## Stereo modules",
    "",
    "| Module | ID | Baseline | Eyes | Output | |",
    "|---|---|---|---|---|---|",
]
TIP_STEREO = (
    "Not viewable in the browser yet: stereo modules (both eyes together) are not rendered "
    "for the docs; its eyes are listed above"
)
for m in sorted(mods, key=lambda m: m["name"]):
    # an eye shown in the viewer links to it (#cam=... is the page script's state)
    link = '<a href="#cam={0}" data-camera="{0}"><code>{0}</code></a>'
    eyes = ", ".join(link.format(e) if e in listed else f"`{e}` (IR)" for e in m["eyes"])
    lines.append(
        f"| {m['name']} | [`{m['id']}`]({m['yaml']}) | {m['baseline_mm']:.0f} mm "
        f"| {eyes or '–'} | {m['output']} | {SHOW_OFF.format(tip=TIP_STEREO)} |"
    )
lines += [
    "",
    '??? info "About the images"',
    "    Each camera is seen from the same pose in two scenes, through its own lens, field",
    "    of view and focus: MuJoCo (robosuite PickPlace, from the Panda's wrist) and Isaac",
    "    Sim 6.1 (a random tabletop). *Pinhole* is what the simulator renders by itself;",
    "    *PSF bank* blurs it with the lens' point-spread functions and applies distortion",
    "    and vignetting (fastest); *pupil views* combine renders from points on the lens'",
    "    pupil along each pixel's traced rays; *ray cast* traces every pixel's rays",
    "    through the lens into the scene (most exact). There is no depth view: TwinRobo",
    "    simulates what cameras image, not yet what depth cameras measure",
    "    ([depth output](cameras.md#depth-output)). Images are 640 px wide, rendered",
    "    offline with [`tools/docs/playground`](" + REPO + "/tree/main/tools/docs/playground).",
    "    For cameras on a robot, see the [simulator demo](playground.md). Near-infrared",
    "    cameras (the RealSense IR imagers) are not shown: TwinRobo renders visible light",
    "    ([roadmap](" + REPO + "/blob/main/ROADMAP.md#infrared-cameras)).",
    "",
    '!!! tip "Your camera is missing?"',
    "    [Add it](contributing.md#adding-a-camera): an entry is a `camera.yaml` and a lens",
    "    file, and this page picks it up. Camera makers can [partner with us](partners.md)",
    "    for verified entries.",
    "",
]
with mkdocs_gen_files.open("catalog.md", "w") as f:
    f.write("\n".join(lines))
