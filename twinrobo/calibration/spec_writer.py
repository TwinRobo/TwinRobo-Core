"""Write calibration results into a catalog entry: a ``measured`` camera.yaml / module.yaml."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml


def _header(text: str) -> list[str]:
    """The leading comment block of a YAML file (its provenance notes)."""
    out = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            out.append(line)
        else:
            break
    while out and not out[-1].strip():
        out.pop()
    return out


def measured_camera_yaml(
    spec_path: str | Path,
    geometry=None,
    focus_m: float | None = None,
    vignetting=None,
    notes: str = "",
) -> str:
    """The camera.yaml text of ``spec_path`` updated with measured values.

    ``geometry`` (`GeometryResult`, rescaled to the spec's resolution if needed),
    ``focus_m`` (from `fit_focus`) and ``vignetting`` (`Vignetting`) replace the
    corresponding fields; ``validation.status`` becomes ``measured``. The original
    header is kept and a "Measured" section is appended to it. Inline comments
    below the header are not preserved.
    """
    path = Path(spec_path)
    text = path.read_text()
    raw = yaml.safe_load(text)
    res = raw["sensor"]["resolution"]
    lines = [f"# Measured {_dt.date.today().isoformat()} (twinrobo.calibration):"]
    cal = raw.setdefault("calibration", {}) or {}
    raw["calibration"] = cal
    if geometry is not None:
        if (geometry.width, geometry.height) != (res["width"], res["height"]):
            geometry = geometry.scaled(res["width"], res["height"])
        cal.update(geometry.to_spec())
        fov = geometry.fov_deg()
        lines.append(
            f"#   geometry: ChArUco, {geometry.views} views, {geometry.corners} corners, "
            f"RMS {geometry.rms_px:.3f} px; pinhole FoV {fov['h']:.1f} x {fov['v']:.1f} deg"
        )
    if focus_m is not None:
        raw.setdefault("lens", {})["focus_distance_m"] = round(float(focus_m), 4)
        lines.append(f"#   focus: {focus_m:.3f} m, fitted to slanted-edge MTF50 (lens model)")
    if vignetting is not None:
        cal["vignetting"] = vignetting.to_spec()
        lines.append(
            f"#   vignetting: flat field, corner {vignetting.corner:.3f} of center, "
            f"fit RMS {vignetting.rms_residual:.4f}"
        )
    if notes:
        lines += [f"#   {line}" for line in notes.splitlines()]
    v = raw.setdefault("validation", {}) or {}
    raw["validation"] = v
    v["status"] = "measured"
    v["version"] = int(v.get("version", 0) or 0) + 1
    head = _header(text)
    return "\n".join([*head, "#", *lines, ""]) + yaml.safe_dump(raw, sort_keys=False)


def measured_module_yaml(module_path: str | Path, stereo, notes: str = "") -> str:
    """The module.yaml text with the right eye's measured pose (`StereoResult`)."""
    path = Path(module_path)
    text = path.read_text()
    raw = yaml.safe_load(text)
    right = raw["eyes"]["right"]
    right["translation_m"] = [float(v) for v in stereo.translation_m]
    right["rotation_rpy_deg"] = [float(v) for v in stereo.rotation_rpy_deg]
    lines = [
        f"# Measured {_dt.date.today().isoformat()} (twinrobo.calibration): stereo, "
        f"{stereo.views} pairs, RMS {stereo.rms_px:.3f} px, "
        f"baseline {1000 * stereo.baseline_m:.2f} mm",
        *[f"#   {line}" for line in notes.splitlines() if notes],
    ]
    return "\n".join([*_header(text), "#", *lines, ""]) + yaml.safe_dump(raw, sort_keys=False)
