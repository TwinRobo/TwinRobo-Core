"""CameraSpec: the structured description of one deployed camera configuration.

The format is pre-1.0; backward compatibility is not guaranteed yet. A spec
describes a complete camera *module* — sensor, lens, operating state,
calibration, sensor/ISP models and validation metadata — not a sensor or lens
in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import SpecError

SUPPORTED_SCHEMA_VERSIONS = ("0.1",)
#: ``lens.deeplens_model.path: catalog:<id>/lens.json`` names a lens of the built-in catalog,
#: so a spec anywhere can reuse a catalog lens.
CATALOG_LENS_PREFIX = "catalog:"
BUILTIN_CATALOG_DIR = Path(__file__).resolve().parent / "catalog"

PROVENANCE_LEVELS = ("estimated", "measured", "verified", "manufacturer_verified")


@dataclass
class Resolution:
    width: int
    height: int


@dataclass
class SensorSpec:
    manufacturer: str | None
    model: str | None
    resolution: Resolution
    pixel_pitch_um: float | None = None
    shutter: str = "global"  # "global" | "rolling"
    bit_depth: int | None = None


@dataclass
class DeepLensModelRef:
    path: Path  # resolved relative to the spec file


@dataclass
class LensSpec:
    manufacturer: str | None
    model: str | None
    focal_length_mm: float | None = None
    f_number: float | None = None
    focus_distance_m: float | None = None
    deeplens_model: DeepLensModelRef | None = None


@dataclass
class ValidationSpec:
    status: str = "estimated"  # one of PROVENANCE_LEVELS
    version: int = 0


@dataclass
class CameraSpec:
    schema_version: str
    id: str
    manufacturer: str | None
    product: str | None
    sensor: SensorSpec
    lens: LensSpec
    calibration: dict[str, Any] = field(default_factory=dict)
    sensor_model: dict[str, Any] = field(default_factory=dict)
    isp: dict[str, Any] = field(default_factory=dict)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    source_path: Path | None = None

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> CameraSpec:
        """Load a spec file; keyword ``overrides`` go to `apply_overrides` (``width=``, ...)."""
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text())
        except FileNotFoundError as e:
            raise SpecError(f"Spec file not found: {path}") from e
        except yaml.YAMLError as e:
            raise SpecError(f"Invalid YAML in {path}: {e}") from e
        if overrides:
            data = apply_overrides(data, **overrides)
        return cls.from_dict(data, base_dir=path.parent, source_path=path)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        base_dir: str | Path | None = None,
        source_path: Path | None = None,
    ) -> CameraSpec:
        if not isinstance(data, dict):
            raise SpecError("CameraSpec must be a mapping")
        base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

        version = str(_require(data, "schema_version"))
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SpecError(
                f"Unsupported schema_version {version!r}; "
                f"supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        sensor_d = _require(data, "sensor")
        res_d = _require(sensor_d, "resolution", ctx="sensor")
        sensor = SensorSpec(
            manufacturer=sensor_d.get("manufacturer"),
            model=sensor_d.get("model"),
            resolution=Resolution(
                width=int(_require(res_d, "width", ctx="sensor.resolution")),
                height=int(_require(res_d, "height", ctx="sensor.resolution")),
            ),
            pixel_pitch_um=sensor_d.get("pixel_pitch_um"),
            shutter=(sensor_d.get("shutter") or {}).get("type", "global"),
            bit_depth=sensor_d.get("bit_depth"),
        )
        if sensor.shutter not in ("global", "rolling"):
            raise SpecError(f"sensor.shutter.type must be global|rolling, got {sensor.shutter!r}")

        lens_d = _require(data, "lens")
        dl_d = lens_d.get("deeplens_model")
        deeplens_model = None
        if dl_d is not None:
            ref = str(_require(dl_d, "path", ctx="lens.deeplens_model"))
            if ref.startswith(CATALOG_LENS_PREFIX):  # a lens of the built-in catalog
                model_path = (BUILTIN_CATALOG_DIR / ref[len(CATALOG_LENS_PREFIX) :]).resolve()
            else:
                model_path = Path(ref)
                if not model_path.is_absolute():
                    model_path = (base_dir / model_path).resolve()
            deeplens_model = DeepLensModelRef(path=model_path)
        lens = LensSpec(
            manufacturer=lens_d.get("manufacturer"),
            model=lens_d.get("model"),
            focal_length_mm=lens_d.get("focal_length_mm"),
            f_number=lens_d.get("f_number"),
            focus_distance_m=lens_d.get("focus_distance_m"),
            deeplens_model=deeplens_model,
        )

        val_d = data.get("validation") or {}
        validation = ValidationSpec(
            status=val_d.get("status", "estimated"),
            version=int(val_d.get("version", 0)),
        )
        if validation.status not in PROVENANCE_LEVELS:
            raise SpecError(
                f"validation.status must be one of {PROVENANCE_LEVELS}, got {validation.status!r}"
            )

        return cls(
            schema_version=version,
            id=str(_require(data, "id")),
            manufacturer=data.get("manufacturer"),
            product=data.get("product"),
            sensor=sensor,
            lens=lens,
            calibration=data.get("calibration") or {},
            sensor_model=data.get("sensor_model") or {},
            isp=data.get("isp") or {},
            validation=validation,
            source_path=source_path,
        )


def _require(d: dict[str, Any], key: str, ctx: str | None = None) -> Any:
    if not isinstance(d, dict) or key not in d or d[key] is None:
        where = f"{ctx}.{key}" if ctx else key
        raise SpecError(f"CameraSpec is missing required field '{where}'")
    return d[key]


def apply_overrides(
    data: dict[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
    focus_distance_m: float | None = None,
    near_m: float | None = None,
    far_m: float | None = None,
    depth_samples: int | None = None,
    field_samples: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """A copy of raw spec ``data`` with common changes applied: the same camera, re-configured.

    - ``width`` / ``height``: the same sensor read out at another resolution. Explicit
      intrinsics and the pixel pitch scale with it (pixel-edge convention), so the
      field of view is unchanged.
    - ``focus_distance_m``: refocus the lens.
    - ``near_m``, ``far_m``, ``depth_samples``, ``field_samples`` ``(w, h)``: the PSF
      bank's depth range and sampling (``calibration.psf``).
    """
    import copy

    raw = copy.deepcopy(data)
    if width is not None or height is not None:
        res = raw["sensor"]["resolution"]
        w0, h0 = res["width"], res["height"]
        res["width"], res["height"] = int(width or w0), int(height or h0)
        intr = (raw.get("calibration") or {}).get("intrinsic") or {}
        if all(intr.get(k) is not None for k in ("fx", "fy", "cx", "cy")):
            sx, sy = res["width"] / w0, res["height"] / h0
            intr["fx"], intr["fy"] = intr["fx"] * sx, intr["fy"] * sy
            intr["cx"], intr["cy"] = (intr["cx"] + 0.5) * sx - 0.5, (intr["cy"] + 0.5) * sy - 0.5
            if raw["sensor"].get("pixel_pitch_um"):
                raw["sensor"]["pixel_pitch_um"] = raw["sensor"]["pixel_pitch_um"] / sx
    if focus_distance_m is not None:
        raw["lens"]["focus_distance_m"] = float(focus_distance_m)
    if any(v is not None for v in (near_m, far_m, depth_samples, field_samples)):
        psf = raw.setdefault("calibration", {}).setdefault("psf", {})
        if near_m is not None:
            psf["near_m"] = float(near_m)
        if far_m is not None:
            psf["far_m"] = float(far_m)
        if depth_samples is not None:
            psf["depth_samples"] = int(depth_samples)
        if field_samples is not None:
            psf["field_samples"] = [int(field_samples[0]), int(field_samples[1])]
    return raw
