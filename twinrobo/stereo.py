"""Stereo camera modules: two eyes in one housing, placed as one unit.

A module file (``<catalog root>/<id>/module.yaml``) names the per-eye CameraSpecs
and the right eye's pose in the left eye's camera frame. The module's pose is
the LEFT eye's (Stereolabs convention), so placing a module means placing its
left eye; the right eye follows rigidly (on a robot link it follows the link).

Simulator side, a module is expanded into two ordinary cameras:
`twinrobo.plugins.mujoco.mounts.expand_mount` turns a module mount into left/right
`CameraMount`s, which render like any
other custom camera. Rectified output (what vendor SDKs deliver) is handled per
eye by `CameraTwin.process(rectify=True)`; with parallel eyes, identical
intrinsics and a horizontal baseline the rectified pair is row-aligned.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .exceptions import SpecError
from .rig import GroupRig

MODULE_FILENAME = "module.yaml"
EYES = ("left", "right")
OUTPUTS = ("rectified", "raw")


@dataclass(frozen=True)
class Eye:
    spec: Path | None  # CameraSpec of this eye (None: simulator pinhole)
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # in the left eye's camera frame
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StereoModule:
    id: str
    left: Eye
    right: Eye
    manufacturer: str | None = None
    product: str | None = None
    housing_mm: tuple[float, float, float] | None = None
    output: str = "rectified"
    source_path: Path | None = field(default=None, compare=False)

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(np.subtract(self.right.translation_m, self.left.translation_m)))

    def with_baseline(self, baseline_m: float) -> StereoModule:
        """The same module with another baseline: the right eye moved along the same
        direction from the left eye (the housing widened or narrowed with it).

        Robots differ in how far apart they mount a stereo pair; this lets one catalog
        module stand for any of them.
        """
        baseline_m = float(baseline_m)
        if not 0.001 <= baseline_m <= 2.0:
            raise ValueError(f"stereo baseline must be within 1 mm and 2 m, got {baseline_m} m")
        t = np.subtract(self.right.translation_m, self.left.translation_m)
        if np.linalg.norm(t) == 0:
            raise ValueError(f"{self.id}: the eyes coincide; no baseline direction")
        right = np.asarray(self.left.translation_m) + t / np.linalg.norm(t) * baseline_m
        housing = self.housing_mm
        if housing is not None:
            housing = (housing[0] + 1000 * (baseline_m - self.baseline_m), *housing[1:])
        return replace(
            self,
            right=replace(self.right, translation_m=tuple(float(v) for v in right)),
            housing_mm=housing,
        )

    @property
    def label(self) -> str:
        return " ".join(x for x in (self.manufacturer, self.product) if x) or self.id

    @classmethod
    def from_yaml(cls, path: str | Path) -> StereoModule:
        path = Path(path)
        try:
            d = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise SpecError(f"cannot read stereo module {path}: {e}") from e
        if not isinstance(d, dict) or d.get("type") != "stereo":
            raise SpecError(f"{path}: not a stereo module (type: stereo)")
        if d.get("reference", "left") != "left":
            raise SpecError(f"{path}: only reference: left is supported")
        eyes = {}
        for name in EYES:
            e = (d.get("eyes") or {}).get(name)
            if e is None:
                raise SpecError(f"{path}: missing eyes.{name}")
            spec = e.get("spec")
            eyes[name] = Eye(
                spec=(path.parent / spec).resolve() if spec else None,
                translation_m=tuple(float(v) for v in e.get("translation_m", (0, 0, 0))),
                rotation_rpy_deg=tuple(float(v) for v in e.get("rotation_rpy_deg", (0, 0, 0))),
            )
        output = d.get("output", "rectified")
        if output not in OUTPUTS:
            raise SpecError(f"{path}: output must be one of {OUTPUTS}")
        housing = d.get("housing_mm")
        return cls(
            id=str(d.get("id") or path.parent.name),
            left=eyes["left"],
            right=eyes["right"],
            manufacturer=d.get("manufacturer"),
            product=d.get("product"),
            housing_mm=tuple(float(v) for v in housing) if housing else None,
            output=output,
            source_path=path,
        )

    def eye(self, name: str) -> Eye:
        return self.left if name == "left" else self.right

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "baseline_mm": round(1000 * self.baseline_m, 3),
            "housing_mm": list(self.housing_mm) if self.housing_mm else None,
            "output": self.output,
            "eyes": {
                n: {"spec": str(self.eye(n).spec) if self.eye(n).spec else None} for n in EYES
            },
        }


def _rot(rpy_deg) -> np.ndarray:
    """Rotation of an eye relative to the left eye's camera frame (x, y, z Euler, degrees)."""
    rx, ry, rz = (math.radians(v) for v in rpy_deg)
    cx, sx, cy, sy, cz, sz = (
        math.cos(rx),
        math.sin(rx),
        math.cos(ry),
        math.sin(ry),
        math.cos(rz),
        math.sin(rz),
    )
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def eye_poses(
    module: StereoModule, pos, R_left: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{eye: (position, camera axes)}`` in the parent frame, from the left eye's pose.

    Camera axes follow MuJoCo (columns x right, y up, z backwards).
    """
    pos, R_left = np.asarray(pos, float), np.asarray(R_left, float)
    out = {}
    for name in EYES:
        e = module.eye(name)
        out[name] = (pos + R_left @ np.asarray(e.translation_m), R_left @ _rot(e.rotation_rpy_deg))
    return out


def eye_name(module_camera: str, eye: str) -> str:
    return f"{module_camera}_{'L' if eye == 'left' else 'R'}"


class StereoTwin:
    """A stereo module: two `CameraTwin` eyes that are captured together.

    Build one with `from_catalog`, then `attach` it to the simulator's two cameras. Eyes with
    the same spec share one `CameraTwin`.

    Args:
        module: The module.
        left: The left eye's twin.
        right: The right eye's twin.
    """

    def __init__(self, module: StereoModule, left: Any, right: Any):
        self.module = module
        self.left = left
        self.right = right

    @classmethod
    def from_module(
        cls,
        module: StereoModule,
        device: Any = None,
        cache: Any = None,
        build_psf: bool = True,
        isp: Any = None,
        *,
        sensor: Any = None,
    ) -> StereoTwin:
        """Build the eyes of a module.

        Args:
            module: The module.
            device: Torch device.
            cache: Where PSF banks are stored.
            build_psf: ``False`` skips the PSF banks, for ray casting only.
            isp: The sensor and ISP (see `twinrobo.camera.build_sensor_isp`).
            sensor: Your sensor, used only with your own `ISPModel`.

        Returns:
            The stereo twin.
        """
        from .camera import CameraTwin

        twins: dict[Path, CameraTwin] = {}
        for eye in (module.left, module.right):
            if eye.spec is None:
                raise SpecError(f"{module.id}: an eye without a spec has no twin")
            if eye.spec not in twins:
                twins[eye.spec] = CameraTwin.from_spec(
                    eye.spec, device, cache, build_psf, isp, sensor=sensor
                )
        return cls(module, twins[module.left.spec], twins[module.right.spec])

    @classmethod
    def from_spec(cls, path: str | Path, **options: Any) -> StereoTwin:
        """Build a stereo twin from a ``module.yaml``; ``options`` are those of `from_module`."""
        return cls.from_module(StereoModule.from_yaml(path), **options)

    @classmethod
    def from_catalog(
        cls, module_id: str, registry: ModuleRegistry | None = None, **options: Any
    ) -> StereoTwin:
        """Build a stereo twin from the catalog.

        Args:
            module_id: The catalog ID, such as ``"stereolabs/zed-x/2.2mm"``.
            registry: The catalog to read. By default the bundled one.
            **options: Passed to `from_module`.

        Returns:
            The stereo twin.
        """
        return cls.from_module((registry or ModuleRegistry()).load(module_id), **options)

    @property
    def baseline_m(self) -> float:
        return self.module.baseline_m

    def set_isp(self, isp: Any, sensor: Any = None) -> None:
        """Switch both eyes' sensor and ISP (see `twinrobo.camera.CameraTwin.set_isp`)."""
        from .camera import build_sensor_isp

        twins = {id(t): t for t in (self.left, self.right)}.values()
        for twin in twins:  # a selection one eye cannot use changes neither
            build_sensor_isp(twin.spec, isp, sensor)
        for twin in twins:
            twin.set_isp(isp, sensor)

    def attach(
        self,
        backend: Any,
        camera: str | None = None,
        *,
        left: str | None = None,
        right: str | None = None,
        **options: Any,
    ) -> GroupRig:
        """Bind the eyes to two simulator cameras.

        Name both cameras, or give ``camera`` for a pair made by
        `twinrobo.plugins.mujoco.mounts.expand_mount`: ``<camera>_L`` and ``<camera>_R``. The
        cameras stay where the simulator has them, so they set the real baseline.

        Args:
            backend: The simulator backend (see `twinrobo.rig`).
            camera: The pair's name.
            left: The left eye's camera.
            right: The right eye's camera.
            **options: Passed to each eye. ``rectify`` defaults to the module's output.

        Returns:
            A `GroupRig`. Its streams are ``"left"`` and ``"right"``.
        """
        left = left or (eye_name(camera, "left") if camera else None)
        right = right or (eye_name(camera, "right") if camera else None)
        if left is None or right is None:
            raise ValueError("give camera=, or both left= and right=")
        baseline = getattr(backend, "baseline", lambda *_: None)(left, right)
        if baseline is not None and abs(baseline - self.baseline_m) > 1e-3:
            warnings.warn(
                f"{left!r} and {right!r} are {baseline * 1000:.1f} mm apart; "
                f"{self.module.id} has a {self.baseline_m * 1000:.1f} mm baseline",
                stacklevel=2,
            )
        options.setdefault("rectify", self.module.output == "rectified")
        return GroupRig(
            {
                "left": self.left.attach(backend, left, **options),
                "right": self.right.attach(backend, right, **options),
            }
        )


class ModuleRegistry:
    """Stereo modules in the catalog roots (same roots as `CatalogRegistry`)."""

    def __init__(self, roots: list[Path] | None = None):
        from .registry import CatalogRegistry

        self.roots = roots if roots is not None else CatalogRegistry().roots

    def list(self) -> list[StereoModule]:
        seen: dict[str, StereoModule] = {}
        for root in self.roots:
            if not Path(root).is_dir():
                continue
            for p in sorted(Path(root).rglob(MODULE_FILENAME)):
                m = StereoModule.from_yaml(p)
                seen.setdefault(m.id, m)
        return list(seen.values())

    def load(self, module_id: str) -> StereoModule:
        from .registry import validate_camera_id

        segments = validate_camera_id(module_id)
        for root in self.roots:
            p = Path(root).joinpath(*segments, MODULE_FILENAME)
            if p.is_file():
                return StereoModule.from_yaml(p)
        raise SpecError(f"stereo module {module_id!r} not found")
