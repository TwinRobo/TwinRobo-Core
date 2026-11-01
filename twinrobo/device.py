"""Devices: several sensors in one product, such as an RGB camera beside a stereo IR pair.

A device file (``<catalog root>/<id>/device.yaml``) names its parts by catalog ID. A part's
pose is in the device's frame; it may be unknown (no ``translation_m``), since vendors do not
always publish where each sensor sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import SpecError
from .rig import GroupRig

DEVICE_FILENAME = "device.yaml"
PART_TYPES = ("camera", "stereo", "lidar")


@dataclass(frozen=True)
class DevicePart:
    type: str  # camera | stereo | lidar
    id: str  # its catalog ID
    translation_m: tuple[float, float, float] | None = None  # in the device's frame
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DeviceSpec:
    id: str
    parts: dict[str, DevicePart]
    manufacturer: str | None = None
    product: str | None = None
    source_path: Path | None = field(default=None, compare=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeviceSpec:
        path = Path(path)
        try:
            d = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise SpecError(f"cannot read device {path}: {e}") from e
        if not isinstance(d, dict) or d.get("type") != "device":
            raise SpecError(f"{path}: not a device (type: device)")
        parts = {}
        for name, p in (d.get("parts") or {}).items():
            if p.get("type") not in PART_TYPES or not p.get("id"):
                raise SpecError(f"{path}: part {name!r} needs an id and a type in {PART_TYPES}")
            t = p.get("translation_m")
            parts[name] = DevicePart(
                type=p["type"],
                id=str(p["id"]),
                translation_m=tuple(float(v) for v in t) if t is not None else None,
                rotation_rpy_deg=tuple(float(v) for v in p.get("rotation_rpy_deg", (0, 0, 0))),
            )
        if not parts:
            raise SpecError(f"{path}: a device needs parts")
        return cls(
            id=str(d.get("id") or path.parent.name),
            parts=parts,
            manufacturer=d.get("manufacturer"),
            product=d.get("product"),
            source_path=path,
        )


class DeviceRegistry:
    """Devices in the catalog roots (same roots as `CatalogRegistry`)."""

    def __init__(self, roots: list[Path] | None = None):
        from .registry import CatalogRegistry

        self.roots = roots if roots is not None else CatalogRegistry().roots

    def list(self) -> list[str]:
        """All device IDs across the roots."""
        seen: dict[str, None] = {}
        for root in self.roots:
            for p in sorted(Path(root).rglob(DEVICE_FILENAME)):
                seen[DeviceSpec.from_yaml(p).id] = None
        return list(seen)

    def load(self, device_id: str) -> DeviceSpec:
        """The device with this ID."""
        from .registry import validate_camera_id

        segments = validate_camera_id(device_id)
        for root in self.roots:
            p = Path(root).joinpath(*segments, DEVICE_FILENAME)
            if p.is_file():
                return DeviceSpec.from_yaml(p)
        raise SpecError(f"device {device_id!r} not found")


class DeviceTwin:
    """Any sensors working as one device: cameras, stereo pairs and lidars.

    Build one from the catalog with `from_catalog`, or combine twins directly:
    ``DeviceTwin({"rgb": camera_twin, "front": lidar_twin})``.

    Args:
        parts: The twins, by part name.
        spec: The catalog entry, if there is one.
    """

    def __init__(self, parts: dict[str, Any], spec: DeviceSpec | None = None):
        if not parts:
            raise ValueError("a device needs parts")
        self.parts = parts
        self.spec = spec

    @classmethod
    def from_spec(
        cls,
        spec: str | Path | DeviceSpec,
        device: Any = None,
        build_psf: bool = True,
        roots: list[Path] | None = None,
    ) -> DeviceTwin:
        """Build a device from a ``device.yaml``.

        Args:
            spec: The file's path, or a loaded `DeviceSpec`.
            device: Torch device.
            build_psf: ``False`` skips the cameras' PSF banks, for ray casting only.
            roots: Catalog roots for the parts. By default the standard ones.

        Returns:
            The device.
        """
        from .camera import CameraTwin
        from .lidar import LidarRegistry, LidarTwin
        from .registry import CatalogRegistry
        from .stereo import ModuleRegistry, StereoTwin

        spec = spec if isinstance(spec, DeviceSpec) else DeviceSpec.from_yaml(spec)
        parts: dict[str, Any] = {}
        for name, part in spec.parts.items():
            if part.type == "lidar":
                parts[name] = LidarTwin.from_catalog(part.id, LidarRegistry(roots), device)
            elif part.type == "stereo":
                parts[name] = StereoTwin.from_catalog(
                    part.id, ModuleRegistry(roots), device=device, build_psf=build_psf
                )
            else:
                parts[name] = CameraTwin.from_catalog(
                    part.id, CatalogRegistry(roots), device, build_psf=build_psf
                )
        return cls(parts, spec)

    @classmethod
    def from_catalog(
        cls, device_id: str, registry: DeviceRegistry | None = None, **options: Any
    ) -> DeviceTwin:
        """Build a device from the catalog.

        Args:
            device_id: The catalog ID, such as ``"intel/realsense-d435"``.
            registry: The catalog to read. By default the bundled one.
            **options: Passed to `from_spec`.

        Returns:
            The device.
        """
        registry = registry or DeviceRegistry()
        options.setdefault("roots", registry.roots)
        return cls.from_spec(registry.load(device_id), **options)

    def attach(self, backend: Any, **bindings: Any) -> GroupRig:
        """Bind every part to a simulator.

        Name each part: a string is its simulator camera, and a dict is what that part's
        ``attach`` takes: ``device.attach(backend, rgb="cam", ir={"left": "l", "right": "r"})``.

        Args:
            backend: The simulator backend (see `twinrobo.rig`).
            **bindings: One entry per part.

        Returns:
            A `GroupRig`. Its streams are named by part: ``"rgb"``, ``"ir/left"``.
        """
        if set(bindings) != set(self.parts):
            raise ValueError(f"bind every part: {sorted(self.parts)}, got {sorted(bindings)}")
        rigs = {}
        for name, twin in self.parts.items():
            bound = bindings[name]
            bound = {"camera": bound} if isinstance(bound, str) else bound
            rigs[name] = twin.attach(backend, **bound)
        return GroupRig(rigs)
