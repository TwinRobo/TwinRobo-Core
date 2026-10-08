"""The lidars of the catalog: ``<maker>/<model>/lidar.yaml`` next to the cameras."""

from __future__ import annotations

from pathlib import Path

from .spec import LidarSpec

SPEC_FILENAME = "lidar.yaml"


class LidarRegistry:
    def __init__(self, roots: list[str | Path] | None = None):
        from ..spec import BUILTIN_CATALOG_DIR

        self.roots = [Path(r) for r in (roots or [BUILTIN_CATALOG_DIR])]

    def list(self) -> list[str]:
        out = []
        for root in self.roots:
            for path in sorted(root.rglob(SPEC_FILENAME)):
                out.append(path.parent.relative_to(root).as_posix())
        return out

    def resolve(self, lidar_id: str) -> Path:
        for root in self.roots:
            path = root.joinpath(*lidar_id.split("/"), SPEC_FILENAME)
            if path.is_file():
                return path
        raise KeyError(f"no lidar {lidar_id!r} in the catalog")

    def load(self, lidar_id: str) -> LidarSpec:
        return LidarSpec.from_yaml(self.resolve(lidar_id))
