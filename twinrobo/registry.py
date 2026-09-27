"""Local camera catalog resolution.

Catalog IDs are slash-separated and must identify the *complete* configuration,
e.g. ``lucid/triton/imx568/edmund-8mm/f2.8``. An ID resolves to
``<root>/<id>/camera.yaml`` in the first catalog root that contains it.

Catalog roots come from, in order: roots passed explicitly, then the
``TWINROBO_CATALOG`` environment variable (``os.pathsep``-separated), then the
built-in catalog shipped with the package (``twinrobo/catalog/``). A remote catalog
client is a Phase 5 concern and is not implemented here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .exceptions import CatalogError
from .spec import BUILTIN_CATALOG_DIR, CameraSpec

SPEC_FILENAME = "camera.yaml"
CATALOG_ENV = "TWINROBO_CATALOG"
BUILTIN_CATALOG = BUILTIN_CATALOG_DIR

_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def validate_camera_id(camera_id: str) -> list[str]:
    """Split and validate a catalog ID; return its path segments."""
    segments = camera_id.split("/")
    if not camera_id or any(not _SEGMENT.match(s) or s in (".", "..") for s in segments):
        raise CatalogError(
            f"Invalid camera ID {camera_id!r}: use lowercase slash-separated segments "
            "of [a-z0-9._-], e.g. 'vendor/model/sensor/lens/f2.8'"
        )
    return segments


class CatalogRegistry:
    """Resolve catalog IDs to CameraSpec files on the local filesystem."""

    def __init__(
        self,
        roots: list[str | Path] | None = None,
        use_env: bool = True,
        use_builtin: bool = True,
    ):
        self.roots: list[Path] = [Path(r) for r in (roots or [])]
        if use_env and os.environ.get(CATALOG_ENV):
            self.roots += [Path(p) for p in os.environ[CATALOG_ENV].split(os.pathsep) if p]
        if use_builtin and BUILTIN_CATALOG.is_dir():
            self.roots.append(BUILTIN_CATALOG)

    def resolve(self, camera_id: str) -> Path:
        segments = validate_camera_id(camera_id)
        for root in self.roots:
            candidate = root.joinpath(*segments, SPEC_FILENAME)
            if candidate.is_file():
                return candidate
        searched = ", ".join(str(r) for r in self.roots) or f"<none; set {CATALOG_ENV}>"
        raise CatalogError(f"Camera {camera_id!r} not found in catalog roots: {searched}")

    def load(self, camera_id: str) -> CameraSpec:
        return CameraSpec.from_yaml(self.resolve(camera_id))

    def list(self) -> list[str]:
        """All camera IDs available across roots (first root wins on duplicates)."""
        ids: dict[str, None] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for spec in sorted(root.rglob(SPEC_FILENAME)):
                ids.setdefault(spec.parent.relative_to(root).as_posix(), None)
        return list(ids)
