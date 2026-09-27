"""On-disk cache of precomputed PSF banks.

Banks are keyed by *content*, not by file path: the caller supplies a
JSON-serializable key (lens file hash, sensor resolution, focus, sampling
parameters, backend version, bank format). The cache stores
``<root>/<sha256(key)[:24]>.pt`` plus a ``.json`` sidecar holding
``{"key": ..., "used_by": [...]}``. ``used_by`` lists the camera presets (spec
files) that use the bank. Content-keyed banks can be shared by presets with
identical optics, so deleting a preset removes a bank only when no other
preset uses it (`PSFCache.release`). Sidecars written before ``used_by``
existed hold the bare key and are never removed automatically.

Root: ``$TWINROBO_CACHE/psf`` if set, else ``~/.cache/twinrobo/psf``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .psf import BANK_FORMAT_VERSION, PSFBank

CACHE_ENV = "TWINROBO_CACHE"


def default_cache_root() -> Path:
    base = os.environ.get(CACHE_ENV)
    return (Path(base) if base else Path.home() / ".cache" / "twinrobo") / "psf"


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PSFCache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_cache_root()

    @staticmethod
    def key_hash(key: dict[str, Any]) -> str:
        key = {**key, "bank_format": BANK_FORMAT_VERSION}
        blob = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def path_for(self, key: dict[str, Any]) -> Path:
        return self.root / f"{self.key_hash(key)}.pt"

    def get(self, key: dict[str, Any], device=None) -> PSFBank | None:
        path = self.path_for(key)
        return PSFBank.load(path, device=device) if path.is_file() else None

    @staticmethod
    def _norm(source: str | Path) -> str:
        return str(Path(source).resolve())

    def _read_sidecar(self, sidecar: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and "used_by" in data else None

    def _add_user(self, pt: Path, key: dict[str, Any], source: str | Path | None) -> None:
        sidecar = pt.with_suffix(".json")
        data = self._read_sidecar(sidecar)
        if data is None:  # new bank, or a legacy sidecar holding only the key
            data = {"key": key, "used_by": []}
        if source is not None and self._norm(source) not in data["used_by"]:
            data["used_by"].append(self._norm(source))
        sidecar.write_text(json.dumps(data, sort_keys=True, indent=2, default=str))

    def put(self, key: dict[str, Any], bank: PSFBank, source: str | Path | None = None) -> Path:
        path = self.path_for(key)
        bank.save(path)
        path.with_suffix(".json").unlink(missing_ok=True)
        self._add_user(path, key, source)
        return path

    def get_or_build(
        self,
        key: dict[str, Any],
        build: Callable[[], PSFBank],
        device=None,
        source: str | Path | None = None,
    ) -> PSFBank:
        """Load or build the bank for ``key``; ``source`` (a preset file) is recorded as a user."""
        bank = self.get(key, device=device)
        if bank is None:
            bank = build()
            self.put(key, bank, source)
            if device is not None:
                bank = bank.to(device)
        elif source is not None:
            self._add_user(self.path_for(key), key, source)
        return bank

    def release(self, source: str | Path) -> list[Path]:
        """Drop ``source`` from every bank's users; delete banks nobody uses any more.

        Returns the deleted ``.pt`` files. Legacy banks (no ``used_by``) are kept.
        """
        src, removed = self._norm(source), []
        if not self.root.is_dir():
            return removed
        for sidecar in sorted(self.root.glob("*.json")):
            data = self._read_sidecar(sidecar)
            if data is None or src not in data["used_by"]:
                continue
            data["used_by"].remove(src)
            if data["used_by"]:
                sidecar.write_text(json.dumps(data, sort_keys=True, indent=2, default=str))
            else:
                pt = sidecar.with_suffix(".pt")
                pt.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                removed.append(pt)
        return removed
