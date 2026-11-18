"""Glass names in lens files, resolved to DeepLens materials.

A catalog glass is looked up by name in DeepLens' bundled catalogs (CDGM, SCHOTT, MISC,
PLASTIC2022, then refractiveindex.info) and in any ``.agf`` catalogs given, the catalog the
file names first. Names are compared without case or punctuation, so CODE V's ``NBK7_SCHOTT``
finds SCHOTT's ``N-BK7``. A glass that is in no catalog is an error unless the caller allows
the file's own nd/vd to stand in, which approximates its dispersion.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import tempfile
from pathlib import Path

from ..exceptions import LensFileError
from .prescription import Material, Prescription

#: DeepLens' bundled catalogs in the order a bare name resolves in DeepLens (its merged table
#: lets SCHOTT override CDGM, CDGM PLASTIC2022, PLASTIC2022 MISC).
BUNDLED = ("SCHOTT", "CDGM", "PLASTIC2022", "MISC")
#: AGF dispersion formulas DeepLens evaluates: 1 Schott, 2 Sellmeier 1.
AGF_FORMULAS = (1, 2)


def _norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.lower())


def _key(catalog: str, name: str, entry: dict) -> str:
    """A DeepLens material key for one catalog's glass. It holds a hash of the glass's data, so
    a changed catalog gives new keys (and new cache keys), never a stale glass."""
    digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:10]
    return f"agf:{catalog.lower()}:{name}:{digest}"


def register(glass_data: dict[str, dict]) -> None:
    """Make glasses saved with a lens (``glass_data`` in its JSON) known to DeepLens."""
    m = _materials_module()
    for key, entry in glass_data.items():
        m.MATERIAL_data[key] = entry


def bundled_catalog(key: str) -> str | None:
    """The bundled catalog whose glass a bare DeepLens material key names, if any."""
    m = _materials_module()
    entry = m.MATERIAL_data.get(key.lower())
    if entry is None:
        return None
    tables = {"SCHOTT": m.SCHOTT_data, "CDGM": m.CDGM_data, "PLASTIC2022": m.PLASTIC_data,
              "MISC": m.MISC_data}  # fmt: skip
    return next((c for c, t in tables.items() if t.get(key.lower()) is entry), None)


def _materials_module():
    try:
        from deeplens.material import materials
    except ImportError as e:  # pragma: no cover - exercised only without DeepLens
        raise LensFileError("resolving glasses needs DeepLens (pip install -e .)") from e
    return materials


class GlassCatalogs:
    """The catalogs glass names are searched in: DeepLens' and any ``.agf`` files given."""

    def __init__(self, agf: dict[str, str | Path | bytes] | list[str | Path] | None = None):
        m = _materials_module()
        self._m = m
        self.catalogs: dict[str, dict[str, str]] = {}  # catalog -> normalized name -> key
        tables = {
            "SCHOTT": m.SCHOTT_data,
            "CDGM": m.CDGM_data,
            "PLASTIC2022": m.PLASTIC_data,
            "MISC": m.MISC_data,
        }
        for cat in BUNDLED:
            index = {}
            for name, entry in tables[cat].items():
                if m.MATERIAL_data.get(name) is entry:
                    index[_norm(name)] = name
                else:  # another catalog's glass of the same name holds DeepLens' bare key
                    key = _key(cat, name, entry)
                    m.MATERIAL_data[key] = entry
                    index[_norm(name)] = key
            self.catalogs[cat] = index
        self.other: dict[str, str] = {}
        for table in ("FORMULA", "INTERP"):
            self.other.update({_norm(k): k for k in m.RII_data.get(table, {})})
        for table in ("SELLMEIER_TABLE", "SCHOTT_TABLE", "INTERP_TABLE", "MATERIAL_TABLE"):
            self.other.update({_norm(k): k for k in m.CUSTOM_data.get(table, {})})
        items = agf.items() if isinstance(agf, dict) else ((Path(a).stem, a) for a in agf or ())
        for name, src in items:
            self.add_agf(Path(str(name)).stem, src)

    def add_agf(self, catalog: str, src: str | Path | bytes) -> None:
        """Register an ``.agf`` catalog (a path or the file's bytes) under ``catalog``."""
        catalog = catalog.upper()
        if isinstance(src, bytes):
            with tempfile.NamedTemporaryFile(suffix=".agf", delete=False) as f:
                f.write(src)
            path = Path(f.name)
        else:
            path = Path(src)
        try:
            data = self._m.read_agf(str(path))
        except (OSError, ValueError, IndexError) as e:
            raise LensFileError(f"cannot read glass catalog {catalog}: {e}") from e
        finally:
            if isinstance(src, bytes):
                path.unlink(missing_ok=True)
        self.add_glasses(catalog, data)

    def add_glasses(self, catalog: str, data: dict[str, dict]) -> None:
        """Register glasses (DeepLens AGF entries by name) under ``catalog``. A catalog given
        by the caller comes first, and its glasses replace those of a catalog of the same name
        (an updated SCHOTT.agf)."""
        catalog = catalog.upper()
        index = {}
        for name, entry in data.items():
            key = _key(catalog, name, entry)
            index[_norm(name)] = key
            self._m.MATERIAL_data[key] = entry  # DeepLens looks materials up by this key
        merged = {**self.catalogs.pop(catalog, {}), **index}
        self.catalogs = {catalog: merged, **self.catalogs}

    def glass_data(self, keys) -> dict[str, dict]:
        """The data of the glasses among ``keys`` that only this process knows (catalog glasses
        registered under TwinRobo's keys), to save with a lens that uses them."""
        return {
            k: self._m.MATERIAL_data[k]
            for k in keys
            if isinstance(k, str) and k.startswith("agf:") and k in self._m.MATERIAL_data
        }

    def holds(self, catalog: str, name: str) -> bool:
        return _norm(name) in self.catalogs.get(catalog.upper(), {})

    def resolve(self, m: Material, search: list[str] | None = None) -> str | None:
        """The DeepLens material key for a catalog glass, or None if no catalog has it."""
        found = self.locate(m, search)
        return None if found is None else found[1]

    def locate(self, m: Material, search: list[str] | None = None) -> tuple[str, str] | None:
        """``(catalog, DeepLens key)`` of a catalog glass: the catalog the glass names, then
        the ``search`` order (a file's GCAT), then the rest; None if no catalog has it.
        Glasses found only in refractiveindex.info report the catalog ``""``."""
        n = _norm(m.name)
        order = [c.upper() for c in ([m.catalog] if m.catalog else []) + list(search or [])]
        order += [c for c in self.catalogs if c not in order]
        for cat in order:
            key = self.catalogs.get(cat, {}).get(n)
            if key is not None:
                self._check_formula(key, m)
                return cat, key
        key = self.other.get(n)
        return None if key is None else ("", key)

    def _check_formula(self, key: str, m: Material) -> None:
        entry = self._m.MATERIAL_data.get(key)
        if entry is not None and int(entry.get("calculate_mode", 0)) not in AGF_FORMULAS:
            raise LensFileError(
                f"glass {m.name}: its catalog uses dispersion formula "
                f"{int(entry['calculate_mode'])}; DeepLens evaluates formulas 1 and 2 only"
            )


def deeplens_material(
    m: Material,
    catalogs: GlassCatalogs,
    search: list[str] | None = None,
    allow_model_glass: bool = False,
    notes: list[str] | None = None,
) -> str:
    """The DeepLens material string for ``m``."""
    if m.is_air:
        return "air"
    if m.is_model:
        if notes is not None:
            notes.append(
                f"model glass nd={m.nd:g} vd={m.vd:g}: dispersion approximated (Cauchy from nd, vd)"
            )
        return f"{m.nd!r}/{m.vd!r}"
    key = catalogs.resolve(m, search)
    if key is not None:
        return key
    if allow_model_glass and m.nd and m.vd:
        if notes is not None:
            notes.append(
                f"glass {m.name} is in no catalog: using the file's nd={m.nd:g} vd={m.vd:g} "
                "(dispersion approximated)"
            )
        return f"{m.nd!r}/{m.vd!r}"
    raise LensFileError(
        f"glass {m.describe()} is in no catalog searched ({', '.join(catalogs.catalogs)}, "
        "refractiveindex.info); pass its .agf catalog, or allow_model_glass=True to use the "
        "file's nd/vd"
    )


def resolve_all(
    p: Prescription, catalogs: GlassCatalogs, allow_model_glass: bool = False
) -> list[str]:
    """DeepLens material strings for the medium after each surface of ``p``."""
    out = []
    for i, s in enumerate(p.surfaces):
        try:
            out.append(
                deeplens_material(
                    s.material, catalogs, p.glass_catalogs, allow_model_glass, p.notes
                )
            )
        except LensFileError as e:
            raise LensFileError(f"{p.source_path or p.name}: surface {i}: {e}") from None
    return out


def with_catalogs(
    p: Prescription, catalogs: GlassCatalogs, one_search_order: bool = False
) -> Prescription:
    """``p`` with each catalog glass labeled with the catalog it resolves in, and the file's
    catalog search order set to those catalogs, so an exported file names exactly that glass.

    ``one_search_order``: the format names glasses without their catalog (Zemax: one GCAT
    order for the whole lens). The order is chosen so that every glass resolves in its own
    catalog; a lens for which no order does (SCHOTT F2 beside CDGM F2) raises.
    """
    q = p.copy()
    used: list[str] = []
    glasses: list[tuple[str, str]] = []
    for s in q.surfaces:
        m = s.material
        if m.is_air or m.is_model:
            continue
        found = catalogs.locate(m, q.glass_catalogs)
        if found is None or not found[0]:
            continue
        cat = found[0]
        s.material = Material(name=m.name, catalog=cat, nd=m.nd, vd=m.vd)
        glasses.append((cat, m.name))
        if cat not in used:
            used.append(cat)
    if one_search_order and len(used) > 1:
        for order in itertools.permutations(used):
            if all(
                not any(catalogs.holds(c, name) for c in order[: order.index(cat)])
                for cat, name in glasses
            ):
                used = list(order)
                break
        else:
            raise LensFileError(
                "this lens uses glasses of the same name from different catalogs "
                f"({sorted(set(glasses))}); a .zmx names glasses without their catalog, so it "
                "cannot tell them apart. Export to .seq or .csv instead."
            )
    q.glass_catalogs = used + [c for c in q.glass_catalogs if c not in used]
    return q
