"""RoboCasa integration: realistic kitchen scenes (robosuite 1.5, MuJoCo) for CameraTwin.

RoboCasa ships procedurally assembled kitchens (layouts x styles, textured
fixtures, appliances, Objaverse objects) and human demonstrations as LeRobot
datasets. Every episode stores its exact generated kitchen (``model.xml.gz``),
its simulator states (``states.npz``) and ``ep_meta.json``, so an episode is
replayed exactly by loading that kitchen and applying the recorded states.

Environment. RoboCasa needs robosuite 1.5 and LIBERO needs 1.4.1, so they
cannot share a Python process: use the ``twinrobo-robocasa`` environment
(see README). Assets are the RoboCasa asset pack, which lives outside the
package; `setup_robocasa` links it into the installed package when missing.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from ..exceptions import SimulatorError

ASSET_PACK_ENV = "ROBOCASA_ASSETS"
#: RoboCasa robots selectable in its kitchens (the demos are PandaOmron).
ROBOTS = ["PandaOmron"]
LINKED = ("objects", "textures", "generative_textures")


def setup_robocasa(asset_pack: str | Path | None = None) -> Path:
    """Check RoboCasa is importable and its assets are present; returns the assets dir.

    ``asset_pack`` (or ``$ROBOCASA_ASSETS``): a RoboCasa asset pack directory
    (with ``objects/``, ``textures/``, ``generative_textures/``, ``fixtures/``).
    Missing top-level asset folders in the installed package are symlinked to it;
    nothing in the pack is modified.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    try:
        import robocasa
        import robosuite
    except ImportError as e:
        raise SimulatorError(
            "RoboCasa is not installed in this environment (needs robosuite 1.5; "
            "use the twinrobo-robocasa environment)"
        ) from e
    if not robosuite.__version__.startswith("1.5"):
        raise SimulatorError(f"RoboCasa needs robosuite 1.5, found {robosuite.__version__}")
    assets = Path(robocasa.__file__).parent / "models" / "assets"
    pack = asset_pack or os.environ.get(ASSET_PACK_ENV)
    if pack:
        pack = Path(pack).expanduser()
        for name in LINKED:
            dst, src = assets / name, pack / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve(), target_is_directory=True)
        _link_fixtures(pack / "fixtures", assets / "fixtures")
    if not (assets / "objects").exists():
        raise SimulatorError(
            f"RoboCasa assets missing in {assets}; pass the asset pack via ${ASSET_PACK_ENV}"
        )
    return assets


def _link_fixtures(src: Path, dst: Path) -> None:
    """Link every asset of the pack that the package lacks.

    Fixture folders mix assets shipped with the code and downloaded ones, at any
    depth (e.g. ``cabinets/cabinet_panels/CabinetDoorPanel018``): a folder the
    package lacks is linked whole, a folder it has is descended into.
    """
    if not src.is_dir():
        return
    for item in src.iterdir():
        tgt = dst / item.name
        if tgt.is_symlink() or not tgt.exists():
            if not tgt.exists():
                tgt.symlink_to(item.resolve(), target_is_directory=item.is_dir())
            continue
        if item.is_dir() and tgt.is_dir():
            _link_fixtures(item, tgt)


def make_env(env_args: dict[str, Any], robot: str | None = None, seed: int = 0):
    """A RoboCasa env from a dataset's ``env_args`` (offscreen rendering, no camera obs)."""
    setup_robocasa()
    import robosuite

    kw = dict(env_args.get("env_kwargs") or {})
    kw.pop("env_name", None)
    if robot is not None:
        kw["robots"] = robot
        if robot != env_args.get("env_kwargs", {}).get("robots"):
            kw.pop("controller_configs", None)
    kw.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        ignore_done=True,
        seed=seed,
        renderer="mjviewer",
    )
    kw.pop("camera_names", None)
    kw.pop("camera_heights", None)
    kw.pop("camera_widths", None)
    kw.pop("camera_depths", None)
    return robosuite.make(env_args["env_name"], **kw)


def reset_to_episode(env, model_xml: str, ep_meta: dict[str, Any]) -> None:
    """Load an episode's exact kitchen (as RoboCasa's own playback script does)."""
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()


def read_episode(dataset_dir: Path, episode: str) -> tuple:
    """``(states [T, S], model_xml, ep_meta)`` of a RoboCasa episode (``episode_000004``)."""
    import numpy as np

    ep = Path(dataset_dir) / "extras" / episode
    with np.load(ep / "states.npz") as z:
        states = np.asarray(z["states"] if "states" in z else z[z.files[0]], dtype=np.float64)
    with gzip.open(ep / "model.xml.gz", "rb") as f:
        xml = f.read().decode()
    ep_meta = json.loads((ep / "ep_meta.json").read_text())
    return states, xml, ep_meta


def env_args(dataset_dir: Path) -> dict[str, Any]:
    meta = json.loads((Path(dataset_dir) / "extras" / "dataset_meta.json").read_text())
    return meta["env_args"]
