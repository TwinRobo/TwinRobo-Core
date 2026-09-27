"""LIBERO integration: CameraTwin cameras in LIBERO environments.

LIBERO runs on robosuite, which runs on MuJoCo. This package provides:

- `CameraTwinLiberoEnv`: wraps a LIBERO env and replaces camera observations
  (e.g. ``agentview_image``) with CameraTwin frames, keeping each key's shape,
  dtype and image convention, so policies and eval scripts run unchanged.
- `make_env` / `make_env_from_bddl`: LIBERO envs with any registered robot
  (`twinrobo.libero.robots`), and `replay` helpers for exact demo replay.

Setup. LIBERO is not pip-installable as a package, since its ``libero/`` has
no ``__init__.py``. Point ``LIBERO_ROOT`` at a LIBERO checkout. Also set
``LIBERO_CONFIG_PATH`` to a directory holding LIBERO's ``config.yaml``:
without it, LIBERO prompts on stdin at first import. `setup_libero` checks
both and makes ``import libero`` work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..exceptions import SimulatorError

LIBERO_ROOT_ENV = "LIBERO_ROOT"
LIBERO_CONFIG_ENV = "LIBERO_CONFIG_PATH"


def setup_libero(
    libero_root: str | Path | None = None, config_path: str | Path | None = None
) -> None:
    """Make ``import libero`` work without LIBERO's interactive first-run prompt.

    Args:
        libero_root: LIBERO checkout (the directory containing ``libero/libero``).
            Defaults to ``$LIBERO_ROOT``. Unneeded if ``libero`` is importable.
        config_path: Directory with LIBERO's ``config.yaml``. Defaults to
            ``$LIBERO_CONFIG_PATH``.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

    if config_path is not None:
        os.environ[LIBERO_CONFIG_ENV] = str(config_path)
    cfg = os.environ.get(LIBERO_CONFIG_ENV)
    if not cfg or not (Path(cfg) / "config.yaml").is_file():
        raise SimulatorError(
            f"Set {LIBERO_CONFIG_ENV} to a directory containing LIBERO's config.yaml "
            "(otherwise LIBERO prompts on stdin at import)."
        )

    try:
        import libero.libero  # noqa: F401

        return
    except ImportError:
        pass
    root = Path(libero_root or os.environ.get(LIBERO_ROOT_ENV, ""))
    pkg = root / "libero"
    if not (pkg / "libero").is_dir():
        raise SimulatorError(
            f"LIBERO not importable. Set {LIBERO_ROOT_ENV} to a LIBERO checkout "
            f"(expected {pkg / 'libero'})."
        )
    # Expose only the `libero` namespace package, not every top-level directory of the checkout
    # (its scripts/ etc. would shadow other modules): a private dir holding one symlink.
    shim = Path.home() / ".cache" / "twinrobo" / "pypath"
    shim.mkdir(parents=True, exist_ok=True)
    link = shim / "libero"
    if link.is_symlink() and link.resolve() != pkg.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(pkg.resolve(), target_is_directory=True)
    if str(shim) not in sys.path:
        sys.path.insert(0, str(shim))
    import libero.libero  # noqa: F401


def load_init_states(task_suite, task_id: int) -> np.ndarray:
    """A task's init states, loaded **without** unpickling arbitrary code.

    LIBERO's loader calls ``torch.load`` with the pre-2.6 default
    (``weights_only=False``). Recent torch refuses that. Here only the numpy
    array reconstruction globals these files contain are allowlisted.
    """
    import torch
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    allow = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        *{
            type(np.dtype(t))
            for t in (np.float64, np.float32, np.int64, np.int32, np.uint8, np.bool_)
        },
    ]
    with torch.serialization.safe_globals(allow):
        return np.asarray(torch.load(path))


def make_env(suite: str, task_id: int, resolution: int = 128, **env_kwargs):
    """``(env, task, init_states)`` for a benchmark task, as LIBERO's eval scripts build it."""
    setup_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution, **env_kwargs
    )
    return env, task, load_init_states(task_suite, task_id)


def make_env_from_bddl(
    bddl_file_name: str, resolution: int = 128, robot: str = "Panda", **env_kwargs
):
    """A LIBERO env for a bddl path as stored in LIBERO datasets (``.../<suite>/<task>.bddl``)."""
    setup_libero()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    rel = Path(str(bddl_file_name))
    bddl = os.path.join(get_libero_path("bddl_files"), rel.parent.name, rel.name)
    if robot != "Panda":
        from .robots import register_robots

        register_robots()
        env_kwargs["robots"] = [robot]
    return OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution, **env_kwargs
    )


from .env import CameraTwinLiberoEnv  # noqa: E402

__all__ = [
    "setup_libero",
    "load_init_states",
    "make_env",
    "make_env_from_bddl",
    "CameraTwinLiberoEnv",
]
