"""RoboCasa replay (needs robosuite 1.5, the RoboCasa asset pack and a demo dataset)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
robosuite = pytest.importorskip("robosuite")
pytest.importorskip("robocasa")
if not robosuite.__version__.startswith("1.5"):
    pytest.skip("RoboCasa needs robosuite 1.5", allow_module_level=True)
DEMOS = Path(os.environ.get("ROBOCASA_DEMOS", "datasets/robocasa"))
if not DEMOS.exists():
    pytest.skip("no RoboCasa demo datasets (set ROBOCASA_DEMOS)", allow_module_level=True)

pytestmark = [pytest.mark.robocasa, pytest.mark.gl]


def _dataset():
    for info in sorted(DEMOS.rglob("meta/info.json")):
        ds = info.parent.parent
        if (ds / "extras" / "episode_000000" / "states.npz").is_file():
            return ds
    pytest.skip("no LeRobot dataset with extras/episode_000000")


def _video(path, w, h):
    ffmpeg = shutil.which("ffmpeg") or pytest.skip("needs ffmpeg")
    cmd = [
        ffmpeg,
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def test_episode_replays_its_recorded_video():
    """An episode's own kitchen + recorded states reproduce its recorded camera video."""
    from twinrobo.datasets.robocasa import (
        env_args,
        make_env,
        read_episode,
        reset_to_episode,
        setup_robocasa,
    )

    setup_robocasa()
    ds = _dataset()
    info = json.loads((ds / "meta" / "info.json").read_text())
    key = "observation.images.robot0_agentview_left"
    h, w = info["features"][key]["shape"][:2]
    rel = info["video_path"].format(episode_chunk=0, video_key=key, episode_index=0)
    rec = _video(ds / rel, w, h)
    env = make_env(env_args(ds))
    try:
        states, xml, ep_meta = read_episode(ds, "episode_000000")
        reset_to_episode(env, xml, ep_meta)
        for t in (0, len(rec) // 2):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            img = env.sim.render(camera_name="robot0_agentview_left", width=w, height=h)[::-1]
            mse = np.mean((img.astype(float) - rec[t]) ** 2)
            assert 10 * np.log10(255**2 / mse) > 30  # the recording is h264
    finally:
        env.close()
