"""Render the playground's MuJoCo scene through every catalog camera and method.

    MUJOCO_GL=egl python tools/docs/playground/render_mujoco.py --out outputs/playground

robosuite's PickPlace (a Panda at a bin of groceries), seen from the Panda's wrist
camera (``robot0_eye_in_hand``); each camera renders its own field of view from that pose.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from common import METHODS, camera_meta, cameras, save_frame, spec_for, write_meta  # noqa: E402

SCENE = {
    "id": "mujoco-pickplace",
    "label": "MuJoCo · robosuite PickPlace",
    "simulator": "MuJoCo",
    "pose": "the Panda's wrist camera (robot0_eye_in_hand)",
    "note": "Looking down from a Franka Panda's wrist into a bin of groceries "
    "(robosuite PickPlace, seed 0).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/playground")
    ap.add_argument("--camera", action="append", help="only these catalog ids")
    args = ap.parse_args()
    import robosuite
    import torch

    from twinrobo import CameraTwin
    from twinrobo.plugins.mujoco import MujocoCameraTwin, RobosuiteRenderer

    np.random.seed(0)
    env = robosuite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    env.reset()
    backend = RobosuiteRenderer(env)
    scene_dir = Path(args.out) / SCENE["id"]
    metas = []
    for cid in args.camera or cameras():
        spec, full = spec_for(cid)
        twin = CameraTwin.from_spec(spec, device="cuda")
        for method in METHODS:
            t = time.perf_counter()
            cam = MujocoCameraTwin(twin, backend, camera="robot0_eye_in_hand", render=method)
            frame = cam.get_frame()
            torch.cuda.synchronize()
            save_frame(scene_dir / cid, method, frame, pinhole=method == "psf")
            print(f"{cid:40s} {method:8s} {time.perf_counter() - t:5.1f} s", flush=True)
        metas.append(camera_meta(cid, full, twin))
    write_meta(scene_dir, SCENE, metas)
    env.close()


if __name__ == "__main__":
    main()
