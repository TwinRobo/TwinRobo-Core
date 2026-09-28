"""LIBERO with a CameraTwin agentview camera.

Runs a LIBERO task with the agentview camera replaced by the example CameraTwin.
It saves ideal-vs-CameraTwin frames and prints how much the optics change the
policy-resolution image.

    export LIBERO_ROOT=/path/to/LIBERO LIBERO_CONFIG_PATH=/dir/with/libero/config.yaml
    python examples/04_libero_twinrobo.py [--suite libero_spatial --task 0 --resolution 128]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from twinrobo import CameraTwin
from twinrobo.datasets.libero import CameraTwinLiberoEnv, make_env
from twinrobo.registry import BUILTIN_CATALOG

ROOT = Path(__file__).resolve().parents[1]
SPEC = BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml"
OUT = ROOT / "outputs/04_libero"


def psnr(a, b):
    return 10 * np.log10(255.0**2 / np.mean((a.astype(float) - b.astype(float)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    twin = CameraTwin.from_spec(SPEC)
    env, task, init_states = make_env(args.suite, args.task, resolution=args.resolution)
    env = CameraTwinLiberoEnv(env, {"agentview": twin}, keep_ideal=True)
    print(f"task: {task.language}")
    print(f"camera: {twin.spec.id}, vFoV {twin.intrinsics.vfov_deg:.1f} deg (LIBERO: 45 deg)")

    env.seed(0)
    env.reset()
    obs = env.set_init_state(init_states[0])
    OUT.mkdir(parents=True, exist_ok=True)
    times = []
    for _ in range(args.steps):
        action = [0.0] * 6 + [-1.0]  # hold still, gripper open
        t0 = time.perf_counter()
        obs, _, _, _ = env.step(action)
        times.append(time.perf_counter() - t0)
    ideal, twin_img = obs["agentview_image_ideal"][::-1], obs["agentview_image"][::-1]  # upright
    Image.fromarray(np.concatenate([ideal, twin_img], 1)).save(OUT / "agentview_ideal_vs_twin.png")
    print(f"step time with CameraTwin: {1000 * np.median(times):.0f} ms (median of {args.steps})")
    print(
        f"LIBERO camera vs CameraTwin camera at {args.resolution}px: {psnr(ideal, twin_img):.1f} dB"
    )
    print(f"wrote {OUT}")
    env.close()


if __name__ == "__main__":
    main()
