"""Mount real cameras on a robot's wrist and compare what each one sees.

A Panda in robosuite's PickPlace scene gets a camera on its wrist link. The same
view is rendered through several catalog cameras; for each, the figure shows
the simulator's ideal pinhole image, the TwinRobo image (lens blur, distortion),
a native-resolution close-up of both, their difference, and metric depth.

    pip install -e ".[libero]" matplotlib      # robosuite + MuJoCo; no LIBERO checkout needed
    MUJOCO_GL=egl python examples/05_mount_camera_on_robot.py [--out cameras.jpg]

The first run builds each camera's PSF bank with DeepLens (~20 s each on a GPU)
and caches it; later runs take seconds.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from twinrobo import CameraSpec, CameraTwin, CatalogRegistry  # noqa: E402
from twinrobo.plugins.mujoco import MujocoCameraTwin, RobosuiteRenderer, to_uint8  # noqa: E402
from twinrobo.plugins.mujoco.mounts import CameraMount, mounted  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {  # catalog id -> label
    "stereolabs/zed-x/2.2mm": "ZED X · 2.2 mm (110° HFoV)",
    "stereolabs/zed-x/4mm": "ZED X · 4 mm (75° HFoV)",
    "examples/cellphone80deg": "Cellphone lens (80° DFoV)",
}
# On the wrist link, 8 cm ahead of the flange, aimed at the objects in the bin
# (yaw, pitch, roll in the link's frame; see twinrobo.plugins.mujoco.mounts).
WRIST = CameraMount("wrist", body="robot0_right_hand", pos=(0.08, 0.0, 0.0), rpy_deg=(67, -48, 173))
DETAIL_BODY, DETAIL_WIDTH = "Cereal_main", 0.2  # close-up on the cereal box: 20% of the width


def render(env, camera_id: str, width: int, height: int):
    """One catalog camera on the wrist: ``(pinhole, twinrobo, depth, detail pixel)``, upright."""
    spec = CameraSpec.from_yaml(CatalogRegistry().resolve(camera_id), width=width, height=height)
    twin = CameraTwin.from_spec(spec)  # builds (or loads) the camera's PSF bank
    model, data = env.sim.model._model, env.sim.data._data
    with mounted(model, data, WRIST) as host:  # the mount renders through a borrowed camera
        frame = MujocoCameraTwin(twin, RobosuiteRenderer(env), camera=host).get_frame()
        uv = project(model, data, host, DETAIL_BODY, twin.intrinsics)
    return to_uint8(frame.rgb_ideal), to_uint8(frame.rgb), frame.depth[0, 0].cpu().numpy(), uv


def project(model, data, camera: str, body: str, k) -> tuple[float, float]:
    """Pixel of a body's origin in a pinhole camera with intrinsics ``k`` (MuJoCo looks down -z)."""
    import mujoco

    c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    p = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)]
    x, y, z = data.cam_xmat[c].reshape(3, 3).T @ (p - data.cam_xpos[c])
    return k.cx + k.fx * x / -z, k.cy - k.fy * y / -z


def crop(img: np.ndarray, uv) -> np.ndarray:
    """A close-up window around pixel ``uv``, at native resolution."""
    h, w = img.shape[:2]
    cw, ch = int(DETAIL_WIDTH * w), int(DETAIL_WIDTH * w * 0.75)
    x0 = int(np.clip(uv[0] - cw / 2, 0, w - cw))
    y0 = int(np.clip(uv[1] - ch / 2, 0, h - ch))
    return img[y0 : y0 + ch, x0 : x0 + cw]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "05_mount_camera.jpg")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import robosuite

    env = robosuite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        ignore_done=True,
    )
    np.random.seed(0)  # PickPlace places its objects at random on reset
    env.reset()

    rows = [
        "Simulator pinhole",
        "TwinRobo camera",
        "Close-up: pinhole | TwinRobo",
        "Difference ×4",
        "Depth (m)",
    ]
    fig, axes = plt.subplots(len(rows), len(CAMERAS), figsize=(4.4 * len(CAMERAS), 14.5))
    for j, (camera_id, label) in enumerate(CAMERAS.items()):
        pinhole, twin, depth, uv = render(env, camera_id, args.width, args.height)
        diff = np.clip(np.abs(twin.astype(float) - pinhole) * 4, 0, 255).astype(np.uint8)
        finite = depth[np.isfinite(depth) & (depth > 0)]
        closeup = np.concatenate([crop(pinhole, uv), crop(twin, uv)], 1)
        panels = [pinhole, twin, closeup, diff, depth]
        for i, img in enumerate(panels):
            ax = axes[i, j]
            if i == len(rows) - 1:
                im = ax.imshow(img, cmap="turbo", vmin=0.2, vmax=1.2)  # same scale for all
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=11)
        axes[0, j].set_title(label, fontsize=12)
        print(f"{camera_id}: depth {finite.min():.2f}-{finite.max():.2f} m")
    fig.suptitle("One wrist-mounted camera, three lenses (robosuite PickPlace)", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
