"""Render an Isaac Sim camera through a real camera's optics.

A small tabletop scene is seen by a USD camera; `IsaacCameraTwin` renders it
through the Stereolabs ZED X 2.2 mm eye (wide-angle lens: distortion and blur)
and saves the simulator's pinhole view next to the TwinRobo image.

    docker/isaac/run.sh examples/01_isaac_camera.py      # -> outputs/isaac/isaac_camera.png
    docker/isaac/run.sh examples/01_isaac_camera.py --render raycast  # trace every pixel's rays
    docker/isaac/run.sh examples/01_isaac_camera.py \
        --usd /workspace/outputs/scene.usd --camera /World/Cam      # your own stage

The first run compiles Isaac's shaders and builds the camera's PSF bank (a few
minutes); both are cached, so later runs start in under a minute.
"""

import argparse
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})  # before any omni / pxr import

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from twinrobo import CameraTwin  # noqa: E402
from twinrobo.isp.color import linear_to_srgb  # noqa: E402
from twinrobo.plugins.isaac import IsaacCameraTwin  # noqa: E402


def tabletop(stage) -> str:
    """A table with objects at several distances and a camera looking down at it."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def box(path, pos, size, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))

    box("/World/Floor", (0, 0, -0.01), (8, 8, 0.02), (0.25, 0.25, 0.27))
    box("/World/Table", (0.6, 0, 0.36), (1.0, 1.4, 0.04), (0.45, 0.32, 0.2))
    colors = [(0.8, 0.1, 0.1), (0.1, 0.5, 0.8), (0.9, 0.7, 0.1), (0.2, 0.7, 0.3), (0.6, 0.3, 0.7)]
    for i, c in enumerate(colors):
        x, y = 0.25 + 0.17 * i, -0.45 + 0.22 * i
        box(f"/World/Block{i}", (x, y, 0.43), (0.08, 0.08, 0.1), c)
    UsdLux.DomeLight.Define(stage, "/World/Sky").GetIntensityAttr().Set(250)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.GetIntensityAttr().Set(600)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35, 20, 30))
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    # close over the table's near edge, as a wrist or head camera would be
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(0.0, -0.35, 0.75), Gf.Vec3d(0.6, 0.0, 0.4), Gf.Vec3d(0, 0, 1)
    )
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())  # camera-to-world
    return "/World/Camera"


def to_png(rgb) -> np.ndarray:
    srgb = linear_to_srgb(rgb[0].clamp(0, 1)).permute(1, 2, 0).cpu().numpy()
    return (srgb * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usd", help="open this stage instead of the built-in tabletop")
    ap.add_argument("--camera", default=None, help="USD camera prim (with --usd)")
    ap.add_argument("--catalog", default="stereolabs/zed-x/2.2mm", help="catalog camera id")
    ap.add_argument("--render", default="psf", choices=["psf", "raycast"])
    ap.add_argument("--out", default="/workspace/outputs/isaac_camera.png")
    args = ap.parse_args()

    ctx = omni.usd.get_context()
    if args.usd:
        ctx.open_stage(args.usd)
        camera = args.camera or "/World/Camera"
    else:
        ctx.new_stage()
        camera = tabletop(ctx.get_stage())
    for _ in range(5):
        app.update()

    twin = CameraTwin.from_catalog(args.catalog, device="cuda", build_psf=args.render == "psf")
    cam = IsaacCameraTwin(twin, camera, rt_subframes=8, render=args.render)
    print(cam)
    for _ in range(3):  # let the renderer settle on the new views
        cam.get_frame()
    t = time.perf_counter()
    frame = cam.get_frame()
    print(f"frame in {time.perf_counter() - t:.2f} s ({args.render})")
    k = twin.intrinsics
    iio.imwrite(args.out, np.concatenate([to_png(frame.rgb_ideal), to_png(frame.rgb)], axis=1))
    print(f"pinhole | {args.catalog} ({k.width}x{k.height}) -> {args.out}")
    cam.close()
    app.close()


if __name__ == "__main__":
    main()
