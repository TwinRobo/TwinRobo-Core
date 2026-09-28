"""Render the playground's Isaac Sim scene through every catalog camera and method.

    docker/isaac/run.sh tools/docs/playground/render_isaac.py   # -> outputs/isaac/playground

A tabletop generated from a fixed seed: random primitives (boxes, spheres,
cylinders, cones) on a checkered table top that makes the lenses' distortion easy to
see, seen from where a robot's wrist camera would be: about 30 cm above, looking down.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})  # before any omni / pxr import

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from common import METHODS, camera_meta, cameras, save_frame, spec_for, write_meta  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

SEED = 7
SCENE = {
    "id": "isaac-tabletop",
    "label": "Isaac Sim · random tabletop",
    "simulator": "Isaac Sim 6.1",
    "pose": "a wrist camera about 30 cm over the table",
    "note": f"Random primitives on a checkered table (seed {SEED}) under a wrist camera, "
    "RTX renderer.",
}


def scene(stage) -> str:
    rng = np.random.default_rng(SEED)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def place(prim, pos, scale, color, rot=0.0):
        prim.GetDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        xf.AddRotateZOp().Set(float(rot))
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))

    def box(path, pos, size, color, rot=0.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        place(cube, pos, size, color, rot)

    box("/World/Floor", (0, 0, -0.01), (8, 8, 0.02), (0.3, 0.3, 0.32))
    box("/World/Table", (0.6, 0, 0.35), (1.0, 1.4, 0.04), (0.5, 0.36, 0.22))
    # checkered table top: straight lines show the lens' distortion
    for i in range(10):
        for j in range(14):
            c = (0.85, 0.85, 0.82) if (i + j) % 2 else (0.2, 0.22, 0.26)
            pos = (0.15 + 0.1 * i, -0.65 + 0.1 * j, 0.375)
            box(f"/World/Top/T{i}_{j}", pos, (0.1, 0.1, 0.01), c)
    for k, (x, y) in enumerate([(0.2, -0.6), (1.0, -0.6), (0.2, 0.6), (1.0, 0.6)]):
        box(f"/World/Leg{k}", (x, y, 0.17), (0.04, 0.04, 0.34), (0.2, 0.2, 0.2))
    shapes = [UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder, UsdGeom.Cone]
    for i in range(10):
        kind = shapes[rng.integers(len(shapes))]
        s = rng.uniform(0.04, 0.09)
        h = s * rng.uniform(0.8, 2.0)
        x, y = rng.uniform(0.4, 0.85), rng.uniform(-0.28, 0.28)
        color = rng.uniform(0.1, 0.9, 3)
        prim = kind.Define(stage, f"/World/Obj{i}")
        if kind is UsdGeom.Cube:
            prim.GetSizeAttr().Set(1.0)
            place(prim, (x, y, 0.38 + h / 2), (s, s, h), color, rng.uniform(0, 90))
        elif kind is UsdGeom.Sphere:
            prim.GetRadiusAttr().Set(1.0)
            place(prim, (x, y, 0.38 + s / 2), (s / 2, s / 2, s / 2), color)
        else:
            prim.GetRadiusAttr().Set(0.5)
            prim.GetHeightAttr().Set(1.0)
            place(prim, (x, y, 0.38 + h / 2), (s, s, h), color)
    UsdLux.DomeLight.Define(stage, "/World/Sky").GetIntensityAttr().Set(250)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.GetIntensityAttr().Set(600)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35, 20, 30))
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(0.42, 0.0, 0.72), Gf.Vec3d(0.64, 0.0, 0.38), Gf.Vec3d(0, 0, 1)
    )
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())  # camera-to-world
    return "/World/Camera"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/outputs/playground")
    ap.add_argument("--camera", action="append", help="only these catalog ids")
    args = ap.parse_args()
    import torch

    from twinrobo import CameraTwin
    from twinrobo.plugins.isaac import IsaacCameraTwin

    ctx = omni.usd.get_context()
    ctx.new_stage()
    camera = scene(ctx.get_stage())
    for _ in range(5):
        app.update()
    scene_dir = Path(args.out) / SCENE["id"]
    metas = []
    for cid in args.camera or cameras():
        spec, full = spec_for(cid)
        twin = CameraTwin.from_spec(spec, device="cuda")
        for method in METHODS:
            t = time.perf_counter()
            cam = IsaacCameraTwin(twin, camera, rt_subframes=8, render=method)
            for _ in range(3):  # let the renderer settle on the new views
                cam.get_frame()
            frame = cam.get_frame()
            torch.cuda.synchronize()
            save_frame(scene_dir / cid, method, frame, pinhole=method == "psf")
            cam.close()
            print(f"{cid:40s} {method:8s} {time.perf_counter() - t:5.1f} s", flush=True)
        metas.append(camera_meta(cid, full, twin))
    write_meta(scene_dir, SCENE, metas)
    app.close()


if __name__ == "__main__":
    main()
