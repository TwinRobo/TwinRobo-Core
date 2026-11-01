"""GPU memory of a ray-cast Isaac camera over many refocuses.

    docker/isaac/run.sh tools/validate/refocus_memory.py [--refocuses 100]

Only the public API: `twin.reference.set_focus(f)`, then `get_frame()`, sweeping focus between
--far and --near and back, as a robot approaching and leaving objects does. Writes one JSON
line per refocus to /workspace/outputs/refocus_memory.jsonl and prints a summary; exits 1 if
free GPU memory falls more than --limit-gib between a third of the way through and the end.
Other jobs on the GPU move these numbers: the process list is logged at both ends.
"""

import argparse
import json
import subprocess
import sys
import time

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default="raspberrypi/camera-module-3/standard")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--rays", type=int, default=32)
ap.add_argument("--views", type=int, default=7)
ap.add_argument("--spp", type=int, default=8)
ap.add_argument("--refocuses", type=int, default=100)
ap.add_argument("--near", type=float, default=0.10)
ap.add_argument("--far", type=float, default=1.0)
ap.add_argument("--limit-gib", type=float, default=0.5, help="allowed fall after the warm-up")
ap.add_argument("--out", default="/workspace/outputs/refocus_memory.jsonl")
args = ap.parse_args()
if args.refocuses < 40:
    ap.error("--refocuses must be at least 40: the check compares two separate stretches")
app = SimulationApp(
    {
        "headless": True,
        "extra_args": ["--/rtx-transient/resourcemanager/enableTextureStreaming=0"],
    }
)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from twinrobo import CameraTwin  # noqa: E402
from twinrobo.plugins.isaac import IsaacCameraTwin  # noqa: E402
from twinrobo.registry import CatalogRegistry  # noqa: E402


def gpu_jobs():
    q = "--query-compute-apps=pid,used_memory,name"
    r = subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()


def scaled_spec():
    spec = CatalogRegistry().load(args.camera)
    r = spec.sensor.resolution
    k = args.width / r.width
    r.width, r.height = args.width, int(round(r.height * k / 2) * 2)
    if spec.sensor.pixel_pitch_um:
        spec.sensor.pixel_pitch_um /= k
    c = spec.calibration["intrinsic"]
    c.update(
        fx=c["fx"] * k, fy=c["fy"] * k, cx=(c["cx"] + 0.5) * k - 0.5, cy=(c["cy"] + 0.5) * k - 0.5
    )
    return spec


s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing")
s.set("/rtx/pathtracing/spp", 1)
s.set("/rtx/pathtracing/totalSpp", args.spp)
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Camera.Define(stage, "/World/Cam")
for i, z in enumerate((-0.15, -0.4, -1.5)):
    c = UsdGeom.Cube.Define(stage, f"/World/Box{i}")
    c.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(0.1 * i - 0.1, 0, z))
    x.AddScaleOp().Set(Gf.Vec3f(0.1 * (i + 1), 0.1 * (i + 1), 0.02))
UsdLux.DomeLight.Define(stage, "/World/Dome").GetIntensityAttr().Set(1000)
for _ in range(3):
    app.update()

twin = CameraTwin.from_spec(scaled_spec(), device="cuda", build_psf=False)
cam = IsaacCameraTwin(
    twin,
    "/World/Cam",
    render="raycast",
    rays_per_pixel=args.rays,
    pupil_views=args.views,
    rt_subframes=args.spp,
)
for _ in range(3):
    cam.get_frame()
print("other GPU jobs, start:", gpu_jobs(), flush=True)
n = args.refocuses
sweep = np.linspace(0, 1, n - n // 2)
sweep = np.concatenate([sweep, sweep[::-1][: n // 2]])
diopters = 1 / args.far + (1 / args.near - 1 / args.far) * sweep
rows = []
with open(args.out, "w") as log:
    for i, d in enumerate(diopters):
        scene = cam._lens_scene
        t = time.perf_counter()
        twin.reference.set_focus(float(1 / d))
        cam.get_frame()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        row = {
            "i": i,
            "focus_m": round(float(1 / d), 4),
            "views": "kept" if cam._lens_scene is scene else "recreated",
            "free_gib": round(free / 2**30, 3),
            "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
            "torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "s": round(time.perf_counter() - t, 2),
        }
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        if i % 10 == 0 or i == n - 1:
            print(row, flush=True)
print("other GPU jobs, end:", gpu_jobs(), flush=True)
cam.close()


# The render products and ray cache take their memory early on; after that it must stay flat
early = np.median([r["free_gib"] for r in rows[n // 3 : n // 3 + 10]])
late = np.median([r["free_gib"] for r in rows[-10:]])
fell = early - late
print(f"fell {fell:.2f} GiB from a third of the way to the end", flush=True)
print("views:", {v: sum(r["views"] == v for r in rows) for v in ("kept", "recreated")}, flush=True)
ok = fell <= args.limit_gib
print("PASS" if ok else "FAIL", flush=True)
app.close()
sys.exit(0 if ok else 1)
