# TwinRobo

**Physically grounded digital twins of real cameras for robot learning in simulation.**

[![CI](https://github.com/TwinRobo/TwinRobo-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/TwinRobo/TwinRobo-Core/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Commercial license](https://img.shields.io/badge/commercial-license-indigo)](COMMERCIAL.md)
[![Camera catalog: Apache-2.0](https://img.shields.io/badge/camera%20catalog-Apache--2.0-green)](twinrobo/catalog/LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)
[![Docs](https://img.shields.io/badge/docs-API%20reference-indigo)](https://twinrobo.github.io/TwinRobo-Core/)

Simulators render through a perfect pinhole. Robots see through real cameras:
lenses that blur, distort and vignette, with fixed focus and a specific field
of view. Policies trained on pinhole images meet a different image at
deployment. TwinRobo closes that gap: it renders simulated scenes through the
actual optics of the camera you will deploy, as a drop-in for the simulators
robot learning already uses.

**One environment, any real camera.** TwinRobo is an open environment for aligning
perception with off-the-shelf cameras: pick a camera from the
[catalog](https://twinrobo.github.io/TwinRobo-Core/catalog/), and every simulator, rendering method and dataset
replay in TwinRobo sees through it. The catalog grows with the community:
[add a camera](#add-your-camera) and it works everywhere at once; camera makers
can [become partners](#for-camera-makers-become-a-partner) and ship verified
models of their products.

![An Isaac Sim tabletop: the simulator's pinhole (left) and the Stereolabs ZED X 2.2 mm, every pixel ray-traced through its lens (right)](docs/images/isaac-raycast.jpg)

## At a glance

| | MuJoCo (incl. LIBERO, RoboCasa) | NVIDIA Isaac Sim 6.x |
|---|---|---|
| Adapter | `twinrobo.mujoco.MujocoCameraTwin` | `twinrobo.isaac.IsaacCameraTwin` |
| `psf`: pinhole + lens blur and distortion | ✅ | ✅ |
| `pupil`: lens rays across the aperture | ✅ | ✅ |
| `raycast`: lens rays cast into the scene | ✅ | ✅ |
| RGB + metric depth, on the GPU | ✅ | ✅ |
| Cameras on robot links, stereo modules | ✅ | any USD camera prim |
| Setup | `pip install` | Docker recipe included |

**Cameras:** 13 in the catalog, including Intel RealSense D435 / D455,
Stereolabs ZED X, ZED X Mini and ZED 2i, Luxonis OAK-D, Logitech C920 and the
Raspberry Pi Camera Module 3, as single cameras or stereo pairs with rectified
output. Every camera works in every simulator and rendering method.
[Add yours](#add-your-camera) or [calibrate one](docs/calibration.md).

## Features

- **Real lens optics.** Lenses are traced with
  [DeepLens](https://github.com/vccimaging/DeepLens): depth- and field-dependent
  blur, the lens' own distortion, chromatic aberration and vignetting, from a
  lens prescription instead of a hand-tuned filter.
- **Three rendering methods, per camera.** A fast 2.5D PSF renderer, and two
  lens-ray renderers that trace every pixel's rays through the lens into the
  scene: *pupil raster* and GPU *ray cast* (NVIDIA Warp), exact even for
  defocused foreground objects. [More](docs/rendering.md)
- **Drop-in for robot simulators.** MuJoCo, LIBERO (robosuite 1.4), RoboCasa
  kitchens (robosuite 1.5) and NVIDIA Isaac Sim (any USD stage, RTX rendering).
  `CameraTwinLiberoEnv` swaps a camera's observations in place, so policies and
  eval scripts run unchanged; recorded demos replay exactly, so any camera can
  be placed in an existing episode. [More](docs/simulators.md)
- **Camera catalog and stereo.** The cameras robots use (RealSense, ZED, OAK-D,
  webcams, Raspberry Pi), stereo modules with rectified output and adjustable
  baselines, and cameras mountable on any robot link. Adding a camera is a YAML
  file and a lens file. [More](docs/cameras.md)
- **Calibrate a real unit.** `python -m twinrobo.calibration` turns ChArUco,
  slanted-edge and flat-field captures into a `measured` entry: intrinsics,
  distortion, focus, vignetting and stereo baseline. [More](docs/calibration.md)
- **Validated, in both simulators.** The PSF renderer matches DeepLens' reference
  renderer at about 46 dB. Points land within a pixel of where the lens' chief
  rays point (within 0.1 px in Isaac). On a defocused foreground object, ray cast
  reproduces the exact partially covered edge to within ~0.03 (Isaac) and ~0.05
  (MuJoCo), where pinhole + PSF errs by 0.13–0.4.

## Installation

TwinRobo requires **Python 3.12** (inherited from DeepLens) and a CUDA GPU
for practical speeds.

```bash
git clone https://github.com/TwinRobo/TwinRobo-Core.git && cd TwinRobo-Core
conda create -n twinrobo python=3.12 -y && conda activate twinrobo
pip install -e ".[dev]"          # core + DeepLens
pip install -e ".[libero]"       # + MuJoCo, NVIDIA Warp and LIBERO support
```

LIBERO itself is used from a checkout (`LIBERO_ROOT`). RoboCasa needs
robosuite 1.5 and gets its own environment; see
[simulators](docs/simulators.md#robocasa). **Isaac Sim** runs in NVIDIA's
container, with TwinRobo installed on Isaac's own Python and torch:

```bash
docker pull nvcr.io/nvidia/isaac-sim:6.1.0        # needs the NVIDIA Container Toolkit
docker/isaac/run.sh examples/01_isaac_camera.py   # builds the image on first use
```

## Quick start: mount a real camera on a robot

Pick a camera from the catalog, mount it on a robot link, and render what it
would record. This runs in robosuite's PickPlace scene (`pip install -e ".[libero]"`
brings robosuite; no LIBERO checkout needed):

```python
import robosuite

from twinrobo import CameraSpec, CameraTwin, CatalogRegistry
from twinrobo.mujoco import MujocoCameraTwin, RobosuiteRenderer, to_uint8
from twinrobo.mujoco.mounts import CameraMount, mounted

env = robosuite.make(
    "PickPlace",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
)
env.reset()

# 1. Pick a real camera from the catalog (here read out at 960x600).
path = CatalogRegistry().resolve("stereolabs/zed-x/2.2mm")
twin = CameraTwin.from_spec(CameraSpec.from_yaml(path, width=960, height=600))

# 2. Mount it on a robot link: position (m) and yaw/pitch/roll (deg) in the link's frame.
wrist = CameraMount("wrist", body="robot0_right_hand", pos=(0.08, 0, 0), rpy_deg=(67, -48, 173))

# 3. Render what that camera would record.
with mounted(env.sim.model._model, env.sim.data._data, wrist) as host:
    frame = MujocoCameraTwin(twin, RobosuiteRenderer(env), camera=host).get_frame()

image = to_uint8(frame.rgb)  # the real camera's image, uint8 [H, W, 3]
pinhole = to_uint8(frame.rgb_ideal)  # the simulator's ideal pinhole, same camera
depth = frame.depth[0, 0]  # metric z-depth (m), aligned to the image
```

The first call builds the lens's PSF bank with DeepLens (about 20 s on a GPU)
and caches it; after that a frame takes a fraction of a second. The mount moves
with the link, so it follows the robot as it acts or replays a demo.

[`examples/05_mount_camera_on_robot.py`](examples/05_mount_camera_on_robot.py)
does this for three catalog cameras on the same wrist mount and draws what each
one gives you:

```bash
MUJOCO_GL=egl python examples/05_mount_camera_on_robot.py   # writes outputs/05_mount_camera.jpg
```

![Three catalog cameras on the same wrist mount: pinhole vs TwinRobo image, close-up, difference and depth](docs/images/quickstart_cameras.jpg)

Read it top to bottom:

- **Simulator pinhole / TwinRobo camera:** the field of view is each lens's
  own. The ZED X 2.2 mm's wide lens shows its barrel distortion, which a
  pinhole cannot.
- **Close-up:** the same patch at native resolution; the real lens softens the
  lettering, as the camera would.
- **Difference ×4:** where the optics change the image: distortion shifts
  edges for the 2.2 mm lens; blur changes edges and texture for the others.
- **Depth:** metric z-depth, aligned pixel for pixel with the camera's image.

### Next steps

- **Other rendering methods:** pass `render="pupil"` or `render="raycast"` to
  `MujocoCameraTwin` to trace real lens rays into the scene (see
  [Rendering methods](#rendering-methods)).
- **Isaac Sim:** the same twin on any USD camera prim:

  ```python
  from twinrobo.isaac import IsaacCameraTwin

  twin = CameraTwin.from_catalog("stereolabs/zed-x/2.2mm", build_psf=False)
  cam = IsaacCameraTwin(twin, "/World/Camera", render="raycast")
  frame = cam.get_frame()  # frame.rgb, frame.depth: the ZED X's view, on the GPU
  ```

  `docker/isaac/run.sh examples/01_isaac_camera.py --render raycast` renders the
  image at the top of this page. [More](docs/simulators.md#isaac-sim)
- **A LIBERO policy's camera:** wrap an env so its observations come from the
  twin, and eval scripts run unchanged:

  ```python
  from twinrobo.libero import CameraTwinLiberoEnv, make_env

  env, task, init_states = make_env("libero_spatial", 0, resolution=128)
  env = CameraTwinLiberoEnv(env, {"agentview": twin})
  obs = env.reset()  # obs["agentview_image"] now comes from the twin
  ```

- **An RGB-D frame from any source:** `twin.process(rgb, depth)` takes linear
  RGB `[B,3,H,W]` and metric depth `[B,1,H,W]` on the GPU.
- **Your own camera:** copy a catalog spec, or add one for everyone; see
  [Cameras](docs/cameras.md) and [CONTRIBUTING](CONTRIBUTING.md#adding-a-camera).

## Rendering methods

| Method | How | Per frame, 1920×1200 (RTX 3090) | Best for |
|---|---|---|---|
| `psf` | pinhole render + depth- and field-dependent PSF blur | ~0.1–0.4 s | fast training data |
| `pupil` | lens rays looked up in views rendered across the aperture | ~0.5 s | real distortion, chromatic aberration, vignetting |
| `raycast` | lens rays intersected with the scene on the GPU | ~0.9 s (MuJoCo), ~1.5 s (Isaac) | exact defocus around occluders |

The methods are simulator independent (`twinrobo.optics.lensrender`): a
simulator supplies the camera pose, views rendered from points on the lens'
aperture, and the scene's triangles.

[Details and validation](docs/rendering.md)

## Supported simulators

| Simulator | Scenes and data | Rendering methods |
|---|---|---|
| MuJoCo | any MJCF scene | `psf`, `pupil`, `raycast` |
| LIBERO | LIBERO tasks and demos; Panda, Sawyer, UR5e, iiwa, Jaco, Gen3, multi-camera variants | `psf`, `pupil`, `raycast` |
| RoboCasa | procedurally generated kitchens, LeRobot demo datasets, PandaOmron | `psf`, `pupil`, `raycast` |
| NVIDIA Isaac Sim 6.x | any USD stage, RTX rendering (Docker recipe included) | `psf`, `pupil`, `raycast` |

## Camera catalog

| Vendor | Cameras | Stereo modules (baseline) |
|---|---|---|
| Intel RealSense | D435 / D435i color and depth imager, D455 color and depth imager | D435 (50 mm), D455 (95 mm) |
| Stereolabs | ZED X 2.2 mm and 4 mm eyes, ZED 2i 2.1 mm eye | ZED X (120 mm), ZED X Mini (50 mm), ZED 2i (120 mm) |
| Luxonis | OAK-D color and mono | OAK-D (75 mm) |
| Logitech | C920 HD Pro Webcam | |
| Raspberry Pi | Camera Module 3, Camera Module 3 Wide | |
| Example | `examples/cellphone80deg` (a DeepLens design lens, teaching example) | |

All entries are `estimated` today: geometry fitted to the datasheet fields of
view, blur from a surrogate lens of matching field, every number sourced in the
entry's header. The [catalog page](https://twinrobo.github.io/TwinRobo-Core/catalog/) lists each with its sensor,
lens and field of view, generated from the catalog files.

> **The camera catalog is Apache-2.0.** Everything in
> [`twinrobo/catalog/`](twinrobo/catalog) (camera specs, stereo modules and lens
> files) is licensed under the [Apache License 2.0](twinrobo/catalog/LICENSE),
> separately from the AGPL-3.0 code: use the camera data in any project,
> commercial or not, with or without TwinRobo.

## Add your camera

**The catalog is the heart of TwinRobo, and it is open.** Every camera added
becomes available to everyone, in every simulator, rendering method and dataset
replay, with no simulator-specific work. If your robot uses a camera that is
not here, it is the most valuable contribution you can make:

1. Create `twinrobo/catalog/<vendor>/<model>/<lens>/` with a `camera.yaml`
   (sensor, lens, focus, calibration; copy an existing entry) and a DeepLens
   `lens.json`. No prescription published? Build a surrogate of the same field
   of view and f-number with `tools/catalog/make_surrogate_lens.py`.
2. Run `pytest tests/test_catalog.py` and render it once:
   `CameraTwin.from_catalog("<vendor>/<model>/<lens>")`.
3. Open a pull request, with where each number came from. Catalog
   contributions are published under Apache-2.0, like the rest of the catalog.

Calibrated a real unit? Upgrading an entry from `estimated` to `measured` is
just as welcome. The full guide is in
[CONTRIBUTING](CONTRIBUTING.md#adding-a-camera); to ask for a camera instead,
[open a camera request](../../issues/new?template=new_camera.md).

## For camera makers: become a partner

**Put your camera in every robot-learning simulator.** Robot teams now choose
and validate cameras in simulation before they buy hardware. The TwinRobo
Camera Partner program lets camera and lens makers ship
`manufacturer_verified` models of their products, and lets companies sponsor
the open environment their customers use:

- **You contribute** any of: optical data (a lens prescription or measured
  PSFs), factory calibration, sample units, or sponsorship of compute and
  maintenance. Published entries are Apache-2.0, free for your customers to use
  anywhere; data you cannot publish can stay private.
- **You get** a verified catalog entry that works in MuJoCo, Isaac Sim and
  every rendering method, your company listed as a partner here and on the
  docs site, a showcase example with your camera, and early integration of
  new products.

[Read about the program](docs/partners.md) or
[open a partner inquiry](../../issues/new?template=camera_partner.md).

## Documentation

The documentation site, with the API reference, is at
**[twinrobo.github.io/TwinRobo-Core](https://twinrobo.github.io/TwinRobo-Core/)**. Build it locally with
`pip install -e ".[docs]" && mkdocs serve`.

| | |
|---|---|
| [Optics and rendering](docs/rendering.md) | PSF pipeline, distortion, the three rendering methods, validation |
| [Cameras and stereo](docs/cameras.md) | CameraSpec, catalog, stereo modules, spec overrides |
| [Simulators](docs/simulators.md) | MuJoCo, LIBERO, RoboCasa and Isaac Sim: setup, behavior, limits |
| [API reference](https://twinrobo.github.io/TwinRobo-Core/api/) | every public class and function, from the docstrings |
| [Development](docs/development.md) | layout, tests, status |

## Contributing

Beyond cameras, also wanted: new simulator adapters (a lens scene is three
methods; see `twinrobo.optics.lensrender`), sensor noise and ISP models, and
documentation. See [CONTRIBUTING](CONTRIBUTING.md).

## Roadmap

- Faster lens-ray rendering in Isaac Sim (fewer, shared pupil views)
- Sensor noise and ISP models (interfaces are in place)
- `measured` catalog entries from calibration captures of real units
- More cameras and lenses in the catalog

## Acknowledgements

TwinRobo builds on [DeepLens](https://github.com/vccimaging/DeepLens) for
lens modeling, [MuJoCo](https://mujoco.org),
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim),
[robosuite](https://robosuite.ai), [LIBERO](https://libero-project.github.io),
[RoboCasa](https://robocasa.ai) and [NVIDIA Warp](https://github.com/NVIDIA/warp).
Third-party files and their licenses are listed in [`NOTICE`](NOTICE).

## License

TwinRobo is dual-licensed: the **GNU AGPL-3.0** ([`LICENSE`](LICENSE), with an
additional permission for NVIDIA Isaac Sim / Omniverse in
[`LICENSE-EXCEPTION.md`](LICENSE-EXCEPTION.md)), free for research, education and
open-source use, or a **commercial license** for proprietary products and hosted
services ([`COMMERCIAL.md`](COMMERCIAL.md)). The camera catalog
(`twinrobo/catalog/`) is Apache-2.0, free to use anywhere. Third-party material is
listed in [`NOTICE`](NOTICE).
