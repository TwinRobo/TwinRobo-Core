# TwinRobo

**Physically grounded digital twins of real cameras for robot learning in simulation.**

[![CI](https://github.com/TwinRobo/TwinRobo-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/TwinRobo/TwinRobo-Core/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)

Simulators render through a perfect pinhole. Robots see through real cameras:
lenses that blur, distort and vignette, with fixed focus and a specific field
of view. Policies trained on pinhole images meet a different image at
deployment. TwinRobo closes that gap: it renders simulated scenes through the
actual optics of the camera you will deploy, as a drop-in for the simulators
robot learning already uses.

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

**Cameras:** Stereolabs ZED X (2.2 mm, 4 mm) and ZED X Mini, as single eyes or
stereo pairs with rectified output, plus a teaching example lens. Every camera
works in every simulator and rendering method. [Add yours](CONTRIBUTING.md#adding-a-camera).

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
- **Camera catalog and stereo.** Real products such as the Stereolabs ZED X
  family, stereo modules with rectified output, and cameras mountable on any
  robot link. Adding a camera is a YAML file and a lens file.
  [More](docs/cameras.md)
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

| ID | Camera | Status |
|---|---|---|
| `stereolabs/zed-x/2.2mm` | ZED X / ZED X Mini eye, 2.2 mm f/2.2, 1920×1200, global shutter | estimated |
| `stereolabs/zed-x/4mm` | ZED X eye, 4 mm f/2.2 | estimated |
| `examples/cellphone80deg` | a DeepLens design lens on a hypothetical sensor (teaching example) | estimated |

Stereo modules: ZED X (2.2 mm, 4 mm; 120 mm baseline) and ZED X Mini (2.2 mm;
50 mm baseline). **Your camera is missing?**
[Add it](CONTRIBUTING.md#adding-a-camera) or
[request it](../../issues/new?template=new_camera.md).

## Documentation

| | |
|---|---|
| [Optics and rendering](docs/rendering.md) | PSF pipeline, distortion, the three rendering methods, validation |
| [Cameras and stereo](docs/cameras.md) | CameraSpec, catalog, stereo modules, spec overrides |
| [Simulators](docs/simulators.md) | MuJoCo, LIBERO, RoboCasa and Isaac Sim: setup, behavior, limits |
| [Development](docs/development.md) | layout, tests, status |

## Contributing

Contributions are welcome, and **new cameras** are the most valuable: a
catalog entry is a `camera.yaml` plus a lens file, checked by
`tests/test_catalog.py`. Also wanted: `measured` calibrations of real units,
new simulator adapters (a lens scene is three methods; see
`twinrobo.optics.lensrender`), and sensor noise and ISP models. See
[CONTRIBUTING](CONTRIBUTING.md).

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

Apache License 2.0. See [`LICENSE`](LICENSE).
