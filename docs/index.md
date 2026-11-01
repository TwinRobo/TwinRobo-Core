# TwinRobo API reference

The classes and functions of the `twinrobo` package. New here? The
[guides](https://twinrobo.github.io/getting-started/) explain how it all fits together.

`DeviceTwin` is the place to start. It models a whole product, however many sensors it has: one
camera, a stereo pair, a lidar, or all of them together.

```python
from twinrobo import DeviceTwin
from twinrobo.plugins.mujoco import MujocoRenderer

device = DeviceTwin.from_catalog("intel/realsense-d435")     # 1. choose it
rig = device.attach(MujocoRenderer(model, data),             # 2. put it in a simulator
                    rgb="rgb_cam", ir={"left": "ir_l", "right": "ir_r"})
obs = rig.capture()                                          # 3. capture
obs["rgb"].rgb, obs["ir/left"].rgb
rig.reset()                                                  # a new episode
```

Combine any twins into your own device:

```python
device = DeviceTwin({"front": camera_twin, "roof": lidar_twin})
```

For a single sensor, the twins inside a device work the same way on their own.

| Twin | What it is | Attach it | Read the result |
|---|---|---|---|
| [`DeviceTwin`](cameras/devices.md) | any sensors in one product | `attach(backend, rgb=..., ir=...)` | `obs["rgb"]`, `obs["ir/left"]` |
| [`CameraTwin`](cameras/camera.md) | one camera | `attach(backend, "wrist")` | `obs["camera"]` |
| [`StereoTwin`](cameras/stereo.md) | a stereo pair | `attach(backend, camera="wrist")` | `obs["left"]`, `obs["right"]` |
| [`LidarTwin`](cameras/lidar.md) | one lidar | `attach(backend, body="link7")` | `obs["lidar"]` |

All four are created with `from_catalog(id)`, except that a custom `DeviceTwin` is built from its
parts as shown above.

Capturing never advances physics: your loop owns the simulation's time. In Isaac Sim a capture
renders one new frame, unless you pass `step=False` because your loop already rendered it.

## I want to

| Task | Start here |
|---|---|
| Create a camera | [`CameraTwin.from_catalog`](cameras/camera.md#twinrobo.camera.CameraTwin.from_catalog) |
| Process my own RGB and depth images | [`CameraTwin.process`](cameras/camera.md#twinrobo.camera.CameraTwin.process) |
| Find a catalog ID | [Specs and catalog](cameras/specs.md) |
| Create my own spec | [Create your own spec](cameras/specs.md#create-your-own-spec) |
| Render in a simulator | [the example above](#twinrobo-api-reference) |
| Use a stereo pair | [`StereoTwin`](cameras/stereo.md) |
| Use a lidar | [`LidarTwin`](cameras/lidar.md) |
| Use a multi-sensor device | [`DeviceTwin`](cameras/devices.md) |
| Run a LIBERO policy through a camera | [`CameraTwinLiberoEnv`](datasets/libero.md) |
| Switch the ISP between frames | [`CameraTwin.set_isp`](cameras/camera.md#twinrobo.camera.CameraTwin.set_isp) |
| Write my own ISP or sensor | [Sensor and ISP](imaging/sensor-isp.md) |
| Read a frame's image and depth | [`CameraFrame`](cameras/frames.md#twinrobo.frame.CameraFrame) |

## Backends

A backend is what the simulator gives TwinRobo.

| Simulator | Backend |
|---|---|
| MuJoCo | `MujocoRenderer(model, data)` |
| robosuite, LIBERO | `RobosuiteRenderer(env)` |
| Isaac Sim | `IsaacBackend()` |

LiDAR runs in MuJoCo for now.

## Conventions

- `rgb` is linear, `[B, 3, H, W]`.
- `depth` is metric z-depth in meters, `[B, 1, H, W]`. It is an input that sets the defocus.
  For catalog cameras it is an output only with `force_depth=True`, as ground truth.

## Observations and rigs

<span id="twinrobo.rig"></span>

::: twinrobo.rig
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Observation, Rig, CameraRig, LidarRig, GroupRig]
