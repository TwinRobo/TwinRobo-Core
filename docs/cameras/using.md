# Using a twin

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
| [`DeviceTwin`](devices.md) | any sensors in one product | `attach(backend, rgb=..., ir=...)` | `obs["rgb"]`, `obs["ir/left"]` |
| [`CameraTwin`](camera.md) | one camera | `attach(backend, "wrist")` | `obs["camera"]` |
| [`StereoTwin`](stereo.md) | a stereo pair | `attach(backend, camera="wrist")` | `obs["left"]`, `obs["right"]` |
| [`LidarTwin`](lidar.md) | one lidar | `attach(backend, body="link7")` | `obs["lidar"]` |

All four are created with `from_catalog(id)`, except that a custom `DeviceTwin` is built from its
parts as shown above.

Capturing never advances physics: your loop owns the simulation's time. In Isaac Sim a capture
renders one new frame, unless you pass `step=False` because your loop already rendered it.

## Backends

A backend is what the simulator gives TwinRobo.

| Simulator | Backend |
|---|---|
| MuJoCo | `MujocoRenderer(model, data)` |
| robosuite, LIBERO | `RobosuiteRenderer(env)` |
| Isaac Sim | `IsaacBackend()` |

LiDAR runs in MuJoCo for now.

## Observations and rigs

<span id="twinrobo.rig"></span>

::: twinrobo.rig
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Observation, Rig, CameraRig, LidarRig, GroupRig]
