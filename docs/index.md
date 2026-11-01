# TwinRobo API reference

The classes and functions of the `twinrobo` package. New here? The
[guides](https://twinrobo.github.io/getting-started/) explain how it all fits together.

```python
from twinrobo import DeviceTwin
from twinrobo.plugins.mujoco import MujocoRenderer

device = DeviceTwin.from_catalog("intel/realsense-d435")   # choose a device
rig = device.attach(MujocoRenderer(model, data),            # put it in a scene
                    rgb="rgb_cam", ir={"left": "ir_l", "right": "ir_r"})
obs = rig.capture()                                         # capture its sensors
obs["rgb"].rgb                                              # the RGB image
```

A single camera, stereo pair or lidar works the same way: see [Using a twin](cameras/using.md).

## I want to

| Task | Start here |
|---|---|
| Create a camera | [`CameraTwin.from_catalog`](cameras/camera.md#twinrobo.camera.CameraTwin.from_catalog) |
| Process my own RGB and depth images | [`CameraTwin.process`](cameras/camera.md#twinrobo.camera.CameraTwin.process) |
| Find a catalog ID | [Specs and catalog](cameras/specs.md) |
| Create my own spec | [Create your own spec](cameras/specs.md#create-your-own-spec) |
| Render in a simulator | [Using a twin](cameras/using.md) |
| Use a stereo pair | [`StereoTwin`](cameras/stereo.md) |
| Use a lidar | [`LidarTwin`](cameras/lidar.md) |
| Use a multi-sensor device | [`DeviceTwin`](cameras/devices.md) |
| Run a LIBERO policy through a camera | [`CameraTwinLiberoEnv`](datasets/libero.md) |
| Switch the ISP between frames | [`CameraTwin.set_isp`](cameras/camera.md#twinrobo.camera.CameraTwin.set_isp) |
| Write my own ISP or sensor | [Sensor and ISP](imaging/sensor-isp.md) |
| Read a frame's image and depth | [`CameraFrame`](cameras/frames.md#twinrobo.frame.CameraFrame) |

## Conventions

- `rgb` is linear, `[B, 3, H, W]`.
- `depth` is metric z-depth in meters, `[B, 1, H, W]`. It is an input that sets the defocus.
  For catalog cameras it is an output only with `force_depth=True`, as ground truth.
