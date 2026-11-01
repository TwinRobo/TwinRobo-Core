# Camera

A single camera is a `CameraTwin`.

[Try it live](https://twinrobo.github.io/catalog/){ .md-button .md-button--primary }

```python
from twinrobo import CameraTwin
from twinrobo.plugins.mujoco import MujocoRenderer

twin = CameraTwin.from_catalog("logitech/c920")
rig = twin.attach(MujocoRenderer(model, data), "wrist")
frame = rig.capture()["camera"]
```

The result is a [`CameraFrame`](frames.md). If you already have images, use
`twin.process(rgb, depth)` instead of attaching to a simulator.

- **Mono sensors** return their image in all three channels, so frames keep the RGB convention.
- **Depth** is readable only if the camera outputs it, or with `force_depth=True`, which gives the
  simulator's ground truth.

## Catalog cameras

See the [supported cameras](specs.md#supported-cameras).

## Create a camera

::: twinrobo.camera.CameraTwin
    options:
      members: false

::: twinrobo.camera.CameraTwin.from_catalog

::: twinrobo.camera.CameraTwin.from_spec

## Attach to a simulator

::: twinrobo.camera.CameraTwin.attach

## Render

::: twinrobo.camera.CameraTwin.process

::: twinrobo.camera.CameraTwin.develop

::: twinrobo.camera.CameraTwin.render_intrinsics

::: twinrobo.camera.CameraTwin.lens_geometry

::: twinrobo.camera.CameraTwin.outputs_depth

::: twinrobo.camera.CameraTwin.sensor_color

## Choose the sensor and ISP

::: twinrobo.camera.CameraTwin.set_isp

::: twinrobo.camera.CameraTwin.reset

::: twinrobo.camera.build_sensor_isp
