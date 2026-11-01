# Monocular

A single camera. [`CameraTwin`](camera.md) is the class; this page shows how to use it and which
cameras you can load.

```python
from twinrobo import CameraTwin
from twinrobo.plugins.mujoco import MujocoCameraTwin, MujocoRenderer

twin = CameraTwin.from_catalog("logitech/c920")
cam = MujocoCameraTwin(twin, MujocoRenderer(model, data), camera="wrist")
frame = cam.get_frame()
```

The result is a [`CameraFrame`](frames.md). If you already have images, use
`twin.process(rgb, depth)` instead of a simulator adapter.

- **Mono sensors** return their image in all three channels, so frames keep the RGB convention.
- **Depth** is readable only if the camera outputs it, or with `force_depth=True`, which gives the
  simulator's ground truth.

## Catalog cameras

Pass an ID to `CameraTwin.from_catalog`. The website's
[camera browser](https://twinrobo.github.io/catalog/) has each camera's sensor, lens and views.

{{ catalog_cameras }}
