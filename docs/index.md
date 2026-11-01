# TwinRobo v0.1.0 API reference

!!! note "Rebuilt after the release"
    Rebuilt on 2026-11-01 from the `v0.1.0` tag's docstrings and the API pages of 2026-10-28.

Generated from the docstrings of the `twinrobo` package. Start with
[`CameraTwin`](cameras/camera.md); a simulator adapter wraps a `CameraTwin` and renders its input from a
simulator camera.

A typical use:

```python
from twinrobo import CameraTwin
from twinrobo.plugins.mujoco import MujocoCameraTwin, MujocoRenderer

twin = CameraTwin.from_catalog("stereolabs/zed-x/2.2mm")      # a real camera
cam = MujocoCameraTwin(twin, MujocoRenderer(model, data), camera="wrist")
frame = cam.get_frame()                                         # CameraFrame
frame.rgb                                                       # the camera's image, on the GPU
```

Tensors follow one convention everywhere: `rgb` is linear `[B, 3, H, W]`,
`depth` is metric z-depth `[B, 1, H, W]` in meters: an input (it sets the defocus),
and an output only with `force_depth=True`, as ground truth (see [`twinrobo.frame`](cameras/frames.md)).
