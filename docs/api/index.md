# API reference

Generated from the docstrings of the `twinrobo` package. Start with the core
objects; a simulator adapter wraps a `CameraTwin` and renders its input from a
simulator camera.

| Area | Main entry points |
|---|---|
| [Core](core.md) | `CameraTwin` (optics → sensor → ISP), `CameraSpec`, `CatalogRegistry`, `CameraFrame`, stereo modules |
| [Optics](optics.md) | `DeepLensOptics`, `PSFBank` and `PSFCache`, `DenseDepthRenderer`, `LensRays`, the lens-ray renderer, distortion |
| [Sensor and ISP](sensor_isp.md) | sensor and ISP stage interfaces, color conversion |
| [MuJoCo](mujoco.md) | `MujocoCameraTwin`, render backends, camera mounts |
| [Isaac Sim](isaac.md) | `IsaacCameraTwin`, the Isaac lens scene |
| [LIBERO and RoboCasa](robot_learning.md) | `CameraTwinLiberoEnv`, robots, demo replay, RoboCasa episodes |
| [Validation](validation.md) | image, geometry and optics metrics |

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
and an output only with `force_depth=True`, as ground truth (see [`twinrobo.frame`](core.md#twinrobo.frame)).
