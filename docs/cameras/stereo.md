# Stereo

Two eyes that are captured together, such as a ZED X.

```python
from twinrobo import StereoTwin

twin = StereoTwin.from_catalog("stereolabs/zed-x/2.2mm")
rig = twin.attach(backend, left="cam_l", right="cam_r")
obs = rig.capture()
obs["left"].rgb, obs["right"].rgb
```

The eyes use the simulator's two cameras, so those cameras set the real baseline. In MuJoCo,
TwinRobo warns if they are not at the module's baseline. The
[supported modules](specs.md#supported-stereo-modules) are in the catalog.

## Modules

<span id="twinrobo.stereo"></span>

::: twinrobo.stereo
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [StereoTwin, StereoModule, Eye, ModuleRegistry]

## Eye helpers

::: twinrobo.stereo.eye_poses

::: twinrobo.stereo.eye_name
