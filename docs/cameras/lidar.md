# LiDAR

<span id="twinrobo.lidar"></span>

Catalog lidars ray cast into simulator scenes. MuJoCo is supported. See the
[supported lidars](specs.md#supported-lidars).

```python
from twinrobo import LidarTwin
from twinrobo.plugins.mujoco import MujocoRenderer

lidar = LidarTwin.from_catalog("ouster/os1-64")
rig = lidar.attach(MujocoRenderer(model, data), body="robot0_link7", pos=(0, 0, 0.1))
frame = rig.capture()["lidar"]
points = frame.points()   # XYZ returns in the sensor frame
```

## In MuJoCo

<span id="twinrobo.plugins.mujoco.lidar"></span>

::: twinrobo.plugins.mujoco.lidar
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Scanning

<span id="twinrobo.lidar.scan"></span>

::: twinrobo.lidar.scan
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Specs

::: twinrobo.lidar.spec.LidarSpec

::: twinrobo.lidar.registry.LidarRegistry
