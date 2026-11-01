# LiDAR

<span id="twinrobo.lidar"></span>

Catalog lidars ray cast into simulator scenes. MuJoCo is supported.

```python
from twinrobo.lidar import LidarRegistry
from twinrobo.plugins.mujoco.lidar import MujocoLidar

lidar = MujocoLidar(LidarRegistry().load("ouster/os1-64"), model, data,
                    body="robot0_link7", pos=(0, 0, 0.1))
frame = lidar.scan()
points = frame.points()   # XYZ returns in the sensor frame
```

## Catalog lidars

Load an ID with `LidarRegistry().load(id)`.

{{ catalog_lidars }}

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
