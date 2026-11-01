# Devices

Several sensors in one product. A RealSense D435, for example, is an RGB camera beside a stereo
IR pair. `DeviceTwin` captures all of them from one simulator frame.

```python
from twinrobo import DeviceTwin

device = DeviceTwin.from_catalog("intel/realsense-d435")
rig = device.attach(backend, rgb="rgb_cam", ir={"left": "ir_l", "right": "ir_r"})
obs = rig.capture()
obs["rgb"], obs["ir/left"], obs["ir/right"]
```

Any twins can make a device, lidars included:

```python
device = DeviceTwin({"front": camera_twin, "roof": lidar_twin})
rig = device.attach(backend, front="wrist", roof={"body": "base", "pos": (0, 0, 0.5)})
```

The cameras stay where the simulator has them. A catalog device does not place them for you, and
it does not simulate the device's own depth output.

A D435's RGB position is not in its datasheet, so it is left unknown. See the
[supported devices](specs.md#supported-devices).

## The API

::: twinrobo.device.DeviceTwin

::: twinrobo.device.DeviceSpec

::: twinrobo.device.DevicePart

::: twinrobo.device.DeviceRegistry
