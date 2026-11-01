# Specs and catalog

A spec is a file that describes one sensor or product. The catalog is the collection of specs that
ships with TwinRobo, and you can [create your own](#create-your-own-spec).

[Try it live](https://twinrobo.github.io/catalog/){ .md-button .md-button--primary }

## Supported cameras

Pass an ID to `CameraTwin.from_catalog`. The website's
[camera browser](https://twinrobo.github.io/catalog/) has each camera's sensor, lens and views.

{{ catalog_cameras }}

## Supported stereo modules

Pass an ID to `StereoTwin.from_catalog`.

{{ catalog_modules }}

## Supported lidars

Pass an ID to `LidarTwin.from_catalog`.

{{ catalog_lidars }}

## Supported devices

Pass an ID to `DeviceTwin.from_catalog`.

{{ catalog_devices }}

## Create your own spec

Each spec is a YAML file. Copying a similar entry from the tables above is the quickest start: its
**spec** link shows a complete file.

### A camera

```yaml
# my_camera/camera.yaml
schema_version: 0.1
id: acme/mycam
manufacturer: Acme
product: MyCam
sensor:
  resolution: {width: 1280, height: 800}
  pixel_pitch_um: 3.0
  shutter: {type: global}            # global | rolling
lens:
  focal_length_mm: 2.2
  f_number: 2.0
  focus_distance_m: 1.0
  deeplens_model:
    path: catalog:stereolabs/zed-x/2.2mm/lens.json   # or lens.json beside this file
calibration:
  intrinsic: {fx: 733.3, fy: 733.3, cx: 639.5, cy: 399.5}
  psf: {model: deeplens}
outputs: {depth: false}
validation: {status: estimated}      # estimated | measured | verified | manufacturer_verified
```

`fx` and `fy` are the focal length divided by the pixel pitch: 2.2 mm / 3 µm is 733.3 pixels.
`cx` and `cy` are the image center. Add `calibration.distortion` if you have the lens's OpenCV
coefficients, and use `color: mono` or `spectrum: nir` under `sensor` for such sensors. The
[calibration guide](https://twinrobo.github.io/calibration/) measures these from a real camera.

```python
from twinrobo import CameraTwin

twin = CameraTwin.from_spec("my_camera/camera.yaml")
```

The first use computes the lens's PSF bank, which takes a while, and later uses reuse it. To change
a catalog camera without writing a file, pass overrides:

```python
from twinrobo import CameraSpec, CameraTwin, CatalogRegistry

path = CatalogRegistry().resolve("stereolabs/zed-x/2.2mm")
twin = CameraTwin.from_spec(CameraSpec.from_yaml(path, width=960, height=600))
```

### A stereo module

Name each eye's camera spec, and give the right eye's position relative to the left, in meters.

```yaml
# my_stereo/module.yaml
schema_version: 0.1
id: acme/mystereo
type: stereo
manufacturer: Acme
product: MyStereo
reference: left
eyes:
  left: {spec: ../my_camera/camera.yaml}
  right: {spec: ../my_camera/camera.yaml, translation_m: [0.06, 0.0, 0.0]}
output: rectified                    # rectified | raw
```

```python
from twinrobo import StereoTwin

twin = StereoTwin.from_spec("my_stereo/module.yaml")
```

### A lidar

```yaml
# my_lidar/lidar.yaml
schema_version: 0.1
type: lidar
id: acme/mylidar
manufacturer: Acme
product: MyLidar
scan:
  pattern: spinning                  # spinning | raster
  channels: 32
  vertical_deg: {min: -15.0, max: 15.0}
  columns: 1024
  frame_hz: 10
range:
  min_m: 0.5
  max_m: 100.0
  at_reflectivity: 0.1
noise:
  range_sigma_m: 0.02
validation: {status: estimated}
```

```python
from twinrobo import LidarTwin

twin = LidarTwin.from_spec("my_lidar/lidar.yaml")
```

### A device

A device names its parts by catalog ID, so its parts must be in a catalog. `translation_m` is
optional: leave it out if you do not know where a sensor sits.

```yaml
# device.yaml
schema_version: 0.1
id: acme/mydevice
type: device
manufacturer: Acme
product: MyDevice
parts:
  rgb: {type: camera, id: stereolabs/zed-x/2.2mm, translation_m: [0.0, 0.0, 0.0]}
  ir: {type: stereo, id: intel/realsense-d435}
  roof: {type: lidar, id: ouster/os1-64}
```

```python
from twinrobo import DeviceTwin

device = DeviceTwin.from_spec("device.yaml")
```

### Your own catalog

To load your specs by ID, keep each at `<root>/<id>/` and point `TWINROBO_CATALOG` at the root:

```bash
export TWINROBO_CATALOG=/path/to/my_catalog   # holds acme/mycam/camera.yaml
```

```python
from twinrobo import CameraTwin

twin = CameraTwin.from_catalog("acme/mycam")
```

To add a camera to the catalog for everyone, see
[adding a camera](https://github.com/TwinRobo/TwinRobo-Core/blob/main/CONTRIBUTING.md#adding-a-camera).

<span id="twinrobo.spec"></span>

## Spec reference

::: twinrobo.spec.CameraSpec

::: twinrobo.spec.apply_overrides

<span id="twinrobo.registry"></span>

## Catalog reference

::: twinrobo.registry.CatalogRegistry

<span id="twinrobo.geometry"></span>

## Intrinsics

::: twinrobo.geometry.CameraIntrinsics
