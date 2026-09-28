# Cameras and stereo modules

A camera is a `CameraSpec` YAML: sensor (resolution, pixel pitch, shutter),
lens (a DeepLens lens file), focus, and optional calibration (intrinsics,
distortion). Stereo products are *modules* that pair two eyes.

## Catalog

Catalog lookup resolves `vendor/model/...` to `<root>/<id>/camera.yaml` under
the roots in `TWINROBO_CATALOG`, then the built-in `catalog/`:

```python
from twinrobo import CameraTwin

camera = CameraTwin.from_catalog("stereolabs/zed-x/2.2mm")
```

The full list, with each entry's sensor, lens and field of view, is the
[camera catalog](catalog.md) (generated from the catalog files). Today:

| Vendor | Cameras | Stereo modules (baseline) |
|---|---|---|
| Intel RealSense | D435 / D435i color and depth imager, D455 color and depth imager | D435 (50 mm), D455 (95 mm) |
| Stereolabs | ZED X 2.2 mm and 4 mm eyes, ZED 2i 2.1 mm eye | ZED X (120 mm), ZED X Mini (50 mm), ZED 2i (120 mm) |
| Luxonis | OAK-D color and mono | OAK-D (75 mm) |
| Logitech | C920 HD Pro Webcam | |
| Raspberry Pi | Camera Module 3, Camera Module 3 Wide | |

`examples/cellphone80deg` is a teaching example: a DeepLens
design lens on a hypothetical sensor, not a real camera.

## Depth output

TwinRobo simulates what cameras **image**. It does not yet simulate what depth
cameras **measure**, so no catalog camera outputs depth: reading `frame.depth`
raises `DepthUnavailableError`. A policy is not trained on a signal no real
camera gives.

The simulator's depth is still there when you ask for it, as ground truth:

```python
frame = cam.get_frame(force_depth=True)   # or twin.process(rgb, depth, force_depth=True)
frame.depth                                # [1, 1, H, W], meters: exact scene depth
```

It is the scene's exact z-depth, seen through the camera's lens geometry: right
for labels, evaluation and debugging, but not a depth sensor's reading. A
RealSense D455, for example, measures depth by matching its two infrared images
(95 mm apart, with a projected dot pattern): its depth has an error that grows
with the square of distance, holes on shiny, dark or unmatched surfaces, a
minimum range of about half a meter and noisy depth edges, none of which ground
truth has. Simulating that is on the
[roadmap](https://github.com/TwinRobo/TwinRobo-Core/blob/main/ROADMAP.md#depth-cameras).

A spec's `outputs.depth: true` is reserved for cameras whose depth output
TwinRobo simulates; `frame.has_depth` tells whether a frame exposes depth.

## Stereo modules

Each `camera.yaml` models one eye, once: products sharing an eye (the ZED X and
ZED X Mini 2.2 mm) reference the same spec. A stereo product is a `module.yaml`
(`twinrobo.stereo`): the per-eye specs, the right eye's pose in the left
eye's frame (the baseline), the housing size and the default output
(`rectified`, as the vendor SDK delivers).

A module is placed as one unit by its left eye's pose and renders as two
cameras, `<name>_L` and `<name>_R`, that move together. With `rectify` (per
camera config) the output is undistorted to the pinhole with the eye's
intrinsics, keeping the lens blur, so the pair is row-aligned for stereo
matching.

| Module | Baseline | Eyes |
|---|---|---|
| `intel/realsense-d435` | 50 mm | `intel/realsense-d435/depth` |
| `intel/realsense-d455` | 95 mm | `intel/realsense-d455/depth` |
| `luxonis/oak-d` | 75 mm | `luxonis/oak-d/mono` |
| `stereolabs/zed-2i/2.1mm` | 120 mm | ZED 2i 2.1 mm eye |
| `stereolabs/zed-x/2.2mm`, `stereolabs/zed-x/4mm` | 120 mm | ZED X 2.2 mm / 4 mm |
| `stereolabs/zed-x-mini/2.2mm` | 50 mm | the ZED X 2.2 mm eye (same lens per the datasheet) |

**Baselines are adjustable.** Robots mount the same module at different spacings,
or you may want to study the baseline itself: `StereoModule.with_baseline(0.09)`
gives the module at 90 mm, a custom mount takes `CameraMount(..., module=...,
baseline_mm=90)`, and `twinrobo.plugins.mujoco.mounts.set_pair_baseline` moves a stereo
pair compiled into a robot model (e.g. the multi-camera arms' wrist module)
without rebuilding it.

## How catalog entries are made

Vendors rarely publish lens prescriptions, so an `estimated` entry is built from
the datasheet:

- **Geometry:** `tools/catalog/fit_fov_geometry.py` fits the focal length (and, for
  wide lenses, OpenCV distortion) to the datasheet's horizontal, vertical and
  diagonal fields of view, keeping the mapping monotonic (a real lens never
  folds the image). Check which sensor area the datasheet's FoV refers to: the
  ZED 2i's figures fit its full 2688×1520 array, while it streams a 2208×1242
  center crop.
- **Blur:** a glass design of similar field (a DeepLens lens) scaled to the
  camera's focal length and stopped to its f-number
  (`tools/catalog/make_surrogate_lens.py`).
- **Focus:** fixed-focus cameras are set near the hyperfocal distance;
  autofocus cameras at 1 m, to be overridden with the working distance.

Every number and its source is recorded in the entry's header. To reach
`measured`, [calibrate a real unit](calibration.md) and see
[CONTRIBUTING](../CONTRIBUTING.md#adding-a-camera).

## Your own cameras

Copy a spec, point `lens.deeplens_model.path` at a DeepLens lens file, and set
the sensor and focus. To reconfigure an existing camera without editing its
file, pass overrides when loading it:

```python
from twinrobo import CameraSpec, CameraTwin, CatalogRegistry

path = CatalogRegistry().resolve("stereolabs/zed-x/2.2mm")
spec = CameraSpec.from_yaml(path, width=960, height=600, focus_distance_m=0.6)
twin = CameraTwin.from_spec(spec)
```

Resolution overrides keep the field of view (intrinsics and pixel pitch scale
with it). To add a camera to the catalog for everyone, see
[CONTRIBUTING](../CONTRIBUTING.md#adding-a-camera).
