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

| ID | Camera | Geometry | Blur | Status |
|---|---|---|---|---|
| `stereolabs/zed-x/2.2mm` | ZED X / ZED X Mini eye, 2.2 mm f/2.2, 1920x1200, 3 µm, global shutter | fx 745.6, k1 -0.0634, k2 0.0074: exact fit to the datasheet FoV 110 x 80 x 120 deg | surrogate: DeepLens rf16mm scaled to f 2.237 mm, f/2.2 | estimated |
| `stereolabs/zed-x/4mm` | ZED X eye, 4 mm f/2.2 | fx 1268.8, k1 ~ 0: fit to FoV 75 x 50 x 83 deg | surrogate: DeepLens rf24mm scaled to f 3.806 mm, f/2.2 | estimated |

`examples/cellphone80deg` is a teaching example: a DeepLens
design lens on a hypothetical sensor, not a real camera.

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
| `stereolabs/zed-x/2.2mm`, `stereolabs/zed-x/4mm` | 120 mm | ZED X 2.2 mm / 4 mm |
| `stereolabs/zed-x-mini/2.2mm` | 50 mm | the ZED X 2.2 mm eye (same lens per the datasheet) |

## How catalog entries are made

Stereolabs publishes no lens prescription, so blur uses a scaled glass design
of similar FoV (`tools/catalog/make_surrogate_lens.py`), and the fixed,
unpublished focus is set near the hyperfocal distance. Geometry (focal length,
distortion) is fitted exactly to the datasheet fields of view.

To reach `measured`: replace intrinsics and distortion with the camera's
factory calibration (per serial number), then fit blur and focus from a capture
session (slanted edge / ChArUco at several distances, flat field for
vignetting). See `tools/calibrate/`, and
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
