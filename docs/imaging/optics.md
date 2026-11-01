# Optics

How light passes through the lens. `CameraTwin` runs this for you, so you only need this page to
choose a method or lens, or to add your own.

## Supported methods

Pick a method when you attach a camera: `twin.attach(backend, "wrist", render="raycast")`.

| Method | How | Best for |
|---|---|---|
| `render="psf"` (default) | blurs the simulator's image with the lens's point-spread functions | speed |
| `render="raycast"` | traces every pixel's rays through the lens into the scene | accuracy around defocused objects |

## Supported optics models

| Model | What it does |
|---|---|
| `IdentityOptics` | nothing: an ideal pinhole |
| `DeepLensOptics` | a traced lens file, with depth- and field-dependent blur |
| Your own `OpticsModel` | [anything you write](#your-own-optics-model) |

## Supported lenses

Every catalog lens is a surrogate: a published glass design scaled to the camera's focal length and
f-number, because vendors publish no prescriptions. Any camera spec can reuse one through its
reference, under `lens.deeplens_model.path`.

{{ catalog_lenses }}

Distortion comes from the spec's `calibration.distortion`, in OpenCV's `k1 k2 p1 p2 k3` form.

## Create your own

### Your own optics model

Subclass `OpticsModel` and override `render`. It takes linear `[B, 3, H, W]` RGB and metric
`[B, 1, H, W]` depth, and returns linear `[B, 3, H, W]` RGB of the same size.

```python
import torch.nn.functional as F

from twinrobo import CameraTwin
from twinrobo.geometry import CameraIntrinsics
from twinrobo.optics import OpticsModel


class BoxBlur(OpticsModel):
    def render(self, rgb, depth, metadata=None):
        return F.avg_pool2d(rgb, 5, stride=1, padding=2)


K = CameraIntrinsics(640, 480, 500.0, 500.0, 319.5, 239.5)
twin = CameraTwin(optics=BoxBlur(), intrinsics=K)
```

The `psf` method runs your model. The `raycast` method traces a lens file instead, so it needs
`DeepLensOptics`.

### Your own lens

A lens is a DeepLens JSON file. Put it beside your spec and name it there:

```yaml
lens:
  deeplens_model:
    path: lens.json
```

If you have no prescription, scale a known design into a surrogate with
`tools/catalog/make_surrogate_lens.py`:

```bash
python tools/catalog/make_surrogate_lens.py SRC.json lens.json \
    --focal-mm 2.2 --fnum 2.0 --sensor-diag-mm 4.5 --info "my surrogate"
```

See [Specs and catalog](../cameras/specs.md#a-camera) for the full spec.

## The interface

<span id="twinrobo.optics.base"></span>

::: twinrobo.optics.base
    options:
      show_root_heading: false
      show_root_toc_entry: false

## PSF rendering

The fast method: blur the simulator's image with the lens's point-spread functions.

::: twinrobo.optics.deeplens.DeepLensOptics

::: twinrobo.optics.psf.PSFBank

::: twinrobo.optics.cache.PSFCache

::: twinrobo.optics.dense_depth.DenseDepthRenderer

## Ray casting

The exact method: trace every pixel's rays through the lens into the scene.

<span id="twinrobo.optics.lensrays"></span>

::: twinrobo.optics.lensrays
    options:
      show_root_heading: false
      show_root_toc_entry: false

<span id="twinrobo.optics.lensrender"></span>

::: twinrobo.optics.lensrender
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Distortion

<span id="twinrobo.optics.distortion"></span>

::: twinrobo.optics.distortion
    options:
      show_root_heading: false
      show_root_toc_entry: false
