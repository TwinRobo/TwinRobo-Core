# Optics

How light passes through the lens. Most users never call these directly: `CameraTwin` runs them.

## Write your own

Subclass this for your own optics.

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
