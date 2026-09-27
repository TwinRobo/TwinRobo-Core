# Optics and rendering

TwinRobo turns what a simulator renders into what a real camera would record.
This page covers how an image is formed: the PSF pipeline, lens distortion, and
the three rendering methods a camera can use.

## Conventions

Defined in `twinrobo/frame.py`, used across the package:

| Quantity | Shape | Units |
|---|---|---|
| `rgb` | `[B, 3, H, W]` float | linear, nominally `[0, 1]` |
| `depth` | `[B, 1, H, W]` float | meters, positive z-depth; `inf`/`nan`/`<=0` = invalid |

Tensors stay on the GPU. DeepLens units (negative millimeters for object depth,
micrometers for wavelength) appear only inside `twinrobo/optics/deeplens.py`,
the single DeepLens boundary.

## The PSF pipeline

```text
offline (DeepLens, once per camera config, cached)
  DeepLensOptics.build_psf_bank -> PSFBank [depth, field_y, field_x, rgb, ks, ks]

runtime (CameraTwin, GPU, no DeepLens calls)
  DenseDepthRenderer.render(rgb, depth)
    soft depth-layer assignment (disparity)
    spatially varying scatter blur (bilinear PSF interpolation, FFT overlap-add)
    occlusion-aware normalized layer compositing (near -> far)
```

```python
from twinrobo import CameraTwin

# The first call builds the PSF bank with DeepLens (~20 s on a GPU) and caches it
# under $TWINROBO_CACHE/psf (default ~/.cache/twinrobo/psf); later calls load it.
camera = CameraTwin.from_catalog("examples/cellphone80deg")
frame = camera.process(rgb, depth)  # linear RGB [B,3,H,W] + metric depth [B,1,H,W], on GPU
frame.rgb_ideal, frame.rgb_optical, frame.rgb
```

To precompute a bank: `python tools/generate_psf/generate_psf.py <spec.yaml>`.

At 1920x1200 the renderer agrees with DeepLens' block PSF-map renderer at
about 46 dB PSNR. Its point responses match PSFs traced directly by DeepLens
close to the Monte Carlo noise floor
([benchmark](benchmarks/phase1_optics.md)). `DeepLensOptics.render_reference`
is the slow physical reference, used for validation only.

**Lens distortion** is modeled separately from blur. A spec with calibrated
intrinsics and `calibration.distortion.model: opencv` (`k1 k2 p1 p2 k3`) is
rendered from a wider pinhole *source* (`camera.render_intrinsics`, same central
focal length). `DistortionWarp` resamples it onto the distorted sensor
(bilinear color, nearest depth) before the PSF blur. `frame.rgb_ideal` is then
the undistorted pinhole view with the camera's intrinsics. With `rectify=True`
the output is undistorted back to that pinhole, keeping the lens blur, as
stereo SDKs deliver.

## Rendering methods

Each TwinRobo camera picks how its image is formed, per `MujocoCameraTwin`.

| Method | How | Per frame, 1920×1200 (RTX 3090) | Gets right |
|---|---|---|---|
| `psf` (default) | pinhole render + depth- and field-dependent PSF blur (2.5D) | ~0.1–0.4 s | blur of what the pinhole sees |
| `pupil` | every pixel's rays traced through the real lens (DeepLens); one MuJoCo view per pupil cell (7), each ray looked up in its cell's view, refined with depth | ~0.5 s | defocus across the aperture, the lens' real distortion, chromatic aberration, vignetting; approximate behind occluders |
| `raycast` | the same lens rays intersected with the scene's triangles on the GPU (NVIDIA Warp BVH); each hit shaded from a pupil view that sees it | ~0.9 s | as `pupil`, with exact per-ray visibility (see-through around defocused foreground) |

```python
from twinrobo.mujoco import MujocoCameraTwin

cam = MujocoCameraTwin(
    twin,
    backend,
    camera="agentview",
    render="raycast",
    rays_per_pixel=8,
    pupil_views=7,
    shading="corrected",
    view_oversample=1,
)
frame = cam.get_frame()  # frame.metadata has the render stats
```

- **Lens rays** (`twinrobo.optics.lensrays`, traced by `DeepLensOptics.trace_lens_rays`):
  - sensor pixel -> exit pupil -> lens -> object-space rays, per RGB wavelength,
    in the MuJoCo/USD camera frame. They depend only on lens, focus and sensor,
    so they are traced once per config (~13 s, 0.6 GB for 1920×1200 at 8
    rays/px) and cached.
  - Brightness is a separate, densely traced relative-illumination map, so a
    few rays per pixel don't turn vignetting into noise. Strongly vignetted
    pixels are retraced with 4x candidates.
- **Shading is the simulator's own** (no PBR). The pupil views are MuJoCo
  rasterizations from points on the entrance pupil, and hits look up colors
  there with depth-aware filtering.
- **Lens shading:** `corrected` (default) is what a camera ISP outputs after
  lens-shading correction; `raw` keeps the sensor's vignetting and cos^4 falloff.
- **Rectify:** lens-ray images are undistorted with the lens' own chief-ray map
  (`LensRays.rectify_grid`), not a fitted model.
- **Validation** (`tests/mujoco/test_lensrender.py`):
  - A marker lands where the lens' chief ray points (282.5 px; the paraxial
    pinhole says 287).
  - A post 0.3 m in front of the focus plane: its partially covered, defocused
    edges match the exact answer computed from the traced rays to ~0.05 with
    ray cast (pupil raster ~0.1, pinhole + PSF 0.2–0.4).
- **Limits:**
  - Objects thinner than a sensor pixel: MuJoCo multisamples color, so their
    silhouettes are blends in every view. Raise `view_oversample` for them.
  - Height fields and SDF geoms are not ray cast.
  - Scenes come from a simulator's *lens scene* (`twinrobo.optics.lensrender`):
    MuJoCo (`twinrobo.mujoco.lensrender`) and Isaac Sim (`twinrobo.isaac.lens`).
- **Lens files matter:** ray-based methods trace the lens file literally. The
  ZED X lens in the catalog is a blur surrogate (a scaled Canon RF16mm
  prescription); its image circle does not cover the full ZED X sensor, so its
  corners render dark.

## Performance notes

At 1920x1200 with 16 layers and a 17x11 PSF grid on an RTX 3090, the PSF
renderer takes about 470 ms when all 16 depth layers are occupied (a ground
plane does that) and about 40 ms for a single plane. The cost scales with the
number of occupied layers. Levers not yet explored: layer/tile sparsity, fewer
blurred alpha channels, low-rank PSF bases.
