# Changelog

## Unreleased

**New**

- One API for every twin: `twin.attach(backend, ...)` gives a rig, and `rig.capture()` gives an
  `Observation`, for `CameraTwin`, the new `StereoTwin`, `LidarTwin` and the new `DeviceTwin`.
  `LidarTwin.from_catalog` is new, and catalog devices are in `device.yaml`.

**Changed**

- `CameraTwin.attach` now takes a simulator backend and returns a rig, and `CameraTwin.get_frame`
  is gone: both only raised an error before. `MujocoCameraTwin` and `IsaacCameraTwin` still work.

**Documentation**

- The API reference moved into this repository, in `docs/`, and is hosted on
  [Read the Docs](https://twinrobo-core.readthedocs.io/en/latest/). The guides stay on twinrobo.github.io, and the old API
  addresses redirect.

**Fixes**

- Ray casting follows the lens: after `twin.reference.set_focus()`, the next frame of
  `IsaacCameraTwin` and `MujocoCameraTwin` uses the new focus instead of the first one.
  Isaac keeps its pupil-view render products across a refocus where they cover the field.
- Long refocus sequences in Isaac use less GPU memory. Isaac does not release a render
  product's memory when its size changes, so views that must grow now grow with headroom, and
  a focus sweep resizes them a few times instead of many. Each resize still costs memory.
- Isaac warns when texture streaming is on: it blurs frames when GPU memory runs low.
  `FULL_TEXTURES_ARG` is the `SimulationApp` argument that turns it off.

**Sensor and ISP**

- `CameraTwin.from_catalog(..., isp=True)`: a real camera's sensor and ISP on any rendering
  method. Raspberry Pi Camera Module 3 (standard, wide) carry their tunings; off by default,
  so existing frames do not change. `TunedSensor` gives auto-exposure, a Bayer mosaic, noise
  and the ADC; `TunedISP` gives Raspberry Pi's Bayesian white balance, demosaic, denoise,
  color matrices, tone curve and sharpening, in torch on the frame's device (white balance
  reads small zone statistics to the CPU).
- Tunings from libcamera for IMX708, IMX708 wide, IMX219, IMX477, IMX296, OV5647 and IMX500
  (`twinrobo.isp.list_tunings`), or a Raspberry Pi color tuning file by path.
- Choose the ISP: `isp=` also takes a tuning name or path, or your own `ISPModel` with
  `sensor=`. `twin.set_isp()` switches between frames, keeping the optics, and starts with no
  state from earlier frames. `ISPModel` documents what a custom ISP must do; `input_format`
  and `output_format` reject a sensor and ISP that do not fit.
- `frame.metadata["pipeline"]` records the sensor and ISP of each frame, and whether the
  tuning is the one in the camera's spec.
- `twin.reset()` and each adapter's `reset()` start frames again, in the ISP as in the sensor.
- `frame.raw` is the Bayer mosaic with these models; `frame.rgb` is the camera's output image
  decoded to linear, so `linear_to_srgb` gives its pixels.

## v0.1.0 (2026-10-28)

The first release: digital twins of real cameras for robot simulators.

**Cameras and optics**

- `CameraTwin`: a catalog camera's optics, sensor geometry and ISP applied to a simulator's
  image. Lenses are traced with [DeepLens](https://github.com/vccimaging/DeepLens):
  depth- and field-dependent blur, distortion, chromatic aberration and vignetting.
- Two rendering methods: `psf` (the simulator's pinhole render, blurred by the lens'
  point-spread functions and distorted; fast) and `raycast` (every pixel's rays traced
  through the lens and cast into the scene's triangles on the GPU, exact around defocused
  occluders).
- Camera catalog (Apache-2.0): Intel RealSense D435 / D455 (RGB and IR imagers),
  Stereolabs ZED X 2.2 mm / 4 mm and ZED 2i, Luxonis OAK-D (color and mono), Logitech
  C920 and the Raspberry Pi Camera Module 3 (standard and wide), as single cameras or
  stereo modules with adjustable baselines and rectified output. Entries are `estimated`
  (datasheet geometry, surrogate lens designs).
- Raw sensor or calibrated (undistorted) output; ground-truth depth on request.
- Calibration tool: intrinsics, distortion, focus, vignetting and stereo baseline from
  ChArUco, slanted-edge and flat-field captures, written as a `measured` catalog entry.

**Simulators**

- MuJoCo (`MujocoCameraTwin`), cameras mountable on any robot link.
- LIBERO (robosuite 1.4): any camera of a task or demo replaced by a twin; policies and
  evaluation scripts unchanged.
- RoboCasa (robosuite 1.5): kitchens, more robots, demos replayed in their own layouts.
- NVIDIA Isaac Sim 6.x (`IsaacCameraTwin`), with a Docker recipe.

**LiDAR (preview)**

- `twinrobo.lidar`: spinning and raster lidars cast into MuJoCo scenes (range, an
  intensity proxy, dropout by reflectivity, seeded noise, FMCW radial velocity), with
  Velodyne VLP-16, Ouster OS1-64 and Hesai XT32 catalog entries.

**Reproducible optics.** DeepLens samples rays at random when it focuses a lens and builds
its point-spread functions; both are now seeded (without touching the caller's random
state), so a camera's PSF bank is the same every time it is built on a device.

**Install** (Python 3.12): `pip install "twinrobo-core[mujoco]"` from PyPI, with DeepLens
`deeplens-core>=2.5.6` (any torch from 2.10). For Isaac Sim,
`pip install "twinrobo-core[isaac]" --extra-index-url https://pypi.nvidia.com` brings Isaac
Sim 6.1 and its torch 2.11 (Ubuntu 22.04 or later; in an environment of its own, since
Isaac Sim pins its own MuJoCo); or use the container recipe (`docker/isaac/run.sh`), which
installs DeepLens on the Isaac Sim image's own Python and torch.

**Known limits:** catalog lenses are surrogates, not the cameras' own prescriptions;
sensor noise, exposure and motion are not simulated; the simulators render visible light
only, so infrared imagers are shown as visible-light mono views; depth cameras' measured
depth is not simulated yet (see the [roadmap](ROADMAP.md)).
