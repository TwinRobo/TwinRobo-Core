# Changelog

## v0.1.0 (2026-10-12)

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

**Install:** Python 3.12;
`pip install "twinrobo-core[mujoco] @ git+https://github.com/TwinRobo/TwinRobo-Core@v0.1.0"`.
DeepLens is installed from its pinned commit, so this release is not on PyPI.

**Known limits:** catalog lenses are surrogates, not the cameras' own prescriptions;
sensor noise, exposure and motion are not simulated; the simulators render visible light
only, so infrared imagers are shown as visible-light mono views; depth cameras' measured
depth is not simulated yet (see the [roadmap](ROADMAP.md)).
