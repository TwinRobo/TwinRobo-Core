# Development

See [CONTRIBUTING](../CONTRIBUTING.md) for setup, adding cameras and pull
requests.

## Layout

```text
twinrobo/          the SDK (simulator independent core + simulator adapters)
  camera.py          CameraTwin: optics -> sensor -> ISP
  frame.py           CameraFrame + tensor conventions
  spec.py            CameraSpec (schema 0.1) and spec overrides
  registry.py        catalog resolution (TWINROBO_CATALOG + built-in catalog)
  stereo.py          stereo modules (two eyes, one housing)
  catalog/           built-in cameras (<id>/camera.yaml + lens.json) and stereo modules
  optics/            DeepLensOptics, PSFBank/PSFCache, DenseDepthRenderer, LensRays,
                     RadialDistortion + DistortionWarp
  optics/lensrender.py  lens-ray methods (pupil raster, ray cast), simulator independent
  sensor/  isp/      interfaces + ideal pass-throughs
  calibration/       real-camera calibration -> measured catalog entries
  plugins/           simulator plugins, each importing its simulator lazily
    mujoco/          MujocoCameraTwin, render backends, camera mounts, MuJoCo lens scene
    isaac/           IsaacCameraTwin, Isaac lens scene (runs inside Isaac, docker/isaac)
  datasets/          robot-learning environments and datasets
    libero/          LIBERO env wrapper, robots and multi-camera variants, exact replay
    robocasa/        RoboCasa env setup and episode replay
  validation/        renderer vs reference
examples/  tests/ (tests/plugins/, tests/datasets/ mirror the package)  tools/  docs/
```

## Tests

```bash
pytest -q          # everything this environment can run (others skip)
pytest -q -m "not gpu and not isaac and not gl and not libero and not robocasa"   # CI
ruff check . && ruff format --check .
```

Simulator tests need their data, set by environment variables:

| Variable | For |
|---|---|
| `LIBERO_ROOT`, `LIBERO_CONFIG_PATH` | LIBERO tests |
| `ROBOCASA_ASSETS`, `ROBOCASA_DEMOS` | RoboCasa tests (in a robosuite 1.5 environment) |
| `MUJOCO_GL=egl` | headless rendering |

Isaac Sim tests run inside the Isaac container:
`docker/isaac/run.sh -m pytest -q tests/plugins/isaac` (see
[simulators](simulators.md#running-in-docker)).

To develop against a local DeepLens checkout:

```bash
pip install -e ../DeepLens
pip install -e ".[dev]" --no-deps && pip install numpy pyyaml pytest ruff
```

## Status

| Area | State |
|---|---|
| Optics: PSF renderer, distortion, lens-ray renderers (pupil raster, ray cast) | working, validated against DeepLens |
| MuJoCo, LIBERO and RoboCasa adapters | working |
| Camera catalog | Stereolabs ZED X family, `estimated` (datasheet geometry, surrogate lenses) |
| Sensor noise and ISP models | interfaces with ideal pass-throughs |
| Isaac Sim adapter | working (`psf`); lens-ray methods planned |
| Real-camera validation (`measured` catalog entries) | planned |

## Notes on DeepLens

- `calc_focal_plane` is Monte Carlo and unstable for lenses focused near
  infinity, so TwinRobo picks depth layers itself.
- Even PSF kernel sizes are centered half a pixel off, so the default is odd
  (65). For an explicit even size TwinRobo traces one size larger and drops the
  last row and column, keeping the center at index `ks // 2`.

## Regenerating the camera views

The [camera catalog](catalog.md) shows every camera's pre-rendered
views (the site is static). After a change to the catalog or the renderers,
render both scenes and rebuild the assets:

```bash
MUJOCO_GL=egl python tools/docs/playground/render_mujoco.py      # -> outputs/playground
docker/isaac/run.sh tools/docs/playground/render_isaac.py        # -> outputs/isaac/playground
python tools/docs/playground/assemble.py                         # -> docs/assets/playground
```

The [simulator demo](playground.md) is a separate static site,
[TwinRobo-Preview](https://github.com/TwinRobo/TwinRobo-Preview): robot demos
precomputed with TwinRobo Studio, regenerated there (`tools/bake.py`).
