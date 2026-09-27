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
  sensor/  isp/      interfaces + ideal pass-throughs
  mujoco/            MuJoCo adapter, lens-ray renderers (pupil raster, ray cast), camera mounts
  libero/            LIBERO env wrapper, robots, exact replay
  robocasa/          RoboCasa env setup and episode replay
  isaac/             Isaac Sim adapter (IsaacCameraTwin), imports Isaac lazily
  validation/        renderer vs reference
examples/  tests/  tools/  docs/
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
`docker/isaac/run.sh -m pytest -q tests/isaac` (see
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
- Even PSF kernel sizes are centered half a pixel off, so TwinRobo uses odd
  kernel sizes (default 65).
