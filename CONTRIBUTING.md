# Contributing to TwinRobo

Thanks for helping make simulated cameras match real ones. The most valuable
contributions are **new cameras** for the catalog, but fixes, simulator
adapters, optics and documentation are all welcome.

## Development setup

```bash
git clone https://github.com/TwinRobo/TwinRobo-Core.git && cd TwinRobo-Core
conda create -n twinrobo python=3.12 -y && conda activate twinrobo
pip install -e ".[dev,mujoco]"
pytest -q            # tests that need missing simulators or data skip themselves
ruff check . && ruff format --check .
```

A CUDA GPU makes the optics tests fast; CI runs the CPU subset
(`pytest -m "not gpu and not isaac and not gl and not libero and not robocasa"`).

## Adding a camera

A catalog entry is one camera configuration: one sensor, one lens, one focus.
The same body with a different lens is a different entry.

1. **Pick an ID.** Lowercase, slash-separated, from the vendor down to what
   distinguishes the configuration: `vendor/model/lens` or
   `vendor/model/sensor/lens/fnum`, e.g. `stereolabs/zed-x/2.2mm`.
2. **Create the folder** `twinrobo/catalog/<id>/` with:
   - `camera.yaml`: the CameraSpec. Copy
     `twinrobo/catalog/stereolabs/zed-x/2.2mm/camera.yaml` as a template.
   - `lens.json`: the DeepLens lens file, kept beside the spec (DeepLens does
     not ship its lens library).
3. **Fill in the spec:**
   - `sensor`: resolution, pixel pitch, shutter type.
   - `lens`: focal length, f-number, focus distance, `deeplens_model.path: lens.json`.
   - `calibration.intrinsic` / `calibration.distortion` (OpenCV `k1 k2 p1 p2 k3`)
     when you have them; otherwise geometry follows the lens file.
   - `calibration.psf`: the depth range (`near_m`, `far_m`) the camera will see.
   - `validation.status`: `estimated` (datasheet, surrogate lens), `measured`
     (fitted to captures of a real unit), `verified` or `manufacturer_verified`.
4. **Document provenance** in the file's header comment: datasheet and
   revision, how each number was obtained, and what is approximated. Reviewers
   and users rely on this to know how far to trust the entry.
5. **Lens files and licensing.** Only contribute lens data you may
   redistribute under Apache-2.0: your own designs or measurements, public
   patents, or files whose license allows it. Name the source in the header
   and add an entry to `NOTICE` for third-party material. If the real
   prescription is unpublished, make a surrogate of similar field of view and
   f-number with `tools/catalog/make_surrogate_lens.py` and say so.
6. **Check it:**

   ```bash
   pytest tests/test_catalog.py -q          # every entry must load
   python -c "from twinrobo import CameraTwin; CameraTwin.from_catalog('<id>')"
   ```

   The second command builds the PSF bank with DeepLens (GPU recommended) and
   is the real end-to-end check.

   The documentation's camera catalog page is generated from the catalog
   files, so your camera appears there without editing any docs.

**Stereo products** add a `module.yaml` next to the eye's spec (see
`twinrobo/catalog/stereolabs/zed-x-mini/2.2mm/module.yaml`): the per-eye
specs, the right eye's pose in the left eye's frame (baseline), the housing
size and the default output (`rectified` or `raw`). Products sharing an eye
reference the same `camera.yaml`.

**Improving an entry to `measured`** is just as welcome: replace the intrinsics
and distortion with a calibration of a real unit, fit the focus and blur from
captures (slanted edges or ChArUco boards at several distances, flat fields for
vignetting), and describe the procedure in the header.

## Other contributions

- **Simulator adapters:** the rendering methods are simulator independent
  (lens rays + views + a ray caster). An adapter renders a camera's pinhole
  RGB-D; see `twinrobo/mujoco/camera.py`. Isaac Sim is next on the roadmap.
- **Robots:** new robot variants for LIBERO scenes go in
  `twinrobo/libero/robots.py`.
- **Optics, sensor noise and ISP models:** interfaces live in
  `twinrobo/optics/base.py`, `twinrobo/sensor/base.py` and
  `twinrobo/isp/base.py`.

## Documentation

The site is built with MkDocs; the API reference comes from the docstrings
(Google style: `Args:`, `Returns:`, one entry per parameter).

```bash
pip install -e ".[docs]"
mkdocs serve          # http://127.0.0.1:8000, rebuilds on save
mkdocs build --strict # what CI runs: warnings (e.g. a docstring parameter that
                      # is not in the signature) fail the build
```

## Pull requests

- One topic per pull request, with tests for new behavior.
- `ruff check .`, `ruff format --check .` and `pytest -q` pass.
- Describe what you validated, and on what hardware for performance claims.
- By contributing, you agree that your contribution is licensed under the
  Apache License 2.0 (see `LICENSE`).
