# Calibrate a real camera

Catalog entries start as `estimated`: geometry fitted to the datasheet, blur from
a surrogate lens design. `twinrobo.calibration` turns captures of a real unit
into a `measured` entry: its own intrinsics and distortion, the focus distance
its sharpness implies, its vignetting, and for stereo products the real baseline
and eye rotation.

You need the camera, a printed ChArUco board, a slanted-edge target (a black
square printed or cut with straight, sharp edges) and an evenly lit white
surface. Every step reads and updates one results file (`calib.json`).

## 1. Print a board

```bash
python -m twinrobo.calibration board --out board.png     # 11 x 8 squares, 15 mm, DICT_5X5_100
```

Print it flat at 100 % scale on stiff card, measure one square with calipers, and
correct `square_m` / `marker_m` in `board.json` if it is not 15 mm.

## 2. Geometry: intrinsics and distortion

Capture 15-40 views at the camera's native resolution, with its own focus: the
board filling the frame and in every corner (wide lenses need the corners), tilted
up to ~45 degrees, sharp (no motion blur).

```bash
python -m twinrobo.calibration geometry views/*.png --board board.json
```

The fit reports the reprojection RMS: below ~0.3 px is good; above 1 px usually
means blurred views (the worst ones are listed). Narrow lenses can add `--fix-k3`.

**Stereo products:** capture simultaneous left/right pairs (the board visible in
both) and run `stereo` instead; it calibrates each eye and then the right eye's
pose, which gives the unit's real baseline:

```bash
python -m twinrobo.calibration stereo --left L/*.png --right R/*.png --board board.json
```

## 3. Sharpness and focus

Photograph the slanted-edge target at 3-5 distances spanning the working range
(in front of and behind where you expect focus), edges tilted 2-15 degrees off
vertical or horizontal, at the center and towards the edges of the frame. For
each distance, list the edge regions (`x,y,w,h` in pixels):

```bash
python -m twinrobo.calibration edges near/*.png --distance-m 0.3 --roi 880,520,160,160 --roi 1500,520,160,160
python -m twinrobo.calibration edges far/*.png  --distance-m 2.0 --roi 880,520,160,160
python -m twinrobo.calibration focus --spec twinrobo/catalog/<id>/camera.yaml
```

`edges` measures each edge's MTF (ISO 12233 slanted-edge method); `focus` refocuses
the entry's lens model (DeepLens) to candidate distances and keeps the one whose
MTF matches the measurements best, printing measured vs model MTF50 per edge.
Large, consistent mismatches mean the surrogate lens is sharper or softer than the
real one: note it in the entry.

Use unprocessed frames where the camera allows (no sharpening, lowest
compression); sharpening in the camera's ISP makes edges look sharper than the
lens is.

## 4. Vignetting

Cover the lens with a diffuser (or face an evenly lit white wall, slightly
defocused), expose to roughly half of full scale, and capture a few frames:

```bash
python -m twinrobo.calibration flatfield flat/*.png
```

The result is a radial falloff `1 + a2 r^2 + a4 r^4 + a6 r^6` (r = 1 at the corners)
and the corner level.

## 5. Write the entry

```bash
python -m twinrobo.calibration write --spec twinrobo/catalog/<id>/camera.yaml \
    --out twinrobo/catalog/<id>/camera.yaml [--module twinrobo/catalog/<id>/module.yaml] \
    --notes "Unit S/N 12345, captured 2026-10-01 by <you>; ISP sharpening off"
```

The camera.yaml gets the measured intrinsics, distortion, focus and vignetting and
`validation.status: measured`; its header keeps the original provenance and adds a
*Measured* section with the numbers above. Results measured at another resolution
(e.g. a binned mode) are rescaled to the entry's. With `--module`, the stereo
module gets the unit's baseline and eye rotation. Check the entry, then open a
pull request (see [Contributing](contributing.md#adding-a-camera)).

## From Python

```python
from twinrobo.calibration import Board, calibrate, slanted_edge_mtf, flat_field

g = calibrate(images, Board(square_m=0.015, marker_m=0.011))
print(g.rms_px, g.to_spec())            # calibration.intrinsic / distortion blocks
print(slanted_edge_mtf(roi).mtf50)      # cycles per pixel
```

See the [API reference](api/calibration.md).
