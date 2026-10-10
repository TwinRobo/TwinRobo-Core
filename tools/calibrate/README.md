# tools/calibrate

Calibration lives in the package: `python -m twinrobo.calibration --help`, or the
[calibration guide](https://twinrobo.github.io/calibration/). It fits a real unit's intrinsics,
distortion, focus and vignetting (and a stereo module's baseline) from ChArUco,
slanted-edge and flat-field captures, and writes a `measured` catalog entry.
