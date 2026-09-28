"""Calibrate a real camera into a ``measured`` catalog entry.

Steps (each adds to one results file; see ``python -m twinrobo.calibration --help``):

1. **Geometry** (`geometry`): ChArUco views -> intrinsics and OpenCV distortion;
   for stereo products, the right eye's pose (`calibrate_stereo`).
2. **Sharpness** (`sharpness`): slanted edges at known distances -> MTF50 per edge;
   `fit_focus` finds the focus distance the lens model needs to match them.
3. **Vignetting** (`vignetting`): flat fields -> radial relative illumination.
4. **Write** (`spec_writer`): the camera.yaml / module.yaml with the measured values
   and ``validation.status: measured``.
"""

from .geometry import Board, GeometryResult, StereoResult, calibrate, calibrate_stereo, detect
from .sharpness import EdgeMeasurement, EdgeMTF, FocusFit, fit_focus, psf_mtf, slanted_edge_mtf
from .spec_writer import measured_camera_yaml, measured_module_yaml
from .vignetting import Vignetting, flat_field

__all__ = [
    "Board",
    "EdgeMTF",
    "EdgeMeasurement",
    "FocusFit",
    "GeometryResult",
    "StereoResult",
    "Vignetting",
    "calibrate",
    "calibrate_stereo",
    "detect",
    "fit_focus",
    "flat_field",
    "measured_camera_yaml",
    "measured_module_yaml",
    "psf_mtf",
    "slanted_edge_mtf",
]
