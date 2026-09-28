"""Camera geometry from ChArUco captures: intrinsics, distortion, and stereo extrinsics.

A ChArUco board is a chessboard with ArUco markers in its white squares, so every
detected corner has an identity: partial views, oblique angles and boards at the
image edges all count. That is what a wide-angle lens needs, since the distortion
lives at the edges.

Conventions match `twinrobo.geometry.CameraIntrinsics` and OpenCV: pixel ``(0, 0)``'s
center is ``0``; distortion is OpenCV's ``k1 k2 p1 p2 k3`` on normalized coordinates,
which is what a catalog ``camera.yaml`` stores (``calibration.distortion.model: opencv``).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DICTIONARIES = ("DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250")


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - opencv ships with DeepLens' dependencies
        raise ImportError("calibration needs OpenCV: pip install opencv-python") from e
    return cv2


@dataclass(frozen=True)
class Board:
    """A ChArUco calibration board.

    Args:
        squares_x: Squares across.
        squares_y: Squares down.
        square_m: Printed side of a chessboard square, in meters (measure the print).
        marker_m: Printed side of an ArUco marker, in meters.
        dictionary: OpenCV ArUco dictionary name.
    """

    squares_x: int = 11
    squares_y: int = 8
    square_m: float = 0.015
    marker_m: float = 0.011
    dictionary: str = "DICT_5X5_100"

    def __post_init__(self):
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a ChArUco board needs at least 3 x 3 squares")
        if not 0 < self.marker_m < self.square_m:
            raise ValueError("marker_m must be positive and smaller than square_m")
        if self.dictionary not in DICTIONARIES:
            raise ValueError(f"dictionary must be one of {DICTIONARIES}")

    def cv(self):
        cv2 = _cv2()
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_m, self.marker_m, d
        )

    @property
    def num_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def image(self, px_per_square: int = 120, margin_px: int = 60) -> np.ndarray:
        """The board as a uint8 image for printing (print it flat, then measure a square)."""
        w = self.squares_x * px_per_square + 2 * margin_px
        h = self.squares_y * px_per_square + 2 * margin_px
        return self.cv().generateImage((w, h), marginSize=margin_px)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Board:
        return cls(**d)


def detect(image: np.ndarray, board: Board):
    """ChArUco corners of one image: ``(object_points [N, 3] m, image_points [N, 2] px, ids)``.

    ``None`` if fewer than 6 corners are found (too few to constrain a view).
    """
    cv2 = _cv2()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cvb = board.cv()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(cvb).detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = cvb.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3), img.reshape(-1, 2), ids.reshape(-1)


@dataclass
class GeometryResult:
    """Fitted intrinsics and distortion at the images' resolution.

    ``rms_px``: reprojection error over all corners; below ~0.3 px is a good
    calibration, above ~1 px usually means blurred or mis-measured views.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    rms_px: float
    views: int
    corners: int
    per_view_rms_px: list[float] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def scaled(self, width: int, height: int) -> GeometryResult:
        """The same camera read out at ``width x height`` (e.g. calibrated on a binned mode).

        Distortion acts on normalized coordinates, so only the intrinsics scale.
        """
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > 1e-3 * sx:
            raise ValueError("scaling to another aspect ratio is a crop, not a readout mode")
        return GeometryResult(
            **{
                **asdict(self),
                "width": width,
                "height": height,
                "fx": self.fx * sx,
                "fy": self.fy * sy,
                "cx": (self.cx + 0.5) * sx - 0.5,
                "cy": (self.cy + 0.5) * sy - 0.5,
            }
        )

    def fov_deg(self) -> dict[str, float]:
        """Pinhole field of view (before distortion), horizontal and vertical."""
        return {
            "h": math.degrees(2 * math.atan(self.width / 2 / self.fx)),
            "v": math.degrees(2 * math.atan(self.height / 2 / self.fy)),
        }

    def to_spec(self) -> dict:
        """The ``calibration.intrinsic`` / ``calibration.distortion`` blocks of a camera.yaml."""
        r = lambda v: round(float(v), 6)  # noqa: E731
        return {
            "intrinsic": {"fx": r(self.fx), "fy": r(self.fy), "cx": r(self.cx), "cy": r(self.cy)},
            "distortion": {
                "model": "opencv",
                **{k: r(getattr(self, k)) for k in ("k1", "k2", "p1", "p2", "k3")},
            },
        }


def calibrate(
    images: list[np.ndarray], board: Board, fix_k3: bool = False, min_views: int = 5
) -> GeometryResult:
    """Fit intrinsics and ``k1 k2 p1 p2 k3`` from ChArUco images of one camera.

    Capture 15-40 views: the board filling the frame and in every corner, tilted up
    to ~45 degrees, at the working distance, with the camera's own focus. ``fix_k3``
    helps narrow lenses, where ``k3`` is poorly constrained.
    """
    cv2 = _cv2()
    obj, img = [], []
    size = None
    for im in images:
        h, w = im.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            raise ValueError("all images of a camera must have the same size")
        found = detect(im, board)
        if found is not None:
            obj.append(found[0].astype(np.float32))
            img.append(found[1].astype(np.float32))
    if len(obj) < min_views:
        raise ValueError(f"the board was found in {len(obj)} images; need at least {min_views}")
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    per_view = []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(o, rv, tv, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i) ** 2, 1)))))
    d = np.zeros(5)
    d[: min(5, D.size)] = D.reshape(-1)[:5]
    return GeometryResult(
        width=size[0],
        height=size[1],
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        k1=float(d[0]),
        k2=float(d[1]),
        p1=float(d[2]),
        p2=float(d[3]),
        k3=float(d[4]),
        rms_px=float(rms),
        views=len(obj),
        corners=int(sum(len(o) for o in obj)),
        per_view_rms_px=per_view,
    )


@dataclass
class StereoResult:
    """The right eye's pose in the left eye's camera frame (a module's ``eyes.right``).

    ``translation_m`` and ``rotation_rpy_deg`` are in TwinRobo's camera frame (x right,
    y up, looking along -z), as ``module.yaml`` stores them.
    """

    translation_m: tuple[float, float, float]
    rotation_rpy_deg: tuple[float, float, float]
    baseline_m: float
    rms_px: float
    views: int


def _rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `twinrobo.stereo._rot` (``Rz @ Ry @ Rx``), degrees."""
    ry = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    rx = math.atan2(R[2, 1], R[2, 2])
    rz = math.atan2(R[1, 0], R[0, 0])
    return tuple(math.degrees(v) for v in (rx, ry, rz))


def calibrate_stereo(
    left: list[np.ndarray],
    right: list[np.ndarray],
    board: Board,
    left_geometry: GeometryResult,
    right_geometry: GeometryResult,
) -> StereoResult:
    """The right eye's pose from simultaneous ChArUco pairs (each eye calibrated first)."""
    cv2 = _cv2()
    if len(left) != len(right):
        raise ValueError("left and right need the same number of simultaneous images")
    obj, pl, pr = [], [], []
    for a, b in zip(left, right, strict=True):
        fa, fb = detect(a, board), detect(b, board)
        if fa is None or fb is None:
            continue
        common = np.intersect1d(fa[2], fb[2])
        if len(common) < 6:
            continue
        ia = [int(np.nonzero(fa[2] == c)[0][0]) for c in common]
        ib = [int(np.nonzero(fb[2] == c)[0][0]) for c in common]
        obj.append(fa[0][ia].astype(np.float32))
        pl.append(fa[1][ia].astype(np.float32))
        pr.append(fb[1][ib].astype(np.float32))
    if len(obj) < 3:
        raise ValueError(f"the board was seen by both eyes in {len(obj)} pairs; need at least 3")
    size = (left_geometry.width, left_geometry.height)
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        obj,
        pl,
        pr,
        left_geometry.K,
        left_geometry.dist,
        right_geometry.K,
        right_geometry.dist,
        size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    # OpenCV: x_right = R x_left + T, camera frames x right, y down, z forward. The right
    # eye's pose in the left frame is (R^T, -R^T T); TwinRobo's frame flips y and z.
    F = np.diag([1.0, -1.0, -1.0])
    Rl = R.T
    tl = (-R.T @ T).reshape(3)
    R_tr, t_tr = F @ Rl @ F, F @ tl
    return StereoResult(
        translation_m=tuple(round(float(v), 6) for v in t_tr),
        rotation_rpy_deg=tuple(round(v, 4) for v in _rpy_from_matrix(R_tr)),
        baseline_m=float(np.linalg.norm(t_tr)),
        rms_px=float(rms),
        views=len(obj),
    )


def load_images(paths: list[str | Path]) -> list[np.ndarray]:
    """Images as arrays (grayscale or RGB), in the given order."""
    import imageio.v3 as iio

    return [np.asarray(iio.imread(p)) for p in paths]


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
