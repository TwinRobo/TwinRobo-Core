"""Calibration from synthetic captures with known answers (no camera needed)."""

import math
import shutil

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from twinrobo.calibration import (  # noqa: E402
    Board,
    EdgeMeasurement,
    calibrate,
    calibrate_stereo,
    flat_field,
    measured_camera_yaml,
    measured_module_yaml,
    slanted_edge_mtf,
)
from twinrobo.calibration.sharpness import mtf_at  # noqa: E402

BOARD = Board(11, 8, 0.015, 0.011)
PPS, MARGIN = 60, 40  # board image: pixels per square, margin
W, H = 640, 400
K = np.array([[400.0, 0, 319.5], [0, 400.0, 199.5], [0, 0, 1]])
D = np.array([-0.12, 0.03, 0.001, -0.0005, 0.0])


@pytest.fixture(scope="module")
def board_img():
    return BOARD.image(PPS, MARGIN)


def undistort(pts, K, D, iters=30):
    """Ideal normalized coordinates of distorted pixels (fixed-point inverse of OpenCV's model)."""
    k1, k2, p1, p2, k3 = D
    xd = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yd = (pts[:, 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / radial, (yd - dy) / radial
    return np.stack([x, y], 1)


def render_view(board_img, rvec, tvec, K=K, D=D):
    """What a camera with ``K, D`` sees of the board at pose ``rvec, tvec`` (OpenCV frame)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x = undistort(np.stack([uu.ravel(), vv.ravel()], 1), K, D)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    n = R[:, 2]
    rays = np.c_[x, np.ones(len(x))]
    s = (n @ t) / (rays @ n)
    B = (rays * s[:, None] - t) @ R  # board coordinates (m): R^T (X - t)
    mx = (MARGIN + B[:, 0] / BOARD.square_m * PPS).reshape(H, W).astype(np.float32)
    my = (MARGIN + B[:, 1] / BOARD.square_m * PPS).reshape(H, W).astype(np.float32)
    bad = (s <= 0).reshape(H, W)
    mx[bad] = my[bad] = -1
    return cv2.remap(board_img, mx, my, cv2.INTER_LINEAR, borderValue=255)


def poses(n=14, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    center = np.array([BOARD.squares_x, BOARD.squares_y, 0]) * BOARD.square_m / 2
    for _ in range(n):
        rvec = rng.uniform(-0.45, 0.45, 3) * np.array([1, 1, 0.3])
        R, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(0.17, 0.3)
        target = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.06, 0.06), z])
        out.append((rvec, target - R @ center))  # board center lands on `target`
    return out


def test_geometry_recovers_intrinsics_and_distortion(board_img):
    imgs = [render_view(board_img, r, t) for r, t in poses()]
    g = calibrate(imgs, BOARD)
    assert g.views >= 10 and g.rms_px < 0.3, (g.views, g.rms_px)
    assert abs(g.fx - 400) < 2 and abs(g.fy - 400) < 2
    assert abs(g.cx - 319.5) < 1.5 and abs(g.cy - 199.5) < 1.5
    assert abs(g.k1 - D[0]) < 0.01 and abs(g.k2 - D[1]) < 0.02
    # the same camera read out at twice the resolution
    g2 = g.scaled(2 * W, 2 * H)
    assert abs(g2.fx - 2 * g.fx) < 1e-9 and abs(g2.cx - ((g.cx + 0.5) * 2 - 0.5)) < 1e-9
    spec = g.to_spec()
    assert spec["distortion"]["model"] == "opencv" and set(spec["intrinsic"]) == {
        "fx",
        "fy",
        "cx",
        "cy",
    }


def test_stereo_recovers_the_baseline(board_img):
    left, right = [], []
    for r, t in poses(10, seed=1):
        left.append(render_view(board_img, r, t))
        right.append(render_view(board_img, r, t - np.array([0.06, 0.0, 0.0])))  # 60 mm right
    gl, gr = calibrate(left, BOARD), calibrate(right, BOARD)
    s = calibrate_stereo(left, right, BOARD, gl, gr)
    assert abs(s.baseline_m - 0.06) < 0.001, s
    assert abs(s.translation_m[0] - 0.06) < 0.001  # right of the left eye (+x image right)
    assert max(abs(v) for v in s.rotation_rpy_deg) < 0.3


def gaussian_edge(sigma_px, angle_deg=5.0, size=128, sub=8):
    """A dark/bright edge tilted ``angle_deg``, blurred by a Gaussian, box-sampled by pixels."""
    n = size * sub
    yy, xx = np.mgrid[0:n, 0:n] / sub
    t = math.tan(math.radians(angle_deg))
    d = (xx - size / 2 - t * (yy - size / 2)) * math.cos(math.atan(t))
    from scipy.special import erf

    v = 0.5 * (1 + erf(d / (math.sqrt(2) * sigma_px)))
    img = 0.1 + 0.8 * v
    return img.reshape(size, sub, size, sub).mean((1, 3))  # pixel aperture


def test_slanted_edge_mtf50_matches_the_analytic_blur():
    sigma = 1.2
    e = slanted_edge_mtf(gaussian_edge(sigma))
    f = np.linspace(0, 1, 20001)
    expected = mtf_at(f, np.exp(-2 * (math.pi * sigma * f) ** 2) * np.abs(np.sinc(f)), 0.5)
    assert e.vertical and abs(e.angle_deg - 5) < 0.3
    assert abs(e.mtf50 - expected) / expected < 0.04, (e.mtf50, expected)
    h = slanted_edge_mtf(gaussian_edge(sigma).T)  # the same edge, near-horizontal
    assert not h.vertical and abs(h.mtf50 - e.mtf50) < 0.01
    with pytest.raises(ValueError, match="tilt"):
        slanted_edge_mtf(gaussian_edge(sigma, angle_deg=0.2))


def test_flat_field_recovers_the_falloff():
    h, w = 300, 480
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / np.hypot(w / 2, h / 2)
    truth = 1 - 0.35 * r**2 + 0.06 * r**4
    rng = np.random.default_rng(0)
    frames = [np.clip(0.7 * truth + rng.normal(0, 0.005, truth.shape), 0, 1) for _ in range(4)]
    v = flat_field(frames)
    assert abs(v.corner - (1 - 0.35 + 0.06)) < 0.01, v
    assert v.rms_residual < 0.01


def test_measured_spec_and_module_are_written(tmp_path, board_img):
    from twinrobo import CameraSpec, CameraTwin
    from twinrobo.registry import BUILTIN_CATALOG
    from twinrobo.stereo import StereoModule

    src = BUILTIN_CATALOG / "stereolabs" / "zed-x" / "2.2mm"
    shutil.copytree(src, tmp_path / "cam")
    g = calibrate([render_view(board_img, r, t) for r, t in poses()], BOARD)
    v = flat_field([np.full((120, 192), 0.5)])
    text = measured_camera_yaml(tmp_path / "cam" / "camera.yaml", g, focus_m=0.55, vignetting=v)
    (tmp_path / "cam" / "camera.yaml").write_text(text)
    assert text.startswith("# CameraSpec") and "# Measured" in text  # provenance kept, extended
    spec = CameraSpec.from_yaml(tmp_path / "cam" / "camera.yaml")
    assert spec.validation.status == "measured" and spec.lens.focus_distance_m == 0.55
    i = spec.calibration["intrinsic"]
    assert i["fx"] == pytest.approx(g.fx * 1920 / W, rel=1e-5)  # rescaled to the spec's 1920 px
    CameraTwin.from_spec(spec, build_psf=False, device="cpu")  # the entry still builds
    from twinrobo.calibration import StereoResult

    s = StereoResult((0.0632, 0.0001, 0.0), (0.0, 0.1, 0.0), 0.0632, 0.2, 12)
    mod = measured_module_yaml(tmp_path / "cam" / "module.yaml", s)
    (tmp_path / "cam" / "module.yaml").write_text(mod)
    m = StereoModule.from_yaml(tmp_path / "cam" / "module.yaml")
    assert m.baseline_m == pytest.approx(0.0632, abs=1e-6)


@pytest.mark.gpu
def test_fit_focus_recovers_a_known_focus():
    pytest.importorskip("deeplens")
    import torch

    from twinrobo import CameraSpec
    from twinrobo.calibration import fit_focus
    from twinrobo.calibration.sharpness import model_mtf
    from twinrobo.camera import build_reference_optics
    from twinrobo.registry import BUILTIN_CATALOG

    spec = CameraSpec.from_yaml(BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml")
    optics = build_reference_optics(spec, device="cuda" if torch.cuda.is_available() else "cpu")
    optics.set_focus(0.8)  # the "real" camera
    ms = []
    for dist in (0.25, 0.5, 2.0, 6.0):
        edges = [EdgeMeasurement(dist, f, []) for f in ((0.0, 0.0), (0.5, 0.0))]
        for e, curve in zip(edges, model_mtf(optics, dist, edges), strict=True):
            e.mtf = list(curve)
        ms += edges
    optics.set_focus(3.0)  # the catalog's guess
    fit = fit_focus(optics, ms)
    assert abs(1 / fit.focus_m - 1 / 0.8) < 0.1, fit.focus_m  # within 0.1 diopter
    assert fit.rms_error < 0.02
