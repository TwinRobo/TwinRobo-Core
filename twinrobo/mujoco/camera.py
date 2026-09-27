"""MuJoCo camera -> CameraTwin frames."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..camera import CameraTwin
from ..exceptions import SimulatorError
from ..frame import CameraFrame
from ..isp.color import linear_to_srgb, srgb_to_linear


class RenderBackend(Protocol):
    """Renders one MuJoCo camera. Returns images **upright** (row 0 = top).

    ``render`` returns ``(rgb uint8 [H, W, 3], depth float32 [H, W])``, with depth
    as metric z-depth in meters.
    """

    model: Any  # mujoco.MjModel

    def render(self, camera: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]: ...


class MujocoRenderer:
    """``mujoco.Renderer`` backend for plain MuJoCo (``MjModel``/``MjData``) scenes.

    Do not use it inside robosuite/LIBERO: a second GL context in the same
    process corrupts robosuite's own offscreen renders (use `RobosuiteRenderer`).
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._renderers: dict[tuple[int, int], Any] = {}

    def render(self, camera: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        r = self._renderers.get((width, height))
        if r is None:
            vis = self.model.vis.global_
            vis.offwidth, vis.offheight = max(vis.offwidth, width), max(vis.offheight, height)
            r = self._renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        depth = r.render().astype(np.float32, copy=True)  # metric z-depth
        r.disable_depth_rendering()
        return rgb, depth

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


class RobosuiteRenderer:
    """robosuite backend: renders in robosuite's own GL context.

    Pass the robosuite env (or anything with ``.sim``) rather than the sim:
    robosuite may rebuild ``env.sim`` on reset, so it is looked up on every call.
    """

    def __init__(self, env):
        self.env = env

    @property
    def sim(self):
        return self.env.sim

    @property
    def model(self):
        return self.sim.model._model

    @property
    def data(self):
        return self.sim.data._data

    @property
    def vopt(self):
        """robosuite's render option (which geom groups it draws); ``None`` before first render."""
        ctx = getattr(self.sim, "_render_context_offscreen", None)
        return getattr(ctx, "vopt", None)

    def render(self, camera: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        from robosuite.utils.camera_utils import get_real_depth_map

        sim = self.sim
        rgb, zbuf = sim.render(camera_name=camera, width=width, height=height, depth=True)
        depth = get_real_depth_map(sim, zbuf)
        # robosuite returns OpenGL-convention (bottom-up) images.
        return rgb[::-1].copy(), np.asarray(depth[::-1], dtype=np.float32).copy()

    def close(self) -> None:
        pass


@contextmanager
def _camera_fovy(model, cam_id: int, fovy_deg: float | None):
    """Temporarily set a camera's vertical FoV; always restore it (the model is shared)."""
    if fovy_deg is None:
        yield
        return
    old = float(model.cam_fovy[cam_id])
    model.cam_fovy[cam_id] = fovy_deg
    try:
        yield
    finally:
        model.cam_fovy[cam_id] = old


def to_uint8(
    rgb_linear: Tensor, size: tuple[int, int] | None = None, crop: str = "center"
) -> np.ndarray:
    """Encode a linear ``[1, 3, H, W]`` frame for consumers: sRGB, uint8 ``[h, w, 3]``, upright.

    Args:
        size: Output ``(width, height)``. ``None`` keeps the full frame.
        crop: ``"center"`` first crops the largest centered region with the
            output aspect ratio (no stretching). ``"none"`` resizes the full
            frame (stretches if the aspect ratios differ).
    """
    x = linear_to_srgb(rgb_linear.float())
    if size is not None:
        w_out, h_out = size
        H, W = x.shape[-2:]
        if crop == "center":
            if W * h_out > H * w_out:  # too wide
                w = round(H * w_out / h_out)
                x = x[..., :, (W - w) // 2 : (W - w) // 2 + w]
            else:
                h = round(W * h_out / w_out)
                x = x[..., (H - h) // 2 : (H - h) // 2 + h, :]
        elif crop != "none":
            raise ValueError(f"crop must be 'center' or 'none', got {crop!r}")
        if x.shape[-2:] != (h_out, w_out):
            x = F.interpolate(
                x, size=(h_out, w_out), mode="bilinear", antialias=True, align_corners=False
            )
    return (x[0].clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


class MujocoCameraTwin:
    """A CameraTwin attached to a MuJoCo camera.

    The camera keeps its pose. While rendering, its vertical FoV is set to the
    twin's (``match_fov=True``) and the image is rendered at the twin's
    resolution. The model is restored afterwards. For a twin with lens
    distortion this is the twin's wider source pinhole (`CameraTwin.render_intrinsics`),
    which the twin warps onto its sensor. With ``match_fov=False`` the scene
    camera's own pinhole geometry is kept and only blur is applied.

    Args:
        twin: The CameraTwin (``CameraTwin.from_spec``). Needs ``intrinsics``
            unless ``resolution`` is given.
        backend: `MujocoRenderer` or `RobosuiteRenderer`.
        camera: MuJoCo camera name (or call `attach` later).
        match_fov: Render with the twin's FoV. ``False`` keeps the scene's fovy.
        resolution: ``(W, H)`` render size. Only for twins without intrinsics.
        device: Torch device for the CameraTwin pipeline.
        rectify: Undistort the output like vendor SDKs (see `CameraTwin.process`).
        render: ``"psf"`` (pinhole render + PSF optics, fast), or trace every pixel
            through the real lens: ``"pupil"`` (pupil-sampled rasterization) or
            ``"raycast"`` (per-ray ray casting). See `twinrobo.mujoco.lensrender`.
        rays_per_pixel: Rays traced per pixel and wavelength (lens-ray methods).
        pupil_views: Views rendered across the lens' entrance pupil (lens-ray methods).
        shading: ``"corrected"`` (ISP lens-shading correction) or ``"raw"`` (sensor
            vignetting and cos^4 falloff kept); lens-ray methods.
        view_oversample: Resolution of the pupil views relative to the sensor; above
            1 resolves detail thinner than a pixel (lens-ray methods).
    """

    def __init__(
        self,
        twin: CameraTwin,
        backend: RenderBackend,
        camera: str | None = None,
        match_fov: bool = True,
        resolution: tuple[int, int] | None = None,
        device: str | torch.device | None = None,
        rectify: bool = False,
        render: str = "psf",
        rays_per_pixel: int = 8,
        pupil_views: int = 7,
        shading: str = "corrected",
        view_oversample: float = 1.0,
    ):
        from .lensrender import METHODS

        if render not in METHODS:
            raise ValueError(f"render must be one of {METHODS}, got {render!r}")
        if render != "psf" and getattr(twin, "reference", None) is None:
            raise ValueError(f"render={render!r} needs a CameraTwin with a lens model (DeepLens)")
        self.twin = twin
        self.rectify = rectify
        self.render_method = render
        self.rays_per_pixel = int(rays_per_pixel)
        self.pupil_views = int(pupil_views)
        self.shading = shading
        self.view_oversample = float(view_oversample)
        self.lens_renderer = None
        self.backend = backend
        self.match_fov = match_fov
        render = twin.render_intrinsics if match_fov else twin.intrinsics
        if render is not None:
            self.resolution = (render.width, render.height)
        elif resolution is not None:
            self.resolution = tuple(resolution)
        else:
            raise ValueError("twin has no intrinsics; pass resolution=(W, H)")
        if device is None:
            bank = getattr(twin.optics, "bank", None)
            device = (
                bank.device
                if bank is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.camera: str | None = None
        self._cam_id: int | None = None
        if camera is not None:
            self.attach(camera)

    def attach(self, camera: str) -> None:
        import mujoco

        cam_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise SimulatorError(f"MuJoCo camera {camera!r} not found")
        if self.match_fov and np.any(self.backend.model.cam_sensorsize[cam_id] > 0):
            raise SimulatorError(
                f"camera {camera!r} uses sensorsize/focal intrinsics; "
                "fovy override is not supported"
            )
        self.camera, self._cam_id = camera, cam_id

    @property
    def fovy_deg(self) -> float | None:
        if not self.match_fov:
            return None
        if self.twin.render_intrinsics is None:
            raise ValueError("match_fov needs twin intrinsics")
        return self.twin.render_intrinsics.vfov_deg

    def render_ideal(self) -> tuple[np.ndarray, np.ndarray]:
        """Ideal render at the twin's resolution and FoV: ``(rgb uint8, depth m)``, upright."""
        if self.camera is None:
            raise SimulatorError("call attach(camera) first")
        W, H = self.resolution
        with _camera_fovy(self.backend.model, self._cam_id, self.fovy_deg):
            return self.backend.render(self.camera, W, H)

    def get_frame(self, timestamp: float | None = None, status=None) -> CameraFrame:
        """Render and process one frame. ``frame.rgb`` is linear ``[1, 3, H, W]`` on ``device``."""
        if self.render_method != "psf":
            return self._lens_frame(timestamp, status)
        rgb8, depth = self.render_ideal()
        rgb = torch.from_numpy(rgb8).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        rgb = srgb_to_linear(rgb)
        depth_t = torch.from_numpy(depth).to(self.device)[None, None]
        return self.twin.process(
            rgb,
            depth_t,
            timestamp=timestamp,
            metadata={"camera": self.camera, "fovy_deg": self.fovy_deg},
            rectify=self.rectify,
        )

    # -- lens-ray methods (A: pupil raster, B: ray cast) ------------------------------------
    def _lens_frame(self, timestamp, status) -> CameraFrame:
        from .lensrender import LensRayRenderer, lens_rays

        if self.camera is None:
            raise SimulatorError("call attach(camera) first")
        if self.lens_renderer is None:
            # Cached on the twin: pupil views and ray-to-cell assignment are per lens config.
            cache = self.twin.__dict__.setdefault("_lens_renderers", {})
            key = (
                self.render_method,
                self.rays_per_pixel,
                self.pupil_views,
                self.shading,
                self.view_oversample,
            )
            if key not in cache:
                rays = lens_rays(self.twin.reference, self.rays_per_pixel, status=status)
                cache.clear()  # one lens renderer per twin (they hold large GPU buffers)
                cache[key] = LensRayRenderer(
                    rays,
                    self.render_method,
                    self.pupil_views,
                    shading=self.shading,
                    oversample=self.view_oversample,
                )
            self.lens_renderer = cache[key]
        lr = self.lens_renderer
        rgb, depth = lr.render(self.backend, self.camera, self._cam_id)
        # The pinhole reference: the paraxial camera of the same sensor.
        K = lr.rays.intrinsics
        fovy = math.degrees(2 * math.atan(K.height / 2 / K.fy))
        with _camera_fovy(self.backend.model, self._cam_id, fovy):
            ideal8, _ = self.backend.render(self.camera, K.width, K.height)
        ideal = srgb_to_linear(
            torch.from_numpy(ideal8).to(rgb.device).permute(2, 0, 1)[None].float() / 255.0
        )
        from ..optics.lensrender import lens_frame

        return lens_frame(
            self.twin, lr, rgb, depth, ideal, self.rectify, timestamp, {"camera": self.camera}
        )
