"""A CameraTwin attached to an Isaac Sim camera.

`IsaacCameraTwin` renders what the twin's pinhole sees from a USD camera's pose,
with metric depth, and runs it through the CameraTwin pipeline, like
`twinrobo.plugins.mujoco.MujocoCameraTwin` does for MuJoCo:

- It does not change your camera. It adds a child camera prim
  (``<camera>/TwinRoboView``) that shares the camera's pose and far clip and carries
  the twin's intrinsics, plus a Replicator render product and ``rgb`` /
  ``distance_to_image_plane`` annotators on it. `close` removes all of it.
- With ``match_fov=True`` (default) the child renders the twin's field of view at
  the twin's resolution (for a distorted lens, the wider source pinhole that the
  twin warps onto its sensor, `CameraTwin.render_intrinsics`). With
  ``match_fov=False`` it keeps your camera's lens and only the optics are applied.
- Frames stay on the GPU: annotator data is read as Warp arrays and viewed as torch
  tensors without a copy.
- ``render`` picks the method, as for MuJoCo: ``"psf"`` (pinhole render + PSF
  optics, fast) or ``"raycast"``, which traces every pixel's rays through the real
  lens with exact per-ray visibility against the stage's triangles (see
  `twinrobo.plugins.isaac.lens`).

Needs a running Isaac Sim app (`isaacsim.SimulationApp`) created before this is
used. Import-time safe without Isaac: Isaac modules are imported inside methods.
"""

from __future__ import annotations

import copy
import warnings

import torch

from ...camera import CameraTwin
from ...exceptions import SimulatorError
from ...frame import CameraFrame
from ...isp.color import srgb_to_linear
from ...sensor import new_stream
from .bridge import require_isaac

#: Name of the child camera prim that renders the twin's view.
VIEW_PRIM = "TwinRoboView"
#: `SimulationApp` setting that keeps textures at full resolution, see `IsaacCameraTwin`.
TEXTURE_STREAMING = "/rtx-transient/resourcemanager/enableTextureStreaming"
FULL_TEXTURES_ARG = f"--{TEXTURE_STREAMING}=0"
#: Growth of views that no longer cover a refocused lens. Isaac keeps about 170 MB per render
#: product each time one changes size, so they must rarely change.
VIEW_HEADROOM = 1.1


class IsaacBackend:
    """The simulator backend for Isaac Sim (see `twinrobo.rig`).

    Cameras are USD prim paths. One render fills every camera of a frame, so cameras
    captured together share it.
    """

    capture_options = {"hold_time": True}  # the caller's loop owns the simulation's time
    same_frame = {"step": False}

    def camera(self, twin: CameraTwin, camera: str, **options) -> IsaacCameraTwin:
        return IsaacCameraTwin(twin, camera, **options)

    def lidar(self, spec, **options):
        raise SimulatorError("lidars run in MuJoCo for now")


class IsaacCameraTwin:
    """A CameraTwin rendering from an Isaac Sim (USD) camera prim.

    Texture streaming lowers texture detail without notice when GPU memory runs low, which
    blurs frames. It is read at startup, so `attach` warns when it is on; turn it off with
    ``SimulationApp({"extra_args": [FULL_TEXTURES_ARG]})``.

    Args:
        twin: The CameraTwin (``CameraTwin.from_spec`` / ``from_catalog``).
        camera: Path of a ``UsdGeom.Camera`` prim (or call `attach` later).
        match_fov: Render with the twin's FoV. ``False`` keeps the camera's own lens.
        resolution: ``(W, H)`` render size; only for twins without intrinsics.
        device: Torch device for the CameraTwin pipeline (default: the PSF bank's).
        rectify: Undistort the output like vendor SDKs (see `CameraTwin.process`).
        rt_subframes: RTX subframes per rendered frame (more: less ghosting after
            large motions, slower).
        near_clip_m: Near clipping distance in meters. USD's default near clip is 1
            scene unit (1 m on a meter stage), which would hide what a robot camera
            sees up close, so the view uses this instead of the camera's.
        render: ``"psf"`` or ``"raycast"`` (see the module docs).
        rays_per_pixel: Rays traced per pixel and wavelength (ray cast).
        pupil_views: Shading views rendered across the lens' entrance pupil (ray cast).
        shading: ``"corrected"`` (ISP lens-shading correction) or ``"raw"`` (sensor
            vignetting and cos^4 falloff kept); ray cast.
        view_oversample: Resolution of the pupil views relative to the sensor; above
            1 resolves detail thinner than a pixel (ray cast).
    """

    def __init__(
        self,
        twin: CameraTwin,
        camera: str | None = None,
        match_fov: bool = True,
        resolution: tuple[int, int] | None = None,
        device: str | torch.device | None = None,
        rectify: bool = False,
        rt_subframes: int = 1,
        near_clip_m: float = 0.01,
        render: str = "psf",
        rays_per_pixel: int = 8,
        pupil_views: int = 7,
        shading: str = "corrected",
        view_oversample: float = 1.0,
    ):
        from ...optics.lensrender import check_method

        check_method(render)
        if render != "psf" and getattr(twin, "reference", None) is None:
            raise ValueError(f"render={render!r} needs a CameraTwin with a lens model (DeepLens)")
        if render != "psf" and not match_fov:
            raise ValueError("lens-ray rendering traces the twin's own lens; use match_fov=True")
        self.twin = twin
        self._stream = new_stream()
        self.render_method = render
        self.rays_per_pixel = int(rays_per_pixel)
        self.pupil_views = int(pupil_views)
        self.shading = shading
        self.view_oversample = float(view_oversample)
        self.lens_renderer = None
        self._lens_scene = None
        self.match_fov = match_fov
        self.rectify = rectify
        self.rt_subframes = int(rt_subframes)
        self.near_clip_m = float(near_clip_m)
        if render != "psf":  # the ideal view is the lens' paraxial pinhole of the sensor
            self.lens_renderer = self._renderer()
            self._lens_focus = twin.reference.focus_distance_m
            self._views_stale = False  # refocused, but not rendered at the new view positions yet
            views = self.lens_renderer.views
            self._view_frame = views.width, views.height, views.f
            k = self.lens_renderer.rays.intrinsics
        else:
            k = twin.render_intrinsics if match_fov else twin.intrinsics
        self._pinhole = k
        if k is not None:
            self.resolution = (k.width, k.height)
        elif resolution is not None:
            self.resolution = tuple(int(v) for v in resolution)
        else:
            raise ValueError("twin has no intrinsics; pass resolution=(W, H)")
        if device is None:
            bank = getattr(twin.optics, "bank", None)
            device = bank.device if bank is not None else "cuda"
        self.device = torch.device(device)
        self.camera: str | None = None
        self.view: str | None = None
        self._render_product = None
        self._annotators: dict = {}
        if camera is not None:
            self.attach(camera)

    # -- setup -------------------------------------------------------------------------------
    def attach(self, camera: str) -> None:
        """Render from USD camera ``camera`` (a prim path)."""
        require_isaac()
        import carb
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        if self.camera is not None:
            self.close()
        if carb.settings.get_settings().get(TEXTURE_STREAMING):
            warnings.warn(
                "Isaac's texture streaming is on: frames can lose texture detail when GPU memory "
                'runs low. Start with SimulationApp({"extra_args": [FULL_TEXTURES_ARG]}) '
                "(twinrobo.plugins.isaac.camera).",
                stacklevel=2,
            )
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(camera)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise SimulatorError(f"no UsdGeom.Camera at {camera!r}")
        src = UsdGeom.Camera(prim)
        view = UsdGeom.Camera.Define(stage, f"{camera}/{VIEW_PRIM}")  # identity: same pose
        self._meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        far = src.GetClippingRangeAttr().Get()[1]
        view.GetClippingRangeAttr().Set((self.near_clip_m / self._meters_per_unit, far))
        view.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        if self.match_fov:
            self._set_intrinsics(view, self._pinhole)
        else:  # the camera's own lens
            for name in ("FocalLength", "HorizontalAperture", "VerticalAperture"):
                getattr(view, f"Get{name}Attr")().Set(getattr(src, f"Get{name}Attr")().Get())
        path = view.GetPath().pathString
        self._render_product = rep.create.render_product(path, self.resolution)
        self._annotators = {
            n: rep.AnnotatorRegistry.get_annotator(n, device="cuda")
            for n in ("rgb", "distance_to_image_plane")
        }
        for a in self._annotators.values():
            a.attach([self._render_product])
        self.camera, self.view = camera, path
        self.reset()
        if self.lens_renderer is not None:
            from .lens import IsaacLensScene

            self._lens_scene = IsaacLensScene(camera, self.lens_renderer.views, self.near_clip_m)

    def reset(self) -> None:
        """Start this camera's frames again: its exposure and any ISP state."""
        self.twin.reset(self._stream)

    def _renderer(self):
        """The lens-ray renderer, cached on the twin (it holds the traced rays and views).

        Rays depend on the focus, so a refocused lens gets a new renderer.
        """
        from ...optics.lensrender import LensRayRenderer, lens_rays

        cache = self.twin.__dict__.setdefault("_lens_renderers", {})
        key = (
            self.render_method,
            self.rays_per_pixel,
            self.pupil_views,
            self.shading,
            self.view_oversample,
            self.twin.reference.focus_distance_m,
        )
        if key not in cache:
            cache.clear()  # one lens renderer per twin (they hold large GPU buffers)
            rays = lens_rays(self.twin.reference, self.rays_per_pixel, **self.twin.lens_geometry())
            cache[key] = LensRayRenderer(
                rays,
                self.render_method,
                self.pupil_views,
                shading=self.shading,
                oversample=self.view_oversample,
            )
        return cache[key]

    def _follow_focus(self) -> bool:
        """Pick up a lens refocused since the last frame; True if it was."""
        focus = self.twin.reference.focus_distance_m
        if self.lens_renderer is not None and focus == self._lens_focus:
            return False
        self.lens_renderer = None  # its rays go before the new ones are traced
        # This camera's own views object: other cameras of the twin share the cached renderer,
        # and their render products have their own frame.
        lr = copy.copy(self._renderer())
        lr.views = copy.copy(lr.views)
        self.lens_renderer = lr
        self._pinhole = lr.rays.intrinsics
        if self.view is not None:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            self._set_intrinsics(UsdGeom.Camera(stage.GetPrimAtPath(self.view)), self._pinhole)
        if self._lens_scene is not None:
            if lr.views.match_frame(*self._view_frame):
                self._lens_scene.move_views(lr.views)
            else:
                from .lens import IsaacLensScene

                self._lens_scene.close()
                lr.views.grow(VIEW_HEADROOM)
                self._lens_scene = IsaacLensScene(self.camera, lr.views, self.near_clip_m)
        self._view_frame = lr.views.width, lr.views.height, lr.views.f
        self._lens_focus = focus
        self._views_stale = True
        return True

    @staticmethod
    def _set_intrinsics(view, k) -> None:
        """Give the view pinhole ``k``: focal length / aperture = fx / width (square pixels)."""
        from .lens import set_pinhole

        if k is None:
            raise ValueError("match_fov needs twin intrinsics")
        if abs(k.fx - k.fy) > 1e-3 * k.fx:
            raise SimulatorError(f"non-square pixels (fx {k.fx}, fy {k.fy}) are not supported")
        set_pinhole(view, k.fx, k.width, k.height)

    @property
    def fovy_deg(self) -> float | None:
        if not self.match_fov:
            return None
        return self._pinhole.vfov_deg

    def close(self) -> None:
        """Remove the render product, annotators and the view prim."""
        if self.camera is None:
            return
        import omni.usd

        if self._lens_scene is not None:
            self._lens_scene.close()
            self._lens_scene = None
        for a in self._annotators.values():
            a.detach()
        if self._render_product is not None:
            self._render_product.destroy()
        stage = omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath(self.view).IsValid():
            stage.RemovePrim(self.view)
        self._annotators, self._render_product = {}, None
        self.camera = self.view = None

    # -- frames ------------------------------------------------------------------------------
    def render_ideal(
        self, step: bool = True, hold_time: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The pinhole view: ``(rgb linear [1, 3, H, W], depth m [1, 1, H, W])``, upright.

        ``step`` renders a new frame first (`omni.replicator.core.orchestrator.step`, with
        Replicator's default time step). Pass ``False`` if your loop already rendered this
        frame, e.g. with ``world.step(render=True)``. ``hold_time`` renders without advancing
        the timeline.
        """
        if self.camera is None:
            raise SimulatorError("call attach(camera) first")
        import warp as wp

        if step:
            self._step(0.0 if hold_time else None)
        rgba = self._annotators["rgb"].get_data(device="cuda")
        dist = self._annotators["distance_to_image_plane"].get_data(device="cuda")
        if rgba is None or len(rgba.shape) != 3:
            raise SimulatorError("the renderer has not produced a frame yet; step once more")
        W, H = self.resolution
        rgb = wp.to_torch(rgba)[..., :3].to(self.device).permute(2, 0, 1)[None].float() / 255.0
        depth = wp.to_torch(dist).to(self.device).reshape(1, 1, H, W).float()
        return srgb_to_linear(rgb), depth * self._meters_per_unit

    def _step(self, delta_time: float | None = None) -> None:
        """Render one frame of every render product (main view and the views across the pupil).

        ``delta_time=0.0`` holds the timeline; ``None`` is Replicator's default.
        """
        import omni.replicator.core as rep

        step = {"rt_subframes": self.rt_subframes, "pause_timeline": False}
        if delta_time is not None:
            step["delta_time"] = delta_time
        rep.orchestrator.step(**step)
        for _ in range(20):  # render products added since the last step need a frame first
            if self._lens_scene is None or self._lens_scene.ready():
                return
            rep.orchestrator.step(**step)
        raise SimulatorError("the shading views across the pupil produced no frame")

    def get_frame(
        self,
        timestamp: float | None = None,
        step: bool = True,
        force_depth: bool = False,
        hold_time: bool = False,
    ) -> CameraFrame:
        """Render one frame through the camera.

        Args:
            timestamp: Stored on the frame.
            step: Render a new frame first (see `render_ideal`). Pass ``False`` if your loop
                already rendered this frame.
            hold_time: Render without advancing the timeline.
            force_depth: Make ``frame.depth`` readable even if the camera's spec has no depth
                output: the simulator's ground truth.

        Returns:
            The `CameraFrame`. ``frame.rgb`` is linear ``[1, 3, H, W]``.
        """
        if self.render_method != "psf":
            from ...optics.lensrender import lens_frame

            self._follow_focus()
            if self._views_stale and not step:
                self._step(delta_time=0.0)  # the moved views need a render; the caller's time holds
            ideal, _ = self.render_ideal(step, hold_time)
            self._views_stale = False
            rgb, depth = self.lens_renderer.render_scene(self._lens_scene)
            return lens_frame(
                self.twin,
                self.lens_renderer,
                rgb,
                depth,
                ideal.to(rgb.device),
                self.rectify,
                timestamp,
                {"camera": self.camera, "stream": self._stream},
                force_depth=force_depth,
            )
        rgb, depth = self.render_ideal(step, hold_time)
        return self.twin.process(
            rgb,
            depth,
            timestamp=timestamp,
            metadata={"camera": self.camera, "stream": self._stream, "fovy_deg": self.fovy_deg},
            rectify=self.rectify,
            force_depth=force_depth,
        )

    def __repr__(self) -> str:
        W, H = self.resolution
        fov = f"vfov {self.fovy_deg:.1f} deg" if self.fovy_deg else "the camera's own lens"
        return f"IsaacCameraTwin({self.camera!r}, {W}x{H}, {fov}, render={self.render_method!r})"
