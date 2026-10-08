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

import torch

from ...camera import CameraTwin
from ...exceptions import SimulatorError
from ...frame import CameraFrame
from ...isp.color import srgb_to_linear
from .bridge import require_isaac

#: Name of the child camera prim that renders the twin's view.
VIEW_PRIM = "TwinRoboView"


class IsaacCameraTwin:
    """A CameraTwin rendering from an Isaac Sim (USD) camera prim.

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
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        if self.camera is not None:
            self.close()
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
        if self.lens_renderer is not None:
            from .lens import IsaacLensScene

            self._lens_scene = IsaacLensScene(camera, self.lens_renderer.views, self.near_clip_m)

    def _renderer(self):
        """The lens-ray renderer, cached on the twin (it holds the traced rays and views)."""
        from ...optics.lensrender import LensRayRenderer, lens_rays

        cache = self.twin.__dict__.setdefault("_lens_renderers", {})
        key = (
            self.render_method,
            self.rays_per_pixel,
            self.pupil_views,
            self.shading,
            self.view_oversample,
        )
        if key not in cache:
            rays = lens_rays(self.twin.reference, self.rays_per_pixel, **self.twin.lens_geometry())
            cache.clear()  # one lens renderer per twin (they hold large GPU buffers)
            cache[key] = LensRayRenderer(
                rays,
                self.render_method,
                self.pupil_views,
                shading=self.shading,
                oversample=self.view_oversample,
            )
        return cache[key]

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
    def render_ideal(self, step: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """The pinhole view: ``(rgb linear [1, 3, H, W], depth m [1, 1, H, W])``, upright.

        ``step`` renders a new frame first (`omni.replicator.core.orchestrator.step`,
        without advancing the timeline). Pass ``False`` if your loop already rendered
        this frame, e.g. with ``world.step(render=True)``.
        """
        if self.camera is None:
            raise SimulatorError("call attach(camera) first")
        import warp as wp

        if step:
            self._step()
        rgba = self._annotators["rgb"].get_data(device="cuda")
        dist = self._annotators["distance_to_image_plane"].get_data(device="cuda")
        if rgba is None or len(rgba.shape) != 3:
            raise SimulatorError("the renderer has not produced a frame yet; step once more")
        W, H = self.resolution
        rgb = wp.to_torch(rgba)[..., :3].to(self.device).permute(2, 0, 1)[None].float() / 255.0
        depth = wp.to_torch(dist).to(self.device).reshape(1, 1, H, W).float()
        return srgb_to_linear(rgb), depth * self._meters_per_unit

    def _step(self) -> None:
        """Render one frame of every render product (main view and the views across the pupil)."""
        import omni.replicator.core as rep

        rep.orchestrator.step(rt_subframes=self.rt_subframes, pause_timeline=False)
        for _ in range(20):  # render products added since the last step need a frame first
            if self._lens_scene is None or self._lens_scene.ready():
                return
            rep.orchestrator.step(rt_subframes=self.rt_subframes, pause_timeline=False)
        raise SimulatorError("the shading views across the pupil produced no frame")

    def get_frame(
        self, timestamp: float | None = None, step: bool = True, force_depth: bool = False
    ) -> CameraFrame:
        """Render and process one frame. ``frame.rgb`` is linear ``[1, 3, H, W]`` on ``device``.

        ``frame.depth`` is readable only if the camera's depth output is simulated
        (``outputs.depth`` in its spec) or with ``force_depth=True`` (simulator ground truth).
        """
        if self.lens_renderer is not None:
            from ...optics.lensrender import lens_frame

            ideal, _ = self.render_ideal(step)
            rgb, depth = self.lens_renderer.render_scene(self._lens_scene)
            return lens_frame(
                self.twin,
                self.lens_renderer,
                rgb,
                depth,
                ideal.to(rgb.device),
                self.rectify,
                timestamp,
                {"camera": self.camera},
                force_depth=force_depth,
            )
        rgb, depth = self.render_ideal(step)
        return self.twin.process(
            rgb,
            depth,
            timestamp=timestamp,
            metadata={"camera": self.camera, "fovy_deg": self.fovy_deg},
            rectify=self.rectify,
            force_depth=force_depth,
        )

    def __repr__(self) -> str:
        W, H = self.resolution
        fov = f"vfov {self.fovy_deg:.1f} deg" if self.fovy_deg else "the camera's own lens"
        return f"IsaacCameraTwin({self.camera!r}, {W}x{H}, {fov}, render={self.render_method!r})"
