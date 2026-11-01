"""Simulator-independent CameraTwin runtime object.

`CameraTwin` composes optics -> sensor -> ISP for one camera module. It only needs
ideal RGB-D input via `process`; simulator adapters wrap it and render that input
from a simulator camera (`twinrobo.plugins.mujoco.MujocoCameraTwin`,
`twinrobo.plugins.isaac.IsaacCameraTwin`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .frame import CameraFrame, check_depth, check_rgb
from .geometry import CameraIntrinsics
from .isp import IdentityISP, ISPModel, TunedISP, Tuning, load_tuning
from .optics import IdentityOptics, OpticsModel
from .optics.cache import PSFCache
from .optics.dense_depth import DenseDepthRenderer
from .optics.distortion import DistortionWarp, RadialDistortion
from .registry import CatalogRegistry
from .rig import CameraRig
from .sensor import IdealSensor, SensorModel, TunedSensor
from .spec import CameraSpec

#: What selects a camera's sensor and ISP: see `build_sensor_isp`.
ISPSelection = bool | str | Path | ISPModel | None


def build_reference_optics(spec: CameraSpec, device: str | torch.device | None = None):
    """The physical (DeepLens) optics of ``spec``: PSF generation and reference rendering."""
    if spec.lens.deeplens_model is None:
        return None
    from .optics.deeplens import DeepLensOptics

    res = spec.sensor.resolution
    return DeepLensOptics(
        spec.lens.deeplens_model.path,
        sensor_resolution=(res.width, res.height),
        focus_distance_m=spec.lens.focus_distance_m,  # YAML `.inf` = infinity focus
        device=device,
        psf_ks=int(spec.calibration.get("psf", {}).get("ks", 65)),
    )


def psf_bank_params(spec: CameraSpec) -> dict[str, Any]:
    """PSF bank sampling from ``calibration.psf`` (``field_samples`` is ``[grid_w, grid_h]``).

    Defaults: a field grid 16 wide, the height following the sensor's aspect ratio
    (16:9 -> 9, 4:3 -> 12, 16:10 -> 10); 16 depth layers from 0.1 m to infinity.
    """
    psf = spec.calibration.get("psf") or {}
    res = spec.sensor.resolution
    gw, gh = psf.get("field_samples") or (16, max(1, round(16 * res.height / res.width)))
    return {
        "grid": (int(gw), int(gh)),
        "num_depths": int(psf.get("depth_samples", 16)),
        "near_m": float(psf.get("near_m", 0.1)),
        "far_m": float(psf.get("far_m", math.inf)),
        "spp": psf.get("spp"),
    }


def spec_intrinsics(spec: CameraSpec, reference=None) -> CameraIntrinsics | None:
    """Intrinsics from ``calibration.intrinsic`` where set, else from the lens (paraxial)."""
    res = spec.sensor.resolution
    derived = reference.intrinsics() if reference is not None else None
    given = spec.calibration.get("intrinsic") or {}
    if all(given.get(k) is not None for k in ("fx", "fy", "cx", "cy")):
        return CameraIntrinsics(
            res.width, res.height, *(float(given[k]) for k in ("fx", "fy", "cx", "cy"))
        )
    return derived


def build_optics(
    spec: CameraSpec,
    device: str | torch.device | None = None,
    cache: PSFCache | None = None,
    reference: OpticsModel | None = None,
) -> OpticsModel:
    """Build the runtime optics: a `DenseDepthRenderer` over a cached `PSFBank`."""
    reference = reference if reference is not None else build_reference_optics(spec, device)
    if reference is None:
        return IdentityOptics()
    params = psf_bank_params(spec)
    cache = cache if cache is not None else PSFCache()
    key = reference.bank_key(**params)
    bank = cache.get_or_build(
        key,
        lambda: reference.build_psf_bank(**params),
        device=reference.device,
        source=spec.source_path,  # lets deleting the preset release its bank
    )
    return DenseDepthRenderer(bank)


def build_sensor_isp(
    spec: CameraSpec | None,
    isp: ISPSelection = None,
    sensor: SensorModel | None = None,
) -> tuple[SensorModel, ISPModel]:
    """The sensor and ISP that ``isp`` selects for a camera.

    - ``None``: the spec's ``isp.enabled``, so the camera's tuning or the ideal pair.
    - ``False``: an ideal sensor and no ISP.
    - ``True``: the camera's own tuning, from ``isp.tuning`` in its spec.
    - A name or path: that tuning's sensor and ISP (see `twinrobo.isp.load_tuning`), on any
      visible-light color camera. Another camera's tuning is a hybrid, not a model of a real
      camera: this camera's lens with that camera's noise, exposure and color.
    - An `ISPModel`: yours, on ``sensor``, by default an ideal sensor.

    ``sensor`` goes only with an `ISPModel`: a tuning already sets its sensor. Tunings take
    ``analogue_gain``, ``lux``, ``seed``, ``noise``, and for a file ``native_size`` and
    ``bits``, from ``isp`` in the spec.
    """
    if isinstance(isp, ISPModel):
        sensor = sensor if sensor is not None else IdealSensor()
        check_raw_format(sensor, isp)
        return sensor, isp
    if sensor is not None:
        raise ValueError("sensor= goes with an ISPModel; a tuning sets its own sensor")
    cfg = spec.isp if spec is not None else {}
    if isp is None:
        isp = cfg.get("enabled", False)
    if isp is False:
        return IdealSensor(), IdentityISP()
    name = spec.id if spec is not None else "this twin"
    if isp is True:
        isp = cfg.get("tuning")
        if not isp:
            raise ValueError(f"{name}: no isp.tuning in its spec; only tuned cameras have an ISP")
    if spec is not None and (spec.sensor.color != "rgb" or spec.sensor.spectrum != "visible"):
        raise ValueError(f"{name}: the tuned ISP models visible-light color sensors")
    tuning = load_tuning(isp, cfg.get("native_size"), cfg.get("bits"))
    gain = cfg.get("analogue_gain", 2.0)
    return (
        TunedSensor(tuning, gain, cfg.get("seed", 0), cfg.get("noise", True)),
        TunedISP(tuning, gain, cfg.get("lux", 300.0)),
    )


def check_raw_format(sensor: SensorModel, isp: ISPModel) -> None:
    """Raise unless ``isp`` takes the raw image ``sensor`` gives."""
    if sensor.output_format != isp.input_format:
        raise ValueError(
            f"{type(isp).__name__} takes {isp.input_format!r} raw images but "
            f"{type(sensor).__name__} gives {sensor.output_format!r}: pass a matching sensor="
        )


#: Linear-RGB luminance weights (Rec. 709) of a mono sensor.
LUMA = (0.2126, 0.7152, 0.0722)


class CameraTwin:
    """One simulated camera: its lens, sensor and ISP.

    Build one with `from_catalog` or `from_spec`, then `attach` it to a simulator. Call
    `process` directly if you already have RGB and depth images.
    """

    def __init__(
        self,
        spec: CameraSpec | None = None,
        optics: OpticsModel | None = None,
        sensor: SensorModel | None = None,
        isp: ISPModel | None = None,
        intrinsics: CameraIntrinsics | None = None,
        distortion: RadialDistortion | None = None,
        reference: OpticsModel | None = None,
    ):
        self.spec = spec
        self.reference = reference  # physical lens model (DeepLensOptics), for lens-ray rendering
        self.intrinsics = intrinsics
        self.distortion = distortion
        # Distortion is rendered by warping a wider pinhole source render (`render_intrinsics`).
        self.warp = (
            DistortionWarp(intrinsics, distortion)
            if distortion is not None and not distortion.is_identity and intrinsics is not None
            else None
        )
        self.optics = optics if optics is not None else IdentityOptics()
        self.sensor = sensor if sensor is not None else IdealSensor()
        self.isp = isp if isp is not None else IdentityISP()
        self._spec_fingerprint: tuple | None = None
        check_raw_format(self.sensor, self.isp)

    @classmethod
    def from_spec(
        cls,
        path: str | Path | CameraSpec,
        device: str | torch.device | None = None,
        cache: PSFCache | None = None,
        build_psf: bool = True,
        isp: ISPSelection = None,
        *,
        sensor: SensorModel | None = None,
    ) -> CameraTwin:
        """Build a camera from a spec file.

        The first use computes the lens's PSF bank, which is slow; later uses reuse it.

        Args:
            path: The ``camera.yaml`` path, or a loaded `CameraSpec`.
            device: Torch device. By default the one the PSF bank uses.
            cache: Where PSF banks are stored. By default the standard cache.
            build_psf: ``False`` skips the PSF bank, for cameras rendered only by ray casting
                (`twinrobo.plugins.mujoco.lensrender`).
            isp: The sensor and ISP (see `build_sensor_isp`): ``True`` for the camera's own,
                ``False`` for none, a tuning's name, or your own `ISPModel`. `set_isp` changes
                it later.
            sensor: Your sensor, used only with your own `ISPModel`.

        Returns:
            The camera.
        """
        spec = path if isinstance(path, CameraSpec) else CameraSpec.from_yaml(path)
        sensor, isp_model = build_sensor_isp(spec, isp, sensor)
        reference = build_reference_optics(spec, device)
        return cls(
            spec=spec,
            reference=reference,
            sensor=sensor,
            isp=isp_model,
            optics=build_optics(spec, device=device, cache=cache, reference=reference)
            if build_psf
            else IdentityOptics(),
            intrinsics=spec_intrinsics(spec, reference),
            distortion=RadialDistortion.from_dict(spec.calibration.get("distortion")),
        )

    @classmethod
    def from_catalog(
        cls,
        camera_id: str,
        registry: CatalogRegistry | None = None,
        device: str | torch.device | None = None,
        cache: PSFCache | None = None,
        build_psf: bool = True,
        isp: ISPSelection = None,
        *,
        sensor: SensorModel | None = None,
    ) -> CameraTwin:
        """Build a camera from the catalog.

        Args:
            camera_id: The catalog id, such as ``"stereolabs/zed-x/2.2mm"``.
            registry: The catalog to read. By default the bundled one.
            device: Torch device. By default the one the PSF bank uses.
            cache: Where PSF banks are stored. By default the standard cache.
            build_psf: ``False`` skips the PSF bank, which only the ``psf`` rendering method
                needs (ray casting traces the lens).
            isp: The sensor and ISP (see `from_spec`).
            sensor: Your sensor, used only with your own `ISPModel`.

        Returns:
            The camera.
        """
        registry = registry if registry is not None else CatalogRegistry()
        return cls.from_spec(
            registry.load(camera_id),
            device=device,
            cache=cache,
            build_psf=build_psf,
            isp=isp,
            sensor=sensor,
        )

    def set_isp(self, isp: ISPSelection, sensor: SensorModel | None = None) -> None:
        """Switch the sensor and ISP from the next frame, for example between episodes.

        A selection that cannot be used changes nothing. The new pair starts with no state from
        earlier frames and serves every adapter of this twin, so use one twin per ISP to compare
        them side by side. Optics, spec and streams stay.

        To change only the ISP, keep the sensor: ``set_isp(MyISP(), sensor=twin.sensor)``.

        Args:
            isp: The selection, as in `build_sensor_isp`.
            sensor: Your sensor, used only with your own `ISPModel`.
        """
        sensor, isp = build_sensor_isp(self.spec, isp, sensor)
        sensor.reset()
        isp.reset()
        self.sensor, self.isp = sensor, isp

    def reset(self, stream: Any = None) -> None:
        """Clear what earlier frames left in the sensor and ISP, such as auto-exposure.

        Args:
            stream: The stream to clear, the key made by `new_stream`. By default all of them.
        """
        self.sensor.reset(stream)
        self.isp.reset(stream)

    def develop(
        self, rgb_optical: Tensor, exposure: float, metadata: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        """Run the sensor and ISP on the light that reaches the sensor.

        ``metadata["pipeline"]`` records what made the frame. Its ``matches_spec_tuning`` is
        true for the spec's own tuning, false for another, and None when there is nothing to
        compare (no tuning in the spec, or a custom model).

        Args:
            rgb_optical: Linear light at the sensor, ``[B, 3, H, W]``.
            exposure: Exposure scale.
            metadata: The frame's metadata, updated in place.

        Returns:
            ``(raw, rgb)``: the sensor's raw image and the developed image.
        """
        check_raw_format(self.sensor, self.isp)  # the attributes can be set directly
        raw = self.sensor.capture(rgb_optical, exposure, metadata)
        out = self.isp.process(raw, metadata)
        B, _, H, W = rgb_optical.shape
        if (
            out.shape != (B, 3, H, W)
            or not out.is_floating_point()
            or out.device != rgb_optical.device
        ):
            raise ValueError(
                f"{type(self.isp).__name__} returned {out.dtype} {tuple(out.shape)} on "
                f"{out.device}; expected float ({B}, 3, {H}, {W}) on {rgb_optical.device}"
            )
        metadata["pipeline"] = self._pipeline()
        return raw, out.to(rgb_optical.dtype)

    def _pipeline(self) -> dict[str, Any]:
        tunings = [getattr(m, "tuning", None) for m in (self.sensor, self.isp)]
        tunings = [t if isinstance(t, Tuning) else None for t in tunings]
        own = self.spec.isp.get("tuning") if self.spec is not None else None
        matches = None
        if own and all(tunings):
            if self._spec_fingerprint is None:
                try:
                    own = load_tuning(
                        own, self.spec.isp.get("native_size"), self.spec.isp.get("bits")
                    )
                    self._spec_fingerprint = (own.fingerprint, own.bits, own.native_size)
                except (ValueError, OSError):  # a tuning that is not in use need not exist
                    self._spec_fingerprint = ()
            if self._spec_fingerprint:
                matches = all(
                    (t.fingerprint, t.bits, t.native_size) == self._spec_fingerprint
                    for t in tunings
                )
        return {
            "sensor": type(self.sensor).__name__,
            "isp": type(self.isp).__name__,
            "raw_format": self.sensor.output_format,
            "sensor_tuning": tunings[0].name if tunings[0] else None,
            "isp_tuning": tunings[1].name if tunings[1] else None,
            "matches_spec_tuning": matches,
        }

    def sensor_color(self, rgb: Tensor) -> Tensor:
        """The light a sensor of this camera records: luminance for mono sensors.

        A mono sensor has one intensity channel; it is returned in all three
        (``[B, 3, H, W]``, equal channels) so frames keep the RGB convention. Color
        sensors are unchanged.
        """
        if self.spec is None or self.spec.sensor.color != "mono":
            return rgb
        w = rgb.new_tensor(LUMA)[None, :, None, None]
        return (rgb * w).sum(1, keepdim=True).expand_as(rgb).contiguous()

    def lens_geometry(self) -> dict[str, Any]:
        """Keyword arguments of `lens_rays` giving the lens-ray methods this camera's geometry.

        The spec's intrinsics and distortion (``lens.geometry: calibration``, the
        default), so every rendering method has the catalog's field of view and
        distortion. Empty, keeping the traced lens' own geometry, for
        ``lens.geometry: lens``, for ``calibration.distortion.model: physical`` and
        when the spec gives no intrinsics.
        """
        spec = self.spec
        if spec is None or self.intrinsics is None or spec.lens.geometry == "lens":
            return {}
        given = spec.calibration.get("intrinsic") or {}
        if any(given.get(k) is None for k in ("fx", "fy", "cx", "cy")):
            return {}
        if (spec.calibration.get("distortion") or {}).get("model") == "physical":
            return {}
        return {"intrinsics": self.intrinsics, "distortion": self.distortion}

    @property
    def outputs_depth(self) -> bool:
        """Whether the camera's depth output is simulated (``outputs.depth``).

        True without a spec (nothing says otherwise)."""
        return self.spec.outputs.depth if self.spec is not None else True

    @property
    def render_intrinsics(self) -> CameraIntrinsics | None:
        """The pinhole a simulator should render for this twin.

        Without distortion these are the camera intrinsics. With distortion it is a
        wider, centered pinhole with the same central focal length, which `process`
        warps onto the distorted sensor.
        """
        return self.warp.source if self.warp is not None else self.intrinsics

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: Tensor,
        depth: Tensor,
        timestamp: float | None = None,
        exposure: float = 1.0,
        metadata: dict[str, Any] | None = None,
        rectify: bool = False,
        force_depth: bool = False,
    ) -> CameraFrame:
        """Turn a simulator's ideal RGB and depth into this camera's image.

        With distortion, an observation rendered at `render_intrinsics` is warped onto
        the sensor first; ``frame.rgb_ideal`` is then the undistorted pinhole view with
        the camera's intrinsics and ``frame.depth`` the depth seen through the lens.
        Observations of any other size skip the warp (blur only).

        Args:
            rgb: Linear ideal image, ``[B, 3, H, W]``.
            depth: Metric z-depth in meters, ``[B, 1, H, W]``. Always required: the lens blur
                uses it.
            timestamp: Stored on the frame.
            exposure: Exposure scale.
            metadata: Stored on the frame.
            rectify: Undistort the output, as vendor SDKs do by default. ``frame.rgb`` and
                ``frame.depth`` are then pinhole images with the camera's intrinsics, with the
                lens blur kept. No effect without distortion.
            force_depth: Expose ``frame.depth`` even if the camera outputs none
                (`outputs_depth` false): the simulator's ground truth, which the real camera
                would not give. Without it, reading such a frame's depth raises
                `DepthUnavailableError`.

        Returns:
            The `CameraFrame`.
        """
        check_rgb(rgb)
        check_depth(depth, rgb)
        metadata = dict(metadata or {})  # the sensor and ISP report into it
        src = self.warp.source if self.warp is not None else None
        warped = src is not None and tuple(rgb.shape[-2:]) == (src.height, src.width)
        if warped:
            ideal, depth_pinhole = self.warp.pinhole(rgb), self.warp.pinhole(depth, "nearest")
            rgb, depth = self.warp(rgb, depth)
        else:
            ideal = rgb
        rgb_optical = self.sensor_color(self.optics.render(rgb, depth, metadata))
        ideal = self.sensor_color(ideal)
        raw, out = self.develop(rgb_optical, exposure, metadata)
        if rectify and warped:
            out, depth = self.warp.rectify(out), depth_pinhole
        return CameraFrame(
            rgb=out,
            rgb_ideal=ideal,
            rgb_optical=rgb_optical,
            raw=raw,
            depth=depth,
            timestamp=timestamp,
            metadata={"camera_id": self.spec.id if self.spec else None, **metadata},
            has_depth=self.outputs_depth or force_depth,
        )

    def attach(self, backend: Any, camera: str, **options: Any) -> CameraRig:
        """Bind this camera to a simulator camera.

        Args:
            backend: The simulator backend (see `twinrobo.rig`).
            camera: The simulator's camera.
            **options: Passed to the simulator's adapter, such as ``render="raycast"``.

        Returns:
            A `CameraRig`. Its stream is ``"camera"``.
        """
        return CameraRig(backend.camera(self, camera, **options), backend)
