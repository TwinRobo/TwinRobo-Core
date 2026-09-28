"""Simulator-independent CameraTwin runtime object.

`CameraTwin` composes optics -> sensor -> ISP for one camera module. It only needs
ideal RGB-D input via `process`; simulator adapters wrap it and render that input
from a simulator camera (`twinrobo.plugins.mujoco.MujocoCameraTwin`,
`twinrobo.plugins.isaac.IsaacCameraTwin`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .frame import CameraFrame, check_depth, check_rgb
from .geometry import CameraIntrinsics
from .isp import IdentityISP, ISPModel
from .optics import IdentityOptics, OpticsModel
from .optics.cache import PSFCache
from .optics.dense_depth import DenseDepthRenderer
from .optics.distortion import DistortionWarp, RadialDistortion
from .registry import CatalogRegistry
from .sensor import IdealSensor, SensorModel
from .spec import CameraSpec


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
    """PSF bank sampling from ``calibration.psf`` (``field_samples`` is ``[grid_w, grid_h]``)."""
    psf = spec.calibration.get("psf") or {}
    gw, gh = psf.get("field_samples", [17, 11])
    return {
        "grid": (int(gw), int(gh)),
        "num_depths": int(psf.get("depth_samples", 16)),
        "near_m": float(psf.get("near_m", 0.3)),
        "far_m": float(psf.get("far_m", 20.0)),
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


#: Linear-RGB luminance weights (Rec. 709) of a mono sensor.
LUMA = (0.2126, 0.7152, 0.0722)


class CameraTwin:
    """A deployed camera configuration: optics, sensor and ISP."""

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

    @classmethod
    def from_spec(
        cls,
        path: str | Path | CameraSpec,
        device: str | torch.device | None = None,
        cache: PSFCache | None = None,
        build_psf: bool = True,
    ) -> CameraTwin:
        """Load a spec; builds and caches its PSF bank on first use (slow), then reuses it.

        ``build_psf=False`` skips the PSF bank (identity optics), for twins rendered
        only with the lens-ray methods (`twinrobo.plugins.mujoco.lensrender`).
        """
        spec = path if isinstance(path, CameraSpec) else CameraSpec.from_yaml(path)
        reference = build_reference_optics(spec, device)
        return cls(
            spec=spec,
            reference=reference,
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
    ) -> CameraTwin:
        """A catalog camera (``vendor/model/...``). ``build_psf=False`` skips the PSF bank,
        which only the ``psf`` rendering method needs (the lens-ray methods trace the lens)."""
        registry = registry if registry is not None else CatalogRegistry()
        return cls.from_spec(
            registry.load(camera_id), device=device, cache=cache, build_psf=build_psf
        )

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
        """Turn an ideal simulator RGB-D observation into a CameraTwin frame.

        With distortion, an observation rendered at `render_intrinsics` is warped onto
        the sensor first; ``frame.rgb_ideal`` is then the undistorted pinhole view with
        the camera's intrinsics and ``frame.depth`` the depth seen through the lens.
        Observations of any other size skip the warp (blur only).

        ``rectify``: undistort the output (as vendor SDKs do by default): ``frame.rgb``
        and ``frame.depth`` are then pinhole images with the camera's intrinsics, the
        lens blur kept. No effect without distortion.

        ``force_depth``: expose ``frame.depth`` even if the camera outputs none
        (`outputs_depth` false): the simulator's ground truth, which the real camera
        would not give. Without it, reading such a frame's depth raises
        `DepthUnavailableError`. ``depth`` is still required: the lens blur uses it.
        """
        check_rgb(rgb)
        check_depth(depth, rgb)
        src = self.warp.source if self.warp is not None else None
        warped = src is not None and tuple(rgb.shape[-2:]) == (src.height, src.width)
        if warped:
            ideal, depth_pinhole = self.warp.pinhole(rgb), self.warp.pinhole(depth, "nearest")
            rgb, depth = self.warp(rgb, depth)
        else:
            ideal = rgb
        rgb_optical = self.sensor_color(self.optics.render(rgb, depth, metadata))
        ideal = self.sensor_color(ideal)
        raw = self.sensor.capture(rgb_optical, exposure, metadata)
        out = self.isp.process(raw, metadata)
        if rectify and warped:
            out, depth = self.warp.rectify(out), depth_pinhole
        return CameraFrame(
            rgb=out,
            rgb_ideal=ideal,
            rgb_optical=rgb_optical,
            raw=raw,
            depth=depth,
            timestamp=timestamp,
            metadata={"camera_id": self.spec.id if self.spec else None, **(metadata or {})},
            has_depth=self.outputs_depth or force_depth,
        )

    # Rendering from a simulator camera is done by adapters that wrap the twin.
    def attach(self, simulator_camera: Any) -> None:
        raise NotImplementedError(
            "wrap the twin in a simulator adapter: twinrobo.plugins.mujoco.MujocoCameraTwin "
            "or twinrobo.plugins.isaac.IsaacCameraTwin"
        )

    def get_frame(self) -> CameraFrame:
        raise NotImplementedError(
            "wrap the twin in a simulator adapter: twinrobo.plugins.mujoco.MujocoCameraTwin "
            "or twinrobo.plugins.isaac.IsaacCameraTwin"
        )
