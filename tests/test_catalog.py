"""Every built-in catalog entry must load: a gate for catalog contributions."""

import pytest

from twinrobo import CameraSpec, CatalogRegistry
from twinrobo.registry import BUILTIN_CATALOG, validate_camera_id
from twinrobo.spec import PROVENANCE_LEVELS
from twinrobo.stereo import ModuleRegistry

REG = CatalogRegistry([], use_env=False)
CAMERAS = REG.list()
MODULES = ModuleRegistry([BUILTIN_CATALOG]).list()


def test_catalog_is_not_empty():
    assert "stereolabs/zed-x/2.2mm" in CAMERAS
    assert MODULES


@pytest.mark.parametrize("camera_id", CAMERAS)
def test_camera_entry(camera_id):
    validate_camera_id(camera_id)  # lowercase, slash-separated segments
    spec = REG.load(camera_id)
    res = spec.sensor.resolution
    assert res.width > 0 and res.height > 0
    lens = spec.lens.deeplens_model
    assert lens is not None and lens.path.is_file(), f"{camera_id}: lens file missing"
    assert lens.path.parent == REG.resolve(camera_id).parent, "keep the lens file with its spec"
    assert spec.validation.status in PROVENANCE_LEVELS, spec.validation.status
    # the override path contributors and users rely on keeps the field of view
    half = CameraSpec.from_yaml(REG.resolve(camera_id), width=res.width // 2)
    assert half.sensor.resolution.width == res.width // 2


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.id)
def test_stereo_module(module):
    for eye in ("left", "right"):
        spec = module.eye(eye).spec
        assert spec is not None and spec.is_file(), f"{module.id}: {eye} eye spec missing"
        CameraSpec.from_yaml(spec)
    assert 0.0 < module.baseline_m < 1.0
    assert module.output in ("rectified", "raw")
