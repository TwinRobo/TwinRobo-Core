import copy

import pytest
import yaml

from twinrobo import CameraSpec, SpecError


def test_example_spec_loads(example_spec_path):
    spec = CameraSpec.from_yaml(example_spec_path)
    assert spec.id == "example.cellphone80deg.v0"
    assert spec.schema_version == "0.1"
    assert (spec.sensor.resolution.width, spec.sensor.resolution.height) == (1920, 1200)
    assert spec.sensor.shutter == "global"
    assert spec.validation.status == "estimated"
    # deeplens_model.path resolves relative to the spec file
    assert spec.lens.deeplens_model.path.is_file()
    assert spec.lens.deeplens_model.path == example_spec_path.parent / "lens.json"


@pytest.fixture
def raw(example_spec_path):
    return yaml.safe_load(example_spec_path.read_text())


def test_rejects_unknown_schema_version(raw):
    raw["schema_version"] = "9.9"
    with pytest.raises(SpecError, match="schema_version"):
        CameraSpec.from_dict(raw)


def test_rejects_missing_resolution(raw):
    bad = copy.deepcopy(raw)
    del bad["sensor"]["resolution"]["width"]
    with pytest.raises(SpecError, match="sensor.resolution.width"):
        CameraSpec.from_dict(bad)


def test_rejects_bad_provenance(raw):
    raw["validation"]["status"] = "looks-good"
    with pytest.raises(SpecError, match="validation.status"):
        CameraSpec.from_dict(raw)


def test_missing_file(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        CameraSpec.from_yaml(tmp_path / "nope.yaml")


def test_overrides_reconfigure_the_same_camera(example_spec_path):
    from twinrobo.spec import apply_overrides

    base = CameraSpec.from_yaml(example_spec_path)
    spec = CameraSpec.from_yaml(
        example_spec_path, width=960, height=600, focus_distance_m=0.8, near_m=0.2
    )
    assert (spec.sensor.resolution.width, spec.sensor.resolution.height) == (960, 600)
    assert spec.lens.focus_distance_m == 0.8
    assert spec.calibration["psf"]["near_m"] == 0.2
    assert base.sensor.resolution.width == 1920  # the file itself is untouched
    raw = {"sensor": {"resolution": {"width": 100, "height": 50}}, "lens": {}}
    assert apply_overrides(raw, width=10)["sensor"]["resolution"] == {"width": 10, "height": 50}
    assert raw["sensor"]["resolution"]["width"] == 100  # a copy


def test_catalog_lens_reference(tmp_path, example_spec_path):
    """A spec outside the catalog can reuse a catalog lens as ``catalog:<id>/lens.json``."""
    raw = yaml.safe_load(example_spec_path.read_text())
    raw["lens"]["deeplens_model"]["path"] = "catalog:examples/cellphone80deg/lens.json"
    spec = CameraSpec.from_dict(raw, base_dir=tmp_path)
    assert spec.lens.deeplens_model.path == example_spec_path.parent / "lens.json"
