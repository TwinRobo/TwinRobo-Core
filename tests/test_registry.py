import pytest

from twinrobo import CatalogError, CatalogRegistry
from twinrobo.registry import CATALOG_ENV, validate_camera_id


def _make_catalog(root, camera_id, example_spec_path):
    d = root.joinpath(*camera_id.split("/"))
    d.mkdir(parents=True)
    text = example_spec_path.read_text().replace(
        "path: lens.json", f"path: {example_spec_path.parent / 'lens.json'}"
    )
    (d / "camera.yaml").write_text(text)
    return d / "camera.yaml"


def test_resolve_and_list(tmp_path, example_spec_path):
    cid = "example/cellphone/80deg/f2.0"
    path = _make_catalog(tmp_path, cid, example_spec_path)
    reg = CatalogRegistry([tmp_path], use_env=False, use_builtin=False)
    assert reg.resolve(cid) == path
    assert reg.load(cid).id == "example.cellphone80deg.v0"
    assert reg.list() == [cid]


def test_distinct_configs_are_distinct(tmp_path, example_spec_path):
    a = "lucid/triton/imx568/edmund-8mm/f2.8"
    b = "lucid/triton/imx568/edmund-6mm/f2.8"
    _make_catalog(tmp_path, a, example_spec_path)
    reg = CatalogRegistry([tmp_path], use_env=False, use_builtin=False)
    reg.resolve(a)
    with pytest.raises(CatalogError, match="not found"):
        reg.resolve(b)


def test_env_roots(tmp_path, example_spec_path, monkeypatch):
    cid = "example/cam"
    _make_catalog(tmp_path, cid, example_spec_path)
    monkeypatch.setenv(CATALOG_ENV, str(tmp_path))
    assert CatalogRegistry().resolve(cid).is_file()


@pytest.mark.parametrize("bad", ["", "Upper/case", "a//b", "../escape", "a/./b", "a b"])
def test_invalid_ids(bad):
    with pytest.raises(CatalogError):
        validate_camera_id(bad)


def test_builtin_catalog_zed_x():
    reg = CatalogRegistry(use_env=False)
    ids = reg.list()
    assert "stereolabs/zed-x/2.2mm" in ids and "stereolabs/zed-x/4mm" in ids
    spec = reg.load("stereolabs/zed-x/2.2mm")
    assert spec.lens.deeplens_model.path.is_file()  # surrogate lens next to camera.yaml
    assert spec.validation.status == "estimated"
