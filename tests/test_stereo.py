import numpy as np
import pytest
import torch

from twinrobo.exceptions import SpecError
from twinrobo.geometry import CameraIntrinsics
from twinrobo.optics.distortion import DistortionWarp, RadialDistortion
from twinrobo.plugins.mujoco.mounts import CameraMount, expand_mount
from twinrobo.stereo import ModuleRegistry, StereoModule, eye_poses


def test_catalog_modules():
    mods = {m.id: m for m in ModuleRegistry().list()}
    assert mods["stereolabs/zed-x/2.2mm"].baseline_m == pytest.approx(0.120)
    assert mods["stereolabs/zed-x/4mm"].baseline_m == pytest.approx(0.120)
    mini = mods["stereolabs/zed-x-mini/2.2mm"]
    assert mini.baseline_m == pytest.approx(0.050) and mini.output == "rectified"
    assert mini.left.spec.is_file() and mini.left.spec == mini.right.spec
    assert mini.left.spec == mods["stereolabs/zed-x/2.2mm"].left.spec  # the same 2.2 mm eye
    assert mini.to_json()["baseline_mm"] == 50.0


def test_module_file_validation(tmp_path):
    p = tmp_path / "module.yaml"
    p.write_text("type: mono\n")
    with pytest.raises(SpecError, match="not a stereo module"):
        StereoModule.from_yaml(p)
    p.write_text("type: stereo\neyes: {left: {}}\n")
    with pytest.raises(SpecError, match="eyes.right"):
        StereoModule.from_yaml(p)


def test_right_eye_is_a_baseline_to_the_image_right():
    m = ModuleRegistry().load("stereolabs/zed-x/2.2mm")
    R = CameraMount("c", rpy_deg=(30, 15, 5)).rotation()
    eyes = eye_poses(m, [1.0, 2.0, 3.0], R)
    (pl, Rl), (pr, Rr) = eyes["left"], eyes["right"]
    assert np.allclose(pl, [1, 2, 3]) and np.allclose(Rl, R) and np.allclose(Rr, R)
    assert np.allclose(pr - pl, 0.12 * R[:, 0])  # camera +x = image right


def test_stereo_mount_expands_to_two_eyes():
    mount = CameraMount(
        "st",
        body="world",
        pos=(0.5, -0.8, 1.4),
        rpy_deg=(120, 25, 0),
        module="stereolabs/zed-x-mini/2.2mm",
    )
    eyes = expand_mount(mount)
    assert set(eyes) == {"st_L", "st_R"}
    L, R = eyes["st_L"], eyes["st_R"]
    assert np.allclose(L.pos, mount.pos) and np.allclose(L.rotation(), mount.rotation(), atol=1e-4)
    assert np.allclose(np.subtract(R.pos, L.pos), 0.05 * mount.rotation()[:, 0], atol=1e-5)
    assert np.allclose(R.rotation(), L.rotation(), atol=1e-4)
    assert expand_mount(CameraMount("mono")) == {"mono": CameraMount("mono")}
    assert CameraMount.from_json(mount.to_json()) == mount  # the module survives a rig round trip


def test_rectify_undoes_the_distortion():
    intr = CameraIntrinsics(160, 100, 80.0, 80.0, 79.5, 49.5)
    warp = DistortionWarp(intr, RadialDistortion(k1=-0.06, k2=0.007))
    src = warp.source
    v, u = torch.meshgrid(
        torch.arange(src.height, dtype=torch.float64),
        torch.arange(src.width, dtype=torch.float64),
        indexing="ij",
    )
    x, y = (u - src.cx) / src.fx, (v - src.cy) / src.fy
    img = (
        (0.5 + 0.25 * torch.sin(7 * x) * torch.cos(5 * y)).float()[None, None].expand(1, 3, -1, -1)
    )
    sensor, _ = warp(img, torch.ones(1, 1, src.height, src.width))
    rect, pin = warp.rectify(sensor), warp.pinhole(img)
    core = (slice(None), slice(None), slice(15, -15), slice(20, -20))  # corners repeat the border
    err_rect = (rect[core] - pin[core]).abs().max()
    err_raw = (sensor[core] - pin[core]).abs().max()
    assert err_rect < 0.01  # two bilinear resamplings
    assert err_raw > 3 * err_rect  # and it was clearly distorted before


def test_module_baseline_can_be_changed():
    from twinrobo.stereo import ModuleRegistry

    zed = ModuleRegistry().load("stereolabs/zed-x-mini/2.2mm")
    wide = zed.with_baseline(0.09)
    assert abs(wide.baseline_m - 0.09) < 1e-12 and abs(zed.baseline_m - 0.05) < 1e-12
    assert wide.left == zed.left and wide.right.spec == zed.right.spec
    assert wide.housing_mm[0] == pytest.approx(zed.housing_mm[0] + 40)
    with pytest.raises(ValueError):
        zed.with_baseline(0)


def test_stereo_mount_baseline_override_round_trips():
    mount = CameraMount(
        "st",
        pos=(0.5, -0.8, 1.4),
        rpy_deg=(120, 25, 0),
        module="stereolabs/zed-x-mini/2.2mm",
        baseline_mm=75,
    )
    L, R = expand_mount(mount).values()
    assert np.allclose(np.subtract(R.pos, L.pos), 0.075 * mount.rotation()[:, 0], atol=1e-5)
    assert CameraMount.from_json(mount.to_json()) == mount
    assert "baseline_mm" not in CameraMount("st", module="stereolabs/zed-x-mini/2.2mm").to_json()
    with pytest.raises(ValueError, match="stereo module"):
        CameraMount("mono", baseline_mm=60)


def test_set_pair_baseline_moves_compiled_eyes_symmetrically():
    mujoco = pytest.importorskip("mujoco")
    from twinrobo.plugins.mujoco.mounts import pair_baseline, set_pair_baseline

    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="b" pos="1 2 3">
             <camera name="L" pos="-0.025 0.1 0.2"/><camera name="R" pos="0.025 0.1 0.2"/>
           </body></worldbody></mujoco>"""
    )
    assert pair_baseline(model, "L", "R") == pytest.approx(0.05)
    assert set_pair_baseline(model, "L", "R", 0.12) == pytest.approx(0.05)
    assert pair_baseline(model, "L", "R") == pytest.approx(0.12)
    assert np.allclose(model.cam_pos[0], (-0.06, 0.1, 0.2)) and np.allclose(
        model.cam_pos[1], (0.06, 0.1, 0.2)
    )
