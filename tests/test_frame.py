import pytest
import torch

from twinrobo import CameraFrame
from twinrobo.frame import check_depth, check_rgb, valid_depth_mask


def test_check_shapes():
    rgb = torch.rand(2, 3, 8, 10)
    check_rgb(rgb)
    check_depth(torch.rand(2, 1, 8, 10), rgb)
    with pytest.raises(ValueError):
        check_rgb(torch.rand(2, 8, 10, 3))
    with pytest.raises(ValueError):
        check_depth(torch.rand(2, 1, 8, 9), rgb)


def test_valid_depth_mask():
    d = torch.tensor([1.0, 0.0, -1.0, float("inf"), float("nan")]).view(1, 1, 1, 5)
    assert valid_depth_mask(d).flatten().tolist() == [True, False, False, False, False]


def test_frame_to():
    f = CameraFrame(rgb=torch.rand(1, 3, 4, 4), depth=torch.rand(1, 1, 4, 4), metadata={"a": 1})
    g = f.to("cpu")
    assert g.rgb.device.type == "cpu" and g.metadata == {"a": 1} and g.raw is None


def test_depth_hidden_unless_the_camera_outputs_it():
    from twinrobo import DepthUnavailableError

    d = torch.rand(1, 1, 4, 4)
    f = CameraFrame(
        rgb=torch.rand(1, 3, 4, 4), depth=d, has_depth=False, metadata={"camera_id": "x/y"}
    )
    with pytest.raises(DepthUnavailableError, match="force_depth=True"):
        _ = f.depth
    assert "has_depth=False" in repr(f)  # repr does not read the hidden depth
    moved = f.to("cpu")
    assert not moved.has_depth
    moved.has_depth = True
    assert torch.equal(moved.depth, d)
