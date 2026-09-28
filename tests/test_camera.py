import torch

from twinrobo import CameraTwin


def test_identity_pipeline():
    cam = CameraTwin()
    rgb = torch.rand(1, 3, 16, 16) * 1.5
    depth = torch.full((1, 1, 16, 16), 2.0)
    frame = cam.process(rgb, depth, timestamp=0.1)
    assert torch.equal(frame.rgb_ideal, rgb)
    assert torch.equal(frame.rgb_optical, rgb)
    assert frame.rgb.max() <= 1.0  # ideal sensor clips
    assert frame.timestamp == 0.1


def test_depth_follows_the_spec_unless_forced():
    import pytest

    from twinrobo import CatalogRegistry, DepthUnavailableError

    cam = CameraTwin()  # no spec: nothing says the camera lacks depth
    rgb, depth = torch.rand(1, 3, 16, 16), torch.full((1, 1, 16, 16), 2.0)
    assert cam.outputs_depth and torch.equal(cam.process(rgb, depth).depth, depth)
    cam.spec = CatalogRegistry().load("logitech/c920")  # an RGB camera
    frame = cam.process(rgb, depth)
    assert not frame.has_depth
    with pytest.raises(DepthUnavailableError, match="logitech/c920"):
        _ = frame.depth
    assert torch.equal(cam.process(rgb, depth, force_depth=True).depth, depth)
    cam.spec.outputs.depth = True  # a camera whose depth output is simulated
    assert torch.equal(cam.process(rgb, depth).depth, depth)
