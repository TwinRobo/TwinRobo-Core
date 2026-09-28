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


def test_mono_sensor_outputs_luminance():
    from twinrobo import CatalogRegistry

    cam = CameraTwin()
    rgb, depth = torch.rand(1, 3, 8, 8), torch.full((1, 1, 8, 8), 2.0)
    assert torch.equal(cam.process(rgb, depth).rgb_optical, rgb)  # no spec: color
    cam.spec = CatalogRegistry().load("luxonis/oak-d/mono")
    frame = cam.process(rgb, depth)
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    for f in (frame.rgb_optical, frame.rgb_ideal):
        assert f.shape == rgb.shape and torch.allclose(f[:, 0], luma, atol=1e-6)
        assert torch.equal(f[:, 0], f[:, 1]) and torch.equal(f[:, 1], f[:, 2])
