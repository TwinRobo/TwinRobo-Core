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
