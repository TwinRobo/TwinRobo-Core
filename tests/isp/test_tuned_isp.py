"""The tuned sensor and ISP: libcamera's tunings on TwinRobo frames."""

import numpy as np
import pytest
import torch

from twinrobo import CameraFrame, CameraTwin
from twinrobo.isp import TunedISP, list_tunings, load_tuning
from twinrobo.isp.color import linear_to_srgb
from twinrobo.isp.demosaic import demosaic_rggb
from twinrobo.isp.tone import bilateral, gaussian_blur, piecewise_linear, rgb_to_ycc, ycc_to_rgb
from twinrobo.registry import CatalogRegistry
from twinrobo.sensor import TunedSensor, new_stream
from twinrobo.sensor.cfa import mosaic_rggb
from twinrobo.sensor.response import adc

CAMERA = "raspberrypi/camera-module-3/standard"


def scene(h=120, w=160, seed=0):
    """A colored scene with gentle gradients."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, h // 20, w // 20, generator=g)
    x = torch.nn.functional.interpolate(base, (h, w), mode="bilinear", align_corners=False)
    return 0.05 + 0.6 * x


def camera(tuning="imx708", noise=True, seed=0):
    t = load_tuning(tuning)
    return TunedSensor(t, seed=seed, noise=noise), TunedISP(t)


@pytest.mark.parametrize("name", list_tunings())
def test_every_bundled_tuning_produces_a_camera_image(name):
    sensor, isp = camera(name)
    meta = {"camera": "a"}
    raw = sensor.capture(scene(), 1.0, meta)
    out = isp.process(raw, meta)
    assert raw.shape == (1, 1, 120, 160) and out.shape == (1, 3, 120, 160)
    assert torch.isfinite(out).all() and 0 <= out.min() and out.max() <= 1
    assert 2000 < meta["isp"]["awb_ct_k"][0] < 12000


def test_tunings_differ():
    a, b = (load_tuning(n) for n in ("imx708", "ov5647"))
    assert a.noise_slope != b.noise_slope and a.black_level != b.black_level
    assert not np.array_equal(a.ccms[0][1], b.ccms[0][1])


def test_an_unknown_tuning_lists_the_bundled_ones():
    with pytest.raises(ValueError, match="imx708"):
        load_tuning("nope")


def test_demosaic_keeps_a_smooth_image():
    rgb = scene()
    back = demosaic_rggb(mosaic_rggb(rgb))
    assert (back - rgb)[..., 4:-4, 4:-4].abs().max() < 0.04


def test_mosaic_samples_rggb():
    rgb = torch.stack([torch.full((2, 2), v) for v in (0.1, 0.2, 0.3)])[None]
    want = torch.tensor([[0.1, 0.2], [0.2, 0.3]])
    assert torch.allclose(mosaic_rggb(rgb)[0, 0], want)


def test_adc_quantizes_and_saturates():
    out = adc(torch.tensor([0.0, 100.0, 1e6]), 4096, 10)
    assert out[0] == 0
    assert (out[1] + 4096) % 64 == 0
    assert out[2] == 1023 * 64 - 4096


def test_tone_curve_matches_numpy_interp():
    xp, fp = torch.tensor([0.0, 1.0, 3.0]), torch.tensor([0.0, 2.0, 2.5])
    x = torch.tensor([-1.0, 0.5, 2.0, 4.0])
    assert piecewise_linear(x, xp, fp).tolist() == pytest.approx(
        np.interp(x.numpy(), xp.numpy(), fp.numpy()).tolist()
    )


def test_ycc_round_trip():
    x = scene()
    assert (ycc_to_rgb(rgb_to_ycc(x)) - x).abs().max() < 2e-3


def test_blur_and_bilateral_keep_a_constant_image():
    x = torch.full((1, 2, 16, 16), 0.3)
    assert (gaussian_blur(x, 1.0) - x).abs().max() < 1e-5
    assert (bilateral(x, 3.2, 1.5) - x).abs().max() < 1e-5


def test_bilateral_keeps_an_edge():
    x = torch.zeros(1, 1, 16, 16)
    x[..., 8:] = 1.0
    assert (bilateral(x, 0.1, 1.5) - x).abs().max() < 1e-3


def test_noise_rises_with_gain_and_is_reproducible():
    flat = torch.full((1, 3, 200, 200), 0.2)
    std = []
    for gain in (1.0, 8.0):
        t = load_tuning("imx708")
        std.append(
            TunedSensor(t, analogue_gain=gain)
            .capture(flat, 1.0, {"camera": "a"})[..., ::2, ::2]
            .std()
        )
    assert std[1] > 2 * std[0]
    a, b = (camera()[0].capture(flat, 1.0, {"camera": "a"}) for _ in range(2))
    assert torch.equal(a, b)


def test_consecutive_frames_have_different_noise():
    sensor, _ = camera()
    flat = torch.full((1, 3, 64, 64), 0.2)
    a, b = (sensor.capture(flat, 1.0, {"camera": "a"}) for _ in range(2))
    assert not torch.equal(a, b)


def test_noise_leaves_the_global_random_state_alone():
    sensor, _ = camera()
    torch.manual_seed(1)
    expected = torch.rand(1)
    torch.manual_seed(1)
    sensor.capture(scene(), 1.0, {"camera": "a"})
    assert torch.equal(torch.rand(1), expected)


def test_a_smaller_readout_has_less_noise():
    t = load_tuning("imx708")
    sensor = TunedSensor(t)
    assert sensor.noise_scale(2592, 4608) == 1.0
    assert sensor.noise_scale(1296, 2304) == pytest.approx(0.5)


def test_cameras_of_one_sensor_keep_their_own_exposure():
    sensor, _ = camera(noise=False)
    dark, bright = 0.1 * scene(), 0.9 * scene()
    meta_a, meta_b = {"camera": "a"}, {"camera": "b"}
    sensor.capture(dark, 1.0, meta_a)
    sensor.capture(bright, 1.0, meta_b)
    again = {"camera": "a"}
    sensor.capture(dark, 1.0, again)
    assert again["sensor"]["exposure"] == pytest.approx(meta_a["sensor"]["exposure"], rel=1e-4)
    assert meta_b["sensor"]["exposure"][0] < meta_a["sensor"]["exposure"][0] / 2


def test_streams_with_one_camera_name_stay_apart():
    sensor, _ = camera()
    dark, bright = 0.1 * scene(), 0.9 * scene()
    a, b = new_stream("agentview"), new_stream("agentview")
    first = {"camera": "agentview", "stream": a}
    raw = sensor.capture(dark, 1.0, first)
    other = sensor.capture(dark, 1.0, {"camera": "agentview", "stream": b})
    sensor.capture(bright, 1.0, {"camera": "agentview", "stream": b})
    again = {"camera": "agentview", "stream": a}
    sensor.capture(dark, 1.0, again)
    assert again["sensor"]["exposure"] == pytest.approx(first["sensor"]["exposure"], rel=1e-4)
    assert not torch.equal(raw, other)


def test_shaded_corners_are_noisier():
    sensor, _ = camera()
    flat = torch.full((1, 3, 256, 256), 0.2)
    shade = torch.ones(3, 256, 256)
    shade[..., 128:] = 0.25
    raw = sensor.capture(flat, 1.0, {"camera": "a", "illumination": shade})[..., ::2, ::2]
    assert raw[..., 64:].std() > 1.5 * raw[..., :64].std()


def test_exposure_settles_gradually_after_a_scene_change():
    sensor, _ = camera(noise=False)
    seen = []
    for x in (scene(), 4 * scene(), 4 * scene()):
        m = {"camera": "a"}
        sensor.capture(x, 1.0, m)
        seen.append(m["sensor"]["exposure"][0])
    assert seen[0] > seen[1] > seen[2]
    settled = seen[0] / 4
    assert seen[2] > settled


def test_reset_starts_a_camera_again():
    sensor, _ = camera()
    x = scene()
    first = sensor.capture(x, 1.0, {"camera": "a"})
    sensor.capture(0.5 * x, 1.0, {"camera": "a"})
    sensor.reset("a")
    assert torch.equal(sensor.capture(x, 1.0, {"camera": "a"}), first)


def test_batches_are_exposed_image_by_image():
    sensor, isp = camera(noise=False)
    a, b = scene(), 0.25 * scene()
    meta = {"camera": "a"}
    raw = sensor.capture(torch.cat([a, b]), 1.0, meta)
    out = isp.process(raw, meta)
    assert len(meta["sensor"]["exposure"]) == 2 and len(meta["isp"]["awb_ct_k"]) == 2
    assert meta["sensor"]["exposure"][1] == pytest.approx(
        4 * meta["sensor"]["exposure"][0], rel=1e-3
    )
    assert out.shape[0] == 2


def test_white_balance_removes_a_color_cast():
    sensor, isp = camera(noise=False)
    grey = torch.full((1, 3, 96, 96), 0.3)
    cast = grey * torch.tensor([1.0, 0.92, 0.75])[None, :, None, None]
    meta = {"camera": "a"}
    out = isp.process(sensor.capture(cast, 1.0, meta), meta).mean((2, 3))[0]
    before = cast.mean((2, 3))[0]
    assert out.max() / out.min() < before.max() / before.min()


def test_odd_frames_keep_their_size():
    sensor, isp = camera()
    assert isp.process(sensor.capture(scene(121, 161))).shape[-2:] == (121, 161)


def test_small_frames_work():
    sensor, isp = camera()
    assert isp.process(sensor.capture(torch.full((1, 3, 8, 16), 0.2))).shape[-2:] == (8, 16)


def test_white_balance_stays_in_range_on_a_hard_color():
    """A uniform color whose search curve is flat: the refinement stays between its points."""
    sensor, isp = camera(noise=False)
    x = torch.tensor([0.16853147, 0.2, 0.07348194])[None, :, None, None].expand(1, 3, 96, 96)
    meta = {"camera": "a"}
    isp.process(sensor.capture(x, 1.0, meta), meta)
    assert max(meta["isp"]["awb_gains"][0]) < 5


def test_a_stream_can_change_device():
    sensor, _ = camera()
    meta = {"camera": "a"}
    sensor.capture(scene(), 1.0, meta)
    if torch.cuda.is_available():
        sensor.capture(scene().cuda(), 1.0, meta)


# -- on a CameraTwin -------------------------------------------------------------------------------


def twin(**kw):
    spec = CatalogRegistry().load(CAMERA)
    return CameraTwin.from_spec(spec, build_psf=False, **kw)


def test_catalog_twins_are_unchanged_by_default():
    t = twin()
    x = scene()
    f = t.process(x, torch.ones(1, 1, *x.shape[-2:]))
    assert torch.equal(f.rgb, x) and f.raw.shape[1] == 3


def test_isp_true_gives_the_camera_output():
    t = twin(isp=True)
    x = scene()
    f = t.process(x, torch.ones(1, 1, *x.shape[-2:]), metadata={"camera": "a"})
    assert isinstance(f, CameraFrame)
    assert f.raw.shape == (1, 1, 120, 160)
    assert torch.equal(f.rgb_optical, x)
    assert f.metadata["sensor"]["tuning"] == "imx708" and "awb_ct_k" in f.metadata["isp"]
    assert not torch.equal(f.rgb, x)


def test_caller_metadata_is_not_mutated():
    t = twin(isp=True)
    x = scene()
    meta = {"camera": "a"}
    t.process(x, torch.ones(1, 1, *x.shape[-2:]), metadata=meta)
    assert meta == {"camera": "a"}


def test_frame_rgb_encodes_back_to_the_display_image():
    from twinrobo.isp.color import srgb_to_linear

    display = torch.linspace(0, 1, 101)
    assert (linear_to_srgb(srgb_to_linear(display)) - display).abs().max() < 2e-6


def test_isp_needs_a_tuning():
    spec = CatalogRegistry().load("logitech/c920")
    with pytest.raises(ValueError, match="isp.tuning"):
        CameraTwin.from_spec(spec, build_psf=False, isp=True)


def test_mono_and_infrared_sensors_have_no_tuned_isp():
    spec = CatalogRegistry().load(CAMERA)
    spec.sensor.color = "mono"
    with pytest.raises(ValueError, match="color sensors"):
        CameraTwin.from_spec(spec, build_psf=False, isp=True)


def test_mujoco_adapters_keep_separate_streams():
    pytest.importorskip("mujoco")
    from twinrobo.plugins.mujoco import MujocoCameraTwin

    t = twin(isp=True)

    class Backend:
        model = None

    a = MujocoCameraTwin(t, Backend())
    b = MujocoCameraTwin(t, Backend())
    assert a._stream != b._stream


def test_highlights_expose_by_the_mean_of_the_brightest_pixels():
    """As libcamera's inter-quantile mean: one percentile would overexpose small lights."""
    t = load_tuning("imx708")
    x = torch.full((1, 3, 100, 100), 0.1)
    x[..., :10, :10] = 1.0  # 1 % of the frame
    sensor = TunedSensor(t, noise=False)
    sensor.capture(x, 1.0, meta := {"camera": "a"})
    top = (0.01 * 1.0 + 0.01 * 0.1) / 0.02
    assert meta["sensor"]["exposure"][0] == pytest.approx(
        max(t.y_target / 0.11, t.highlight_y / top), rel=0.2
    )


def test_a_black_frame_does_not_spoil_the_next_ones():
    sensor = TunedSensor(load_tuning("imx708"), noise=False)
    black, grey = torch.zeros(1, 3, 40, 40), torch.full((1, 3, 40, 40), 0.3)
    sensor.capture(black, 1.0, meta := {"camera": "a"})
    assert meta["sensor"]["exposure"][0] == 1.0
    sensor.reset()
    sensor.capture(grey, 1.0, settled := {"camera": "a"})
    sensor.capture(black, 1.0, held := {"camera": "a"})
    assert held["sensor"]["exposure"] == pytest.approx(settled["sensor"]["exposure"])
    assert sensor.capture(grey, 1.0, {"camera": "a"}).max() < 1.0
