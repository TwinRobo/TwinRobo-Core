"""Choosing and switching a camera's sensor and ISP."""

import textwrap
from pathlib import Path

import pytest
import torch

from twinrobo import CameraTwin
from twinrobo.isp import IdentityISP, ISPModel, TunedISP, list_tunings, load_tuning
from twinrobo.isp.color import srgb_to_linear
from twinrobo.registry import CatalogRegistry
from twinrobo.sensor import IdealSensor, SensorModel, TunedSensor

CAMERA = "raspberrypi/camera-module-3/standard"
UNTUNED = "logitech/c920"
TUNINGS = Path(load_tuning.__code__.co_filename).parent / "tunings"


def documented_example():
    """The example in `ISPModel`'s documentation, run as a reader would copy it."""
    text = ISPModel.__doc__.split("::")[1].split("What a `process`")[0]
    names = {"ISPModel": ISPModel}
    exec(textwrap.dedent(text), names)
    return names["Gamma"]


Gamma = documented_example()


class Temporal(ISPModel):
    """Counts each stream's frames."""

    def __init__(self):
        self.frames = {}

    def reset(self, stream=None):
        if stream is None:
            self.frames.clear()
        else:
            self.frames.pop(stream, None)

    def process(self, raw, metadata=None):
        name = metadata.get("stream", metadata.get("camera"))
        self.frames[name] = self.frames.get(name, 0) + 1
        return raw


class Bayer(ISPModel):
    input_format = "rggb"

    def process(self, raw, metadata=None):
        return raw.repeat(1, 3, 1, 1)


class Wrong(ISPModel):
    def process(self, raw, metadata=None):
        return raw[:, :1]


def twin(camera=CAMERA, **kw):
    return CameraTwin.from_spec(CatalogRegistry().load(camera), build_psf=False, **kw)


def run(t, stream="a", h=60, w=80):
    x = 0.2 + 0.5 * torch.rand(1, 3, h, w)
    return t.process(x, torch.ones(1, 1, h, w), metadata={"stream": stream})


def test_a_bundled_tuning_by_name():
    t = twin(isp="imx477")
    f = run(t)
    assert f.raw.shape[1] == 1
    assert isinstance(t.sensor, TunedSensor) and t.sensor.tuning.name == "imx477"
    assert f.metadata["pipeline"]["matches_spec_tuning"] is False


def test_a_tuning_works_on_a_camera_without_one():
    f = run(twin(UNTUNED, isp="imx219"))
    assert f.metadata["pipeline"]["isp_tuning"] == "imx219"
    assert f.metadata["pipeline"]["matches_spec_tuning"] is None


def test_a_tuning_by_path():
    t = twin(isp=TUNINGS / "imx219.json")
    assert t.isp.tuning.name == "imx219"


def test_the_cameras_own_tuning_matches_its_spec():
    assert run(twin(isp=True)).metadata["pipeline"]["matches_spec_tuning"] is True


def test_a_changed_file_with_the_cameras_name_does_not_match(tmp_path):
    text = (TUNINGS / "imx708.json").read_text()
    path = tmp_path / "imx708.json"
    path.write_text(text + "\n")
    assert run(twin(isp=path)).metadata["pipeline"]["matches_spec_tuning"] is False


def test_the_documented_custom_isp():
    t = twin(isp=Gamma())
    x = torch.full((1, 3, 8, 8), 0.5)
    f = t.process(x, torch.ones(1, 1, 8, 8))
    assert torch.allclose(f.rgb, srgb_to_linear(x**0.8))
    assert f.metadata["pipeline"]["isp"] == "Gamma"
    assert f.metadata["pipeline"]["matches_spec_tuning"] is None


def test_a_custom_isp_on_a_tuned_sensor():
    sensor = TunedSensor(load_tuning("imx708"))
    f = run(twin(isp=Bayer(), sensor=sensor))
    assert f.rgb.shape[1] == 3 and f.metadata["pipeline"]["raw_format"] == "rggb"
    assert f.metadata["pipeline"]["sensor_tuning"] == "imx708"
    assert f.metadata["pipeline"]["matches_spec_tuning"] is None


def test_formats_must_match():
    t = load_tuning("imx708")
    with pytest.raises(ValueError, match="TunedISP takes 'rggb'"):
        twin(isp=TunedISP(t))
    with pytest.raises(ValueError, match="IdentityISP takes 'rgb'"):
        CameraTwin(sensor=TunedSensor(t))
    with pytest.raises(ValueError, match="Gamma takes 'rgb'"):
        twin(isp=Gamma(), sensor=TunedSensor(t))


def test_a_directly_set_isp_is_checked_too():
    t = twin()
    t.isp = TunedISP(load_tuning("imx708"))
    with pytest.raises(ValueError, match="takes 'rggb'"):
        run(t)


def test_sensor_goes_only_with_an_isp_model():
    for isp in (None, True, False, "imx219"):
        with pytest.raises(ValueError, match="sensor= goes with"):
            twin(isp=isp, sensor=IdealSensor())


def test_infrared_and_mono_cameras_reject_tunings_but_take_custom_isps():
    spec = CatalogRegistry().load(CAMERA)
    spec.sensor.color = "mono"
    with pytest.raises(ValueError, match="color sensors"):
        CameraTwin.from_spec(spec, build_psf=False, isp="imx219")
    assert run(CameraTwin.from_spec(spec, build_psf=False, isp=Gamma())).rgb.shape[1] == 3


def test_an_isp_with_a_wrong_output_is_named():
    with pytest.raises(ValueError, match=r"Wrong returned .*expected float \(1, 3, 60, 80\)"):
        run(twin(isp=Wrong()))


def test_a_bad_selection_fails_before_the_lens_is_built(monkeypatch):
    def fail(*a):
        raise AssertionError("built the lens")

    monkeypatch.setattr("twinrobo.camera.build_reference_optics", fail)
    with pytest.raises(ValueError, match="no tuning 'nope'"):
        twin(isp="nope")


def test_no_isp_is_the_default():
    t = twin()
    assert type(t.sensor) is IdealSensor and type(t.isp) is IdentityISP


@pytest.mark.parametrize("name", list_tunings())
def test_every_tuning_can_be_switched_to(name):
    t = twin()
    run(t)
    t.set_isp(name)
    assert run(t).metadata["pipeline"]["sensor_tuning"] == name


def test_switching_through_every_kind_of_isp():
    t = twin()
    assert run(t).raw.shape[1] == 3
    t.set_isp("imx477")
    assert run(t).raw.shape[1] == 1
    t.set_isp(Gamma())
    f = run(t)
    assert f.raw.shape[1] == 3 and f.metadata["pipeline"]["isp"] == "Gamma"
    t.set_isp(True)
    assert run(t).metadata["pipeline"]["matches_spec_tuning"] is True
    t.set_isp(False)
    f = run(t)
    assert type(t.isp) is IdentityISP and torch.equal(f.rgb, f.rgb_optical)


def test_a_failed_switch_keeps_the_old_pair():
    t = twin(isp=True)
    sensor, isp = t.sensor, t.isp
    for bad in ("nope", TunedISP(load_tuning("imx708"))):
        with pytest.raises(ValueError):
            t.set_isp(bad)
    assert t.sensor is sensor and t.isp is isp


def test_a_switch_keeps_the_optics_and_spec():
    t = twin(isp=True)
    optics, spec = t.optics, t.spec
    t.set_isp("imx219")
    assert t.optics is optics and t.spec is spec


def test_a_switch_clears_the_old_state():
    isp = Temporal()
    t = twin(isp=isp)
    run(t)
    t.set_isp(False)
    t.set_isp(isp)
    assert isp.frames == {}


def test_keeping_the_sensor_changes_only_the_isp():
    t = twin(isp=True)
    sensor = t.sensor
    t.set_isp(Bayer(), sensor=t.sensor)
    assert t.sensor is sensor and isinstance(t.isp, Bayer)


def test_reset_reaches_the_isp_stream_by_stream():
    isp = Temporal()
    t = twin(isp=isp)
    run(t, "a")
    run(t, "b")
    t.reset("a")
    assert isp.frames == {"b": 1}
    t.reset()
    assert isp.frames == {}


def test_a_custom_sensor_defaults_to_rgb():
    class Mine(SensorModel):
        def capture(self, irradiance, exposure=1.0, metadata=None):
            return irradiance * exposure

    assert run(twin(isp=Gamma(), sensor=Mine())).metadata["pipeline"]["sensor"] == "Mine"


def test_a_custom_isp_can_call_its_settings_tuning():
    class Settings(Gamma):
        tuning = {"gamma": 0.8}

    f = run(twin(isp=Settings()))
    assert f.metadata["pipeline"]["isp_tuning"] is None


def test_a_double_precision_isp_gives_the_frames_type():
    class Double(Gamma):
        def process(self, raw, metadata=None):
            return raw.double()

    assert run(twin(isp=Double())).rgb.dtype == torch.float32


def test_a_missing_spec_tuning_does_not_stop_another_tuning():
    spec = CatalogRegistry().load(CAMERA)
    spec.isp["tuning"] = "/nonexistent/tuning.json"
    t = CameraTwin.from_spec(spec, build_psf=False, isp="imx477")
    assert run(t).metadata["pipeline"]["matches_spec_tuning"] is None


@pytest.mark.parametrize("text", ['{"version": 2}', "[1]", '{"algorithms": [1]}'])
def test_a_file_that_is_not_a_tuning_is_named(tmp_path, text):
    path = tmp_path / "other.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="not a full color tuning"):
        load_tuning(path)
    spec = CatalogRegistry().load(CAMERA)
    spec.isp["tuning"] = str(path)
    t = CameraTwin.from_spec(spec, build_psf=False, isp="imx477")
    assert run(t).metadata["pipeline"]["matches_spec_tuning"] is None


def test_a_renamed_file_changes_nothing(tmp_path):
    path = tmp_path / "imx477.json"
    path.write_text((TUNINGS / "imx477.json").read_text())
    file, bundled = load_tuning(path), load_tuning("imx477")
    assert file.bits == 10 and file.native_size is None
    assert (bundled.bits, bundled.native_size) != (file.bits, file.native_size)
    given = load_tuning(path, bundled.native_size, bundled.bits)
    assert (given.bits, given.native_size) == (bundled.bits, bundled.native_size)


def test_spec_gives_a_files_size_and_bits(tmp_path):
    path = tmp_path / "mine.json"
    path.write_text((TUNINGS / "imx477.json").read_text())
    spec = CatalogRegistry().load(CAMERA)
    spec.isp.update(native_size=[4056, 3040], bits=12)
    t = CameraTwin.from_spec(spec, build_psf=False, isp=path)
    assert t.sensor.tuning.bits == 12 and t.sensor.tuning.native_size == (4056, 3040)


def test_same_bytes_with_other_sensor_settings_do_not_match(tmp_path):
    path = tmp_path / "mine.json"
    path.write_text((TUNINGS / "imx708.json").read_text())
    assert run(twin(isp=path)).metadata["pipeline"]["matches_spec_tuning"] is False


def test_libero_init_states_start_the_cameras_again():
    from twinrobo.datasets.libero.env import CameraTwinLiberoEnv

    class Env:
        def set_init_state(self, state):
            return {}

    class Camera:
        attached = 0

        def attach(self, name):
            self.attached += 1

    env = CameraTwinLiberoEnv.__new__(CameraTwinLiberoEnv)
    env.env, env.cameras, env._attached = Env(), {"agentview": Camera()}, True
    env.set_init_state(None)
    assert env.cameras["agentview"].attached == 1
