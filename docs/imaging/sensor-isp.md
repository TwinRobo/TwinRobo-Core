# Sensor and ISP

What happens after the lens: the sensor records the light and the ISP turns it into an image. The
[guide](https://twinrobo.github.io/sensor-isp/) explains switching ISPs between frames.

## Supported sensors and ISPs

| Model | What it does |
|---|---|
| `IdealSensor` | no noise: raw is the light times the exposure, clipped to 0 to 1 (the default) |
| `TunedSensor` | noise, exposure control and Bayer raw from a tuning |
| `IdentityISP` | passes the raw image through (the default) |
| `TunedISP` | white balance, color and tone from a tuning |
| Your own `SensorModel` or `ISPModel` | [anything you write](#create-your-own) |

## Supported tunings

Pass a name to use its sensor and ISP on a visible-light color camera:
`CameraTwin.from_catalog("logitech/c920", isp="imx477")`. `isp=True` uses the camera's own tuning,
if it has one. `list_tunings()` gives the names in code.

{{ catalog_tunings }}

## Create your own

### Your own ISP

Subclass `ISPModel` and override `process`. Its full contract is in `ISPModel` below.

```python
from twinrobo import CameraTwin
from twinrobo.isp import ISPModel
from twinrobo.isp.color import srgb_to_linear


class Gamma(ISPModel):
    def process(self, raw, metadata=None):
        return srgb_to_linear(raw.clamp(0, 1) ** 0.8)


twin = CameraTwin.from_catalog("logitech/c920", isp=Gamma())
```

### Your own sensor

Subclass `SensorModel` and override `capture`. Pair it with an ISP: `sensor=` goes with an
`ISPModel`.

```python
import torch

from twinrobo.isp import IdentityISP
from twinrobo.sensor import SensorModel


class Noisy(SensorModel):
    def capture(self, irradiance, exposure=1.0, metadata=None):
        raw = irradiance * exposure
        return (raw + 0.01 * torch.randn_like(raw)).clamp(0, 1)


twin = CameraTwin.from_catalog("logitech/c920", isp=IdentityISP(), sensor=Noisy())
```

### Your own tuning

A tuning is a Raspberry Pi libcamera tuning file with the `rpi.*` algorithms. Pass its path:

```python
twin = CameraTwin.from_catalog("logitech/c920", isp="my_sensor.json")
```

Give the sensor's size and bit depth under `isp` in the camera's spec, as `native_size` and `bits`.
Without them, a file gets 10 bits and no noise scaling for smaller frames.

## The interfaces

<span id="twinrobo.isp.base"></span>

::: twinrobo.isp.base
    options:
      show_root_heading: false
      show_root_toc_entry: false

<span id="twinrobo.sensor.base"></span>

::: twinrobo.sensor.base
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Tuned models

<span id="twinrobo.isp.tuned"></span>

::: twinrobo.isp.tuned
    options:
      show_root_heading: false
      show_root_toc_entry: false

<span id="twinrobo.sensor.tuned"></span>

::: twinrobo.sensor.tuned
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Tunings

<span id="twinrobo.isp.tuning"></span>

::: twinrobo.isp.tuning
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Color conversion

<span id="twinrobo.isp.color"></span>

::: twinrobo.isp.color
    options:
      show_root_heading: false
      show_root_toc_entry: false
