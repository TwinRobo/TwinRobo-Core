# Lens files

Read a lens from optical design software and render through it, or export any TwinRobo lens back
to it. `twinrobo.lensio` reads each format into one neutral `Prescription`, so every format
converts to every other.

## Supported formats

| Format | Read | Write | Notes |
|---|---|---|---|
| Zemax OpticStudio `.zmx` | yes | yes | UTF-16 (OpticStudio's default), UTF-8 or ANSI; written in UTF-16 |
| Zemax archive `.zar` | yes | no | needs `pip install zmxtools` (AGPL-3.0); uses the glass catalogs packed in it |
| CODE V `.seq` | yes | yes | the commands that enter a lens; see below |
| Prescription table `.csv` | yes | yes | for lenses typed in from patents and datasheets |
| DeepLens `.json` | yes | yes | what TwinRobo renders from |

Not supported: OpticStudio's binary `.zos`, OSLO, LightTools and Ansys Speos files. Export a
`.zmx` from those programs instead.

## Supported lenses

Centered, rotationally symmetric, sequential lenses:

| Surface | Zemax | CODE V |
|---|---|---|
| Sphere, plane, conic | `STANDARD` | `S` rows, `CON` + `K` |
| Even asphere | `EVENASPH` (r² to r¹⁶) | `ASP`: `K`, `A` (r⁴) to `H` (r¹⁸), `J` (r²⁰) |
| Extended asphere | `XASPHERE` (any order, normalization radius) | – |
| Odd asphere with only even terms | `ODDASPHE` | – |
| Ideal thin lens | `PARAXIAL` | – |
| Axial spacer | `COORDBRK` with no decenter or tilt | – |

Glasses are catalog glasses from DeepLens' CDGM, SCHOTT, MISC and PLASTIC2022 catalogs and
refractiveindex.info, glasses from `.agf` catalogs you give, or model glasses (nd, vd).

A file that uses anything else is refused with its file, line and surface: tilts, decenters,
mirrors, freeforms, gradient index, diffractive surfaces, zoom configurations, non-circular
apertures, obscurations and curved images. Records that do not change the optics are skipped; an
unknown record is an error unless you read with `strict=False`.

## Use a lens file

As a camera:

```python
from twinrobo import CameraTwin

twin = CameraTwin.from_lens("design.zmx", resolution=(1920, 1080), pixel_pitch_um=2.0)
```

In a spec, the lens file takes the place of `lens.json`. `geometry: lens` takes the focal length
and distortion from the lens, and puts it on a sensor of the spec's resolution and pixel pitch:

```yaml
sensor:
  resolution: {width: 1920, height: 1080}
  pixel_pitch_um: 2.0
lens:
  geometry: lens
  deeplens_model:
    path: design.zmx
    glass_catalogs: [MYGLASS.agf]   # optional
```

From the shell:

```bash
python -m twinrobo.lensio info design.zmx            # prescription and first-order data
python -m twinrobo.lensio convert design.zmx design.seq
python -m twinrobo.lensio convert design.zmx lens.json
```

## How files are read

- **Aperture.** An entrance pupil diameter (`ENPD`, `EPD`) or F-number (`FNUM`, `FNO`) is
  paraxial in OpticStudio and CODE V. TwinRobo sizes the stop so its paraxial entrance pupil has
  that diameter. "Float by stop size" uses the stop's semi-diameter.
- **Glasses.** A glass is looked up by name in the catalog the file names (CODE V's
  `NBK7_SCHOTT`, Zemax's `GCAT` order), then in the others. The nd and vd that a `.zmx` stores
  beside a catalog glass are not used: they can be out of date. A glass in no catalog is an error
  unless you pass its `.agf`, or `allow_model_glass=True` to use the file's nd and vd. That
  approximates the glass's dispersion, and is listed in the prescription's `notes`.
- **Missing sizes.** A surface with no semi-diameter gets the paraxial marginal plus chief ray
  height, with a 10% margin. An image with no size gets the paraxial height of the edge field.
  Both are listed in `notes`.
- **Exported files** name each glass's catalog, because one name can be two glasses: `F2` is in
  both SCHOTT and CDGM.

## Validation

`tools/validate/lensio_rayoptics.py` exports the catalog's lenses and a test triplet to `.zmx` and
`.seq`, reads them back, and traces real rays through them with DeepLens. It then has
[RayOptics](https://github.com/mjhoptics/ray-optics), an independent program with its own file
readers, glass data and tracer, read the same files and trace the same rays. The script fails if
a ray traced by one is missed by the other, or if image heights or focal lengths differ by more
than 0.1 µm. Image heights agree within 0.01 µm, and focal lengths within 0.03 µm.

RayOptics' `.zmx` reader ignores a file's `GCAT` catalog order, so the script restores the named
catalog's glass where RayOptics picked another catalog's glass of the same name, and reports it.

This shows that TwinRobo and RayOptics read the files the same way. It is not yet a comparison with
OpticStudio or CODE V themselves. These have not been checked against files the two programs
write:

- the layout of the `.zmx` records, which has no public grammar;
- whether OpticStudio and CODE V open the files TwinRobo writes.

## The interface

<span id="twinrobo.lensio"></span>

::: twinrobo.lensio.load_lens

::: twinrobo.lensio.save_lens

::: twinrobo.lensio.convert

::: twinrobo.lensio.prescription.Prescription

::: twinrobo.lensio.prescription.Surface

::: twinrobo.lensio.prescription.Material

::: twinrobo.lensio.materials.GlassCatalogs

::: twinrobo.lensio.deeplens_json.to_deeplens

::: twinrobo.lensio.paraxial.paraxial
