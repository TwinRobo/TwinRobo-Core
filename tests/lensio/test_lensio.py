"""Lens design files: reading, writing, glasses, and rendering an imported lens."""

import math
from pathlib import Path

import pytest

from twinrobo import lensio
from twinrobo.exceptions import LensFileError
from twinrobo.lensio.codev import parse_seq
from twinrobo.lensio.zemax import decode, format_zmx, parse_zmx

DATA = Path(__file__).resolve().parent.parent / "data" / "lensio"
CATALOG = Path(__file__).resolve().parents[2] / "twinrobo" / "catalog"
TRIPLET = {ext: DATA / f"triplet.{ext}" for ext in ("csv", "seq", "zmx")}


def norm(name):
    return None if name is None else "".join(c for c in name.upper() if c.isalnum())


def assert_same(a: lensio.Prescription, b: lensio.Prescription, rel=1e-9):
    """Two prescriptions describe the same lens (system data and every surface)."""
    assert len(a.surfaces) == len(b.surfaces)
    assert a.aperture_type == b.aperture_type
    assert a.aperture_value == pytest.approx(b.aperture_value, rel=rel)
    assert a.field_type == b.field_type
    assert [y for _, y in a.fields] == pytest.approx([y for _, y in b.fields], rel=rel)
    assert a.primary_wavelength_um == pytest.approx(b.primary_wavelength_um, rel=rel)
    assert sorted(a.wavelengths_um) == pytest.approx(sorted(b.wavelengths_um), rel=rel)
    for i, (s, t) in enumerate(zip(a.surfaces, b.surfaces, strict=True)):
        msg = f"surface {i}"
        assert s.kind == t.kind, msg
        assert s.radius == pytest.approx(t.radius, rel=rel), msg
        assert s.thickness == pytest.approx(t.thickness, rel=rel), msg
        assert s.conic == pytest.approx(t.conic, rel=rel, abs=1e-15), msg
        assert sorted(s.coeffs) == sorted(t.coeffs), msg
        for k in s.coeffs:
            assert s.coeffs[k] == pytest.approx(t.coeffs[k], rel=rel), msg
        assert s.stop == t.stop, msg
        assert s.aperture_radius == pytest.approx(t.aperture_radius, rel=rel), msg
        assert norm(s.material.name) == norm(t.material.name), msg
        assert s.material.is_model == t.material.is_model, msg
        if s.material.is_model:
            assert (s.material.nd, s.material.vd) == pytest.approx((t.material.nd, t.material.vd))


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def test_hand_written_files_agree():
    """The same lens typed by hand as a table, a CODE V file and a Zemax file."""
    csv, seq, zmx = (lensio.load_lens(TRIPLET[e]) for e in ("csv", "seq", "zmx"))
    assert_same(csv, seq)
    assert_same(csv, zmx)
    assert zmx.name.startswith("TwinRobo test triplet")
    assert zmx.stop_index == 5 and math.isinf(zmx.object_distance)
    assert zmx.surfaces[6].coeffs == {4: -2e-6, 6: 1e-9} and zmx.surfaces[6].conic == -0.5
    assert zmx.wavelengths_um == [0.4861, 0.5876, 0.6563]  # FTYP: 3 of the 4 WAVM slots
    assert seq.surfaces[3].material.catalog == "SCHOTT"


@pytest.mark.parametrize(
    "encode",
    [
        lambda t: t.encode("utf-16"),  # OpticStudio's default, with a byte order mark
        lambda t: t.encode("utf-16-le"),
        lambda t: t.replace("\n", "\r\n").encode("utf-8"),
        lambda t: t.encode("cp1252"),
    ],
)
def test_zmx_encodings(encode):
    text = TRIPLET["zmx"].read_text()
    assert_same(parse_zmx(decode(encode(text))), lensio.load_lens(TRIPLET["zmx"]))


def zmx(*surfaces: str, head: str = "") -> str:
    """A minimal .zmx: object at infinity, the given surfaces, a flat image."""
    body = "".join(f"SURF {i}\n{s}\n" for i, s in enumerate(surfaces, start=1))
    n = len(surfaces) + 1
    return (
        f"MODE SEQ\nUNIT MM X W X CM MR CPMM\nENPD 4\n{head}"
        "SURF 0\n  TYPE STANDARD\n  CURV 0\n  DISZ INFINITY\n"
        f"{body}SURF {n}\n  TYPE STANDARD\n  CURV 0\n  DISZ 0\n"
    )


LENS1 = "  STOP\n  CURV 0.05\n  DISZ 3\n  GLAS N-BK7 0 0 1.5168 64.17\n  DIAM 4 1"
LENS2 = "  CURV -0.01\n  DISZ 40\n  DIAM 4 1"


def test_zemax_units():
    """UNIT IN: lengths in inches, aspheric terms scaled to millimeters."""
    mm = parse_zmx(zmx(LENS1, "  TYPE EVENASPH\n  CURV -0.01\n  PARM 2 1e-4\n  DISZ 40"))
    inch = parse_zmx(
        zmx(
            "  STOP\n  CURV 1.27\n  DISZ 0.11811023622047244\n  GLAS N-BK7 0 0 1.5 64\n"
            "  DIAM 0.15748031496062992 1",
            "  TYPE EVENASPH\n  CURV -0.254\n  PARM 2 1.638706\n  DISZ 1.574803149606299",
        )
        .replace("UNIT MM", "UNIT IN")
        .replace("ENPD 4", "ENPD 0.15748031496062992")
    )
    for s, t in zip(mm.lens_surfaces, inch.lens_surfaces, strict=True):
        for r in (0.0, 1.0, 3.0):
            assert t.sag(r) == pytest.approx(s.sag(r), rel=1e-6, abs=1e-12)
        assert t.thickness == pytest.approx(s.thickness)
    assert inch.aperture_value == pytest.approx(4.0)


def test_extended_asphere_normalization():
    """XASPHERE: XDAT 1 terms, XDAT 2 normalization radius, XDAT i+2 on (r / Rn)^(2i)."""
    p = parse_zmx(
        zmx(
            LENS1,
            "  TYPE XASPHERE\n  CURV -0.01\n  XDAT 1 3\n  XDAT 2 2.5\n  XDAT 3 1e-3\n"
            "  XDAT 4 -2e-4\n  XDAT 5 5e-5\n  XDAT 6 9\n  DISZ 40",  # XDAT 6 beyond 3 terms
        )
    )
    s = p.surfaces[2]
    c = -0.01
    for r in (0.5, 1.5, 3.0):
        base = c * r * r / (1 + math.sqrt(1 - c * c * r * r))
        rho = r / 2.5
        assert s.sag(r) == pytest.approx(base + 1e-3 * rho**2 - 2e-4 * rho**4 + 5e-5 * rho**6)


def test_odd_asphere_with_even_terms_only():
    p = parse_zmx(zmx(LENS1, "  TYPE ODDASPHE\n  CURV -0.01\n  PARM 4 2e-5\n  DISZ 40"))
    assert p.surfaces[2].coeffs == {4: 2e-5}
    with pytest.raises(LensFileError, match="odd aspheric"):
        parse_zmx(zmx(LENS1, "  TYPE ODDASPHE\n  CURV -0.01\n  PARM 3 2e-5\n  DISZ 40"))


def test_identity_coordinate_break_folds_into_the_gap():
    p = parse_zmx(zmx(LENS1, "  TYPE COORDBRK\n  DISZ 5", LENS2))
    assert len(p.surfaces) == 4
    assert p.surfaces[1].thickness == pytest.approx(8.0)


@pytest.mark.parametrize(
    "surfaces,head,match",
    [
        ((LENS1, "  TYPE COORDBRK\n  PARM 3 5\n  DISZ 5", LENS2), "", "tilted or decentered"),
        ((LENS1, "  CURV -0.01\n  DISZ 40\n  GLAS MIRROR 0 0"), "", "mirrors"),
        ((LENS1, "  TYPE TOROIDAL\n  DISZ 40"), "", "TOROIDAL"),
        ((LENS1, LENS2), "CONF 2\n", "multi-configuration"),
        ((LENS1, LENS2), "OBNA 0.1 0\n", "numerical aperture"),
        ((LENS1, LENS2 + "\n  SQAP 1 1"), "", "rectangular"),
        ((LENS1, LENS2), "ZZZZ 1\n", "unknown record ZZZZ"),
        ((LENS1, LENS2 + "\n  STOP"), "", "more than one aperture stop"),
    ],
)
def test_zemax_rejects(surfaces, head, match):
    with pytest.raises(LensFileError, match=match):
        parse_zmx(zmx(*surfaces, head=head))


def test_zemax_lenient_reads_unknown_records():
    p = parse_zmx(zmx(LENS1, LENS2, head="ZZZZ 1\n"), strict=False)
    assert any("ZZZZ" in n for n in p.notes)


def test_curved_image_rejected():
    text = zmx(LENS1, LENS2).replace("SURF 3\n  TYPE STANDARD\n  CURV 0", "SURF 3\n  CURV 0.01")
    with pytest.raises(LensFileError, match="image surface must be flat"):
        parse_zmx(text)


def seq(*rows: str) -> str:
    return "\n".join(["EPD 4", "SO 0 1e10", *rows, "SI 0 0"]) + "\n"


def test_codev_asphere_letters():
    """A is r^4 ... H is r^18, J is r^20: CODE V has no I."""
    letters = "ABCDEFGHJ"
    rows = ["S 20 3 NBK7_SCHOTT", "  STO; CIR 4", "S -100 40", "  ASP", "  K -1"]
    rows += [f"  {L} {i + 1}e-9" for i, L in enumerate(letters)]
    p = parse_seq(seq(*rows))
    assert p.surfaces[2].coeffs == pytest.approx({4 + 2 * i: (i + 1) * 1e-9 for i in range(9)})
    assert p.surfaces[2].conic == -1
    with pytest.raises(LensFileError, match="unknown command I"):
        parse_seq(seq("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40", "  ASP", "  I 1e-9"))


def test_codev_units_and_curvature_mode():
    mm = parse_seq(seq("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40"))
    cm = parse_seq("DIM C\nRDM N\n" + seq("S 0.5 0.3 NBK7_SCHOTT", "  STO", "S -0.1 4"))
    for s, t in zip(mm.surfaces, cm.surfaces, strict=True):
        assert t.radius == pytest.approx(s.radius) and t.thickness == pytest.approx(s.thickness)
    assert cm.aperture_value == pytest.approx(40.0)  # EPD 4 cm


def test_codev_fictitious_glasses():
    p = parse_seq(seq("S 20 3 1.5168:64.17", "  STO", "S -100 2 6034.5617", "S -50 40"))
    assert (p.surfaces[1].material.nd, p.surfaces[1].material.vd) == (1.5168, 64.17)
    assert p.surfaces[2].material.nd == pytest.approx(1.6034)
    assert p.surfaces[2].material.vd == pytest.approx(56.17)


@pytest.mark.parametrize(
    "rows,match",
    [
        (("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40", "  ADE 5"), "tilted"),
        (("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40", "ZOO 2"), "zoom"),
        (("S 20 3 NBK7_SCHOTT", "  STO 4", "S -100 40"), "STO with data"),
        (("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40", "THI S1 2"), "qualifiers"),
        (("S 20 3 REFL", "  STO", "S -100 40"), "mirrors"),
        (("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40", "  CIR EDG 3"), "edge apertures"),
    ],
)
def test_codev_rejects(rows, match):
    with pytest.raises(LensFileError, match=match):
        parse_seq(seq(*rows))


def test_codev_image_defocus_moves_the_image():
    p = parse_seq(seq("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40").replace("SI 0 0", "SI 0 0.5"))
    assert p.surfaces[-2].thickness == pytest.approx(40.5)


# ----------------------------------------------------------------------
# Writing: every format reads back as the same lens
# ----------------------------------------------------------------------
LENSES = [TRIPLET["csv"], *sorted(CATALOG.glob("**/lens.json"))]


@pytest.mark.parametrize("src", LENSES, ids=lambda p: str(p.parent.name) + "/" + p.name)
@pytest.mark.parametrize("ext", ["zmx", "seq", "csv"])
def test_round_trip(src, ext, tmp_path):
    pytest.importorskip("deeplens")
    p = lensio.load_lens(src)
    out = tmp_path / f"lens.{ext}"
    lensio.save_lens(p, out)
    q = lensio.load_lens(out)
    if ext == "seq" and p.aperture_type == "float":  # CODE V has no float-by-stop-size
        assert q.aperture_type == "epd"
        p = p.copy()
        p.aperture_type, p.aperture_value = "epd", q.aperture_value
    assert_same(p, q)


def test_zmx_written_in_utf16(tmp_path):
    lensio.save_lens(lensio.load_lens(TRIPLET["csv"]), tmp_path / "a.zmx")
    assert (tmp_path / "a.zmx").read_bytes()[:2] == b"\xff\xfe"


def test_export_names_each_glass_catalog(tmp_path):
    """F2 is in SCHOTT and CDGM: exported files say which."""
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    lensio.save_lens(p, tmp_path / "a.seq")
    lensio.save_lens(p, tmp_path / "a.zmx")
    assert " F2_SCHOTT" in (tmp_path / "a.seq").read_text()
    assert "GCAT SCHOTT" in (tmp_path / "a.zmx").read_text(encoding="utf-16")


def test_cannot_write_what_the_format_cannot_hold():
    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[6].coeffs[22] = 1e-15  # beyond CODE V's J (r^20)
    from twinrobo.lensio.codev import format_seq

    with pytest.raises(LensFileError, match="r\\^4 .. r\\^20"):
        format_seq(p)
    assert "TYPE XASPHERE" in format_zmx(p)  # Zemax holds it as an extended asphere


# ----------------------------------------------------------------------
# Glasses
# ----------------------------------------------------------------------
@pytest.fixture
def catalogs():
    pytest.importorskip("deeplens")
    return lensio.GlassCatalogs()


def index(key, w=0.5876):
    from deeplens.material.materials import Material

    return Material(key).refractive_index(w)


def test_glass_catalog_is_respected(catalogs):
    schott = catalogs.resolve(lensio.Material(name="F2", catalog="SCHOTT"))
    cdgm = catalogs.resolve(lensio.Material(name="F2", catalog="CDGM"))
    assert index(schott) == pytest.approx(1.62004, abs=2e-5)
    assert index(cdgm) == pytest.approx(1.61293, abs=2e-5)
    assert catalogs.resolve(lensio.Material(name="F2")) == schott  # DeepLens' own precedence
    assert catalogs.resolve(lensio.Material(name="F2"), ["CDGM"]) == cdgm  # a file's GCAT
    assert catalogs.resolve(lensio.Material(name="NBK7", catalog="SCHOTT")) == "n-bk7"


def test_unknown_glass(catalogs):
    from twinrobo.lensio.materials import deeplens_material

    glass = lensio.Material(name="NOT-A-GLASS", nd=1.6, vd=40.0)
    with pytest.raises(LensFileError, match="in no catalog"):
        deeplens_material(glass, catalogs)
    notes = []
    key = deeplens_material(glass, catalogs, allow_model_glass=True, notes=notes)
    assert index(key) == pytest.approx(1.6, abs=2e-3) and "approximated" in notes[0]


def test_agf_catalog(tmp_path):
    pytest.importorskip("deeplens")
    agf = tmp_path / "MYGLASS.agf"
    # N-BK7's Sellmeier coefficients (formula 2) under a new name
    agf.write_text(
        "CC My glass\nNM MY-BK7 2 0 1.5168 64.17 0 0\n"
        "CD 1.03961212 6.00069867E-3 0.231792344 2.00179144E-2 1.01046945 103.560653 0 0 0 0\n"
        "LD 0.3 2.5\n"
    )
    cats = lensio.GlassCatalogs([agf])
    key = cats.resolve(lensio.Material(name="MY-BK7"))
    assert index(key) == pytest.approx(index("n-bk7"), abs=1e-6)


# ----------------------------------------------------------------------
# Rendering an imported lens
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def triplet_optics():
    pytest.importorskip("deeplens")
    import torch

    from twinrobo.optics.deeplens import DeepLensOptics

    return {
        ext: DeepLensOptics(
            TRIPLET[ext], sensor_resolution=(320, 240), device="cpu", dtype=torch.float64
        )  # fmt: skip
        for ext in ("csv", "seq", "zmx")
    }


def test_imported_lens_is_the_file(triplet_optics):
    """DeepLens traces the file's surfaces, and its first-order data match the prescription."""
    from twinrobo.lensio.deeplens_json import indices
    from twinrobo.lensio.materials import resolve_all
    from twinrobo.lensio.paraxial import paraxial

    p = lensio.load_lens(TRIPLET["zmx"])
    par = paraxial(p, indices(p, resolve_all(p, lensio.GlassCatalogs())))
    for o in triplet_optics.values():
        L = o._lens
        assert o.focal_length_mm == pytest.approx(par.efl, rel=1e-5)
        assert float(L.d_sensor) == pytest.approx(sum(s.thickness for s in p.lens_surfaces))
        assert L.aper_idx == 4  # the stop: the 5th lens surface
        for s, ref in zip(L.surfaces, p.lens_surfaces, strict=True):
            for r in (0.0, 0.5 * ref.aperture_radius, 0.95 * ref.aperture_radius):
                import torch

                x = torch.tensor([r], dtype=torch.float64)
                assert float(s.sag(x, torch.zeros_like(x))) == pytest.approx(ref.sag(r), abs=1e-9)
    assert o._lens_sha256 == triplet_optics["zmx"]._lens_sha256  # same lens, same cache key


def test_entrance_pupil_is_the_file_epd(triplet_optics):
    """Zemax and CODE V define the EPD paraxially; the stop is sized to give it."""
    L = triplet_optics["zmx"]._lens
    z, r = L.calc_pupil_paraxial(reverse=True)  # the entrance pupil
    assert 2 * r == pytest.approx(10.0, rel=1e-9)
    assert z == pytest.approx(10.7726, abs=1e-4)  # RayOptics: 10.772628 mm behind surface 1


def test_camera_from_lens():
    pytest.importorskip("deeplens")
    from twinrobo import CameraTwin

    cam = CameraTwin.from_lens(
        TRIPLET["zmx"], resolution=(640, 480), pixel_pitch_um=10.0, build_psf=False, device="cpu"
    )
    ref = cam.reference
    assert ref.pixel_pitch_um == pytest.approx(10.0)
    assert ref.sensor_size_mm == (6.4, 4.8)
    assert cam.intrinsics.fx == pytest.approx(ref.focal_length_mm / 0.010)
    assert cam.spec.lens.geometry == "lens"


def test_cli(tmp_path, capsys):
    pytest.importorskip("deeplens")
    from twinrobo.lensio.__main__ import main

    assert main(["info", str(TRIPLET["zmx"])]) == 0
    assert "EFL 48.88" in capsys.readouterr().out
    assert main(["convert", str(TRIPLET["zmx"]), str(tmp_path / "t.seq")]) == 0
    assert_same(lensio.load_lens(tmp_path / "t.seq"), lensio.load_lens(TRIPLET["zmx"]))


# ----------------------------------------------------------------------
# Regressions from review
# ----------------------------------------------------------------------
BK7_CD = "1.03961212 6.00069867E-3 0.231792344 2.00179144E-2 1.01046945 103.560653 0 0 0 0"
F2_CD = "1.34533359 9.97743871E-3 0.209073176 4.70450767E-2 0.937357162 111.886764 0 0 0 0"


def agf(path, name, cd, nd=1.5168):
    path.write_text(f"NM {name} 2 0 {nd} 64.17 0 0\nCD {cd}\nLD 0.3 2.5\n")
    return path


def test_changed_catalog_glass_changes_the_key(tmp_path):
    """Same name, other coefficients: another key, so another cache key (no stale glass)."""
    pytest.importorskip("deeplens")
    a = lensio.GlassCatalogs([agf(tmp_path / "MINE.agf", "G1", BK7_CD)])
    key_a = a.resolve(lensio.Material(name="G1"))
    b = lensio.GlassCatalogs({"MINE": agf(tmp_path / "MINE2.agf", "G1", F2_CD)})
    key_b = b.resolve(lensio.Material(name="G1"))
    assert key_a != key_b and index(key_a) != pytest.approx(index(key_b), abs=1e-3)
    assert a.resolve(lensio.Material(name="G1")) == key_a  # the first instance is unchanged


def test_given_catalog_replaces_a_bundled_one(tmp_path):
    pytest.importorskip("deeplens")
    cats = lensio.GlassCatalogs([agf(tmp_path / "SCHOTT.agf", "NEW-GLASS", BK7_CD)])
    assert cats.resolve(lensio.Material(name="NEW-GLASS", catalog="SCHOTT")) is not None
    assert cats.resolve(lensio.Material(name="N-BK7", catalog="SCHOTT")) == "n-bk7"


def two_f2(cat2: str) -> lensio.Prescription:
    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[3].material = lensio.Material(name="F2", catalog="SCHOTT")
    p.surfaces[1].material = lensio.Material(name="F2", catalog=cat2)
    return p


def test_zmx_export_orders_catalogs_or_refuses(tmp_path):
    pytest.importorskip("deeplens")
    with pytest.raises(LensFileError, match="cannot tell them apart"):
        lensio.save_lens(two_f2("CDGM"), tmp_path / "a.zmx")
    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[3].material = lensio.Material(name="F2", catalog="CDGM")  # beside SCHOTT N-SK16
    lensio.save_lens(p, tmp_path / "b.zmx")
    q = lensio.load_lens(tmp_path / "b.zmx")
    assert q.glass_catalogs[0] == "CDGM"
    cats = lensio.GlassCatalogs()
    assert cats.locate(q.surfaces[3].material, q.glass_catalogs)[0] == "CDGM"
    lensio.save_lens(two_f2("CDGM"), tmp_path / "c.seq")  # CODE V names each glass's catalog
    r = lensio.load_lens(tmp_path / "c.seq")
    assert {r.surfaces[1].material.catalog, r.surfaces[3].material.catalog} == {"CDGM", "SCHOTT"}


def first_order(p):
    from twinrobo.lensio.deeplens_json import indices
    from twinrobo.lensio.materials import resolve_all
    from twinrobo.lensio.paraxial import paraxial

    return paraxial(p, indices(p, resolve_all(p, lensio.GlassCatalogs())))


def test_paraxial_includes_the_r2_term():
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    q = p.copy()
    s = q.surfaces[1]
    a2 = 0.004
    s.kind, s.coeffs = "asphere", {2: a2}
    r = p.copy()
    r.surfaces[1].radius = 1 / (1 / p.surfaces[1].radius + 2 * a2)
    assert first_order(q).efl == pytest.approx(first_order(r).efl, rel=1e-12)


def test_float_aperture_uses_the_clear_aperture():
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.aperture_type, p.aperture_value = "float", None
    stop = p.surfaces[p.stop_index]
    stop.semi_diameter, stop.clear_aperture = 5.0, 2.0
    q = p.copy()
    q.surfaces[q.stop_index].semi_diameter, q.surfaces[q.stop_index].clear_aperture = 2.0, None
    assert first_order(p).epd == pytest.approx(first_order(q).epd)


def test_x_fields_count(tmp_path):
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.fields = [(0.0, 0.0), (10.0, 0.0)]
    q = p.copy()
    q.fields = [(0.0, 0.0), (0.0, 10.0)]
    assert first_order(p).image_height == pytest.approx(first_order(q).image_height)
    lensio.save_lens(p, tmp_path / "x.csv")
    assert lensio.load_lens(tmp_path / "x.csv").fields == [(0.0, 0.0), (10.0, 0.0)]


def test_codev_mixed_quotes():
    p = parse_seq(
        'EPD 4\nTITLE "start"; EPD 8; TITLE "Bob\'s lens" ! a comment; EPD 1\n'
        + seq("S 20 3 NBK7_SCHOTT", "  STO", "S -100 40")[len("EPD 4\n") :]
    )
    assert p.aperture_value == 8 and p.name == "Bob's lens"


def test_codev_export_resolves_with_the_given_catalogs(tmp_path):
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.aperture_type, p.aperture_value = "float", None
    p.surfaces[1].material = lensio.Material(name="G1", nd=1.6, vd=40.0)
    with pytest.raises(LensFileError, match="in no catalog"):
        lensio.save_lens(p, tmp_path / "a.seq")  # no catalog has G1, no approximation allowed
    cats = lensio.GlassCatalogs([agf(tmp_path / "MINE.agf", "G1", BK7_CD)])
    lensio.save_lens(p, tmp_path / "b.seq", catalogs=cats)
    assert " G1_MINE" in (tmp_path / "b.seq").read_text()


def test_converted_json_carries_its_glasses(tmp_path):
    """A lens using a given .agf glass, saved as DeepLens JSON, loads in a later process."""
    pytest.importorskip("deeplens")
    from deeplens.material import materials

    from twinrobo.optics.deeplens import DeepLensOptics

    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[1].material = lensio.Material(name="G1")
    cats = lensio.GlassCatalogs([agf(tmp_path / "MINE.agf", "G1", BK7_CD)])
    lensio.save_lens(p, tmp_path / "lens.json", catalogs=cats)
    for k in [k for k in materials.MATERIAL_data if k.startswith("agf:mine:")]:
        del materials.MATERIAL_data[k]  # as in a fresh process
    o = DeepLensOptics(tmp_path / "lens.json", device="cpu")
    assert o.focal_length_mm > 0
    assert lensio.load_lens(tmp_path / "lens.json").surfaces[1].material.catalog == "MINE"


def test_camera_from_lens_without_an_image_size(tmp_path):
    pytest.importorskip("deeplens")
    from twinrobo import CameraTwin

    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[-1].semi_diameter = None
    p.fields = [(0.0, 0.0)]
    lensio.save_lens(p, tmp_path / "t.csv")
    cam = CameraTwin.from_lens(
        tmp_path / "t.csv", resolution=(400, 300), pixel_pitch_um=5.0, build_psf=False, device="cpu"
    )
    assert cam.reference.sensor_size_mm == (2.0, 1.5)


def test_json_reexport_keeps_embedded_glasses(tmp_path):
    """A JSON lens with a given glass converts to other formats with that glass."""
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[1].material = lensio.Material(name="N-BK7", catalog="SCHOTT")
    cats = lensio.GlassCatalogs([agf(tmp_path / "SCHOTT.agf", "N-BK7", F2_CD, nd=1.62)])
    lensio.save_lens(p, tmp_path / "lens.json", catalogs=cats)  # SCHOTT's N-BK7, replaced
    q = lensio.load_lens(tmp_path / "lens.json")
    lensio.save_lens(q, tmp_path / "again.json")
    q = lensio.load_lens(tmp_path / "again.json")
    from twinrobo.lensio.deeplens_json import indices
    from twinrobo.lensio.materials import resolve_all

    n_q = indices(q, resolve_all(q, lensio.catalogs_for(q)))[1]
    assert n_q == pytest.approx(
        index(cats.resolve(lensio.Material(name="N-BK7", catalog="SCHOTT")))
    )
    assert n_q != pytest.approx(index("n-bk7"), abs=1e-3)


def test_codev_title_with_quotes_round_trips(tmp_path):
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.name = "Bob's lens; prototype"
    lensio.save_lens(p, tmp_path / "t.seq")
    assert lensio.load_lens(tmp_path / "t.seq").name == "Bob's lens; prototype"


def test_json_reexport_keeps_bare_bundled_glasses(tmp_path):
    """An embedded CDGM F2 beside DeepLens' bare (SCHOTT) f2: each keeps its own glass."""
    pytest.importorskip("deeplens")
    p = two_f2("CDGM")
    lensio.save_lens(p, tmp_path / "lens.json")
    q = lensio.load_lens(tmp_path / "lens.json")
    assert [s.material.catalog for s in (q.surfaces[1], q.surfaces[3])] == ["CDGM", "SCHOTT"]
    assert first_order(q).efl == pytest.approx(first_order(p).efl, rel=1e-12)


@pytest.mark.parametrize("ext", ["seq", "zmx", "csv"])
def test_title_line_breaks_cannot_inject_records(ext, tmp_path):
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.name = "Triplet\nRDM N\n!"
    lensio.save_lens(p, tmp_path / f"t.{ext}")
    q = lensio.load_lens(tmp_path / f"t.{ext}")
    assert q.surfaces[1].radius == pytest.approx(22.0)


def test_one_name_for_two_glasses_is_refused(tmp_path):
    """SCHOTT's N-BK7 beside a replacement SCHOTT N-BK7: refused, never merged."""
    pytest.importorskip("deeplens")
    p = lensio.load_lens(TRIPLET["csv"])
    p.surfaces[1].material = lensio.Material(name="N-BK7", catalog="SCHOTT")
    lensio.save_lens(p, tmp_path / "a.json")
    data = __import__("json").loads((tmp_path / "a.json").read_text())
    data["surfaces"][2]["mat2"] = "n-bk7"  # a second N-BK7: DeepLens' own
    cats = lensio.GlassCatalogs([agf(tmp_path / "SCHOTT.agf", "N-BK7", F2_CD, nd=1.62)])
    data["surfaces"][0]["mat2"] = cats.resolve(lensio.Material(name="N-BK7", catalog="SCHOTT"))
    data["glass_data"] = cats.glass_data([data["surfaces"][0]["mat2"]])
    (tmp_path / "b.json").write_text(__import__("json").dumps(data))
    with pytest.raises(LensFileError, match="cannot be told apart"):
        lensio.load_lens(tmp_path / "b.json")


def test_two_saved_glasses_with_one_name_are_refused(tmp_path):
    pytest.importorskip("deeplens")
    import json

    a = lensio.GlassCatalogs([agf(tmp_path / "SCHOTT.agf", "N-BK7", BK7_CD)])
    b = lensio.GlassCatalogs({"SCHOTT": agf(tmp_path / "other.agf", "N-BK7", F2_CD, nd=1.62)})
    ka = a.resolve(lensio.Material(name="N-BK7", catalog="SCHOTT"))
    kb = b.resolve(lensio.Material(name="N-BK7", catalog="SCHOTT"))
    p = lensio.load_lens(TRIPLET["csv"])
    lensio.save_lens(p, tmp_path / "t.json")
    data = json.loads((tmp_path / "t.json").read_text())
    data["surfaces"][0]["mat2"], data["surfaces"][5]["mat2"] = ka, kb
    data["glass_data"] = {**a.glass_data([ka]), **b.glass_data([kb])}
    (tmp_path / "t.json").write_text(json.dumps(data))
    with pytest.raises(LensFileError, match="cannot be told apart"):
        lensio.load_lens(tmp_path / "t.json")
