"""First-order (paraxial) properties of a prescription: focal length, pupil, ray heights.

Used to fill in what a file may leave out (surface semi-diameters, the image size) and to
report a lens' first-order data. A y-nu trace at the primary wavelength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .prescription import Prescription


@dataclass
class Paraxial:
    efl: float  # effective focal length (mm)
    bfl: float  # last surface to paraxial focus (mm)
    epd: float  # entrance pupil diameter (mm)
    enp: float  # entrance pupil position, from the first surface (mm)
    fnum: float  # efl / epd
    marginal: list[float]  # marginal ray height at each lens surface (mm)
    chief: list[float]  # chief ray height at each lens surface, at the edge field (mm)
    image_height: float  # paraxial chief ray height on the image surface (mm)
    stop_index: int  # the aperture stop (the first lens surface when the file names none)
    stop_radius: float  # the stop semi-diameter that gives this EPD (mm)


def _trace(p: Prescription, n: list[float], y: float, u: float) -> list[tuple[float, float]]:
    """(height, slope after refraction) at each lens surface and the image, from surface 1."""
    out = []
    surfs = p.surfaces
    for i in range(1, len(surfs)):
        s = surfs[i]
        if i < len(surfs) - 1:
            if s.kind == "paraxial":
                phi = 1.0 / s.focal_length
            else:  # the r^2 aspheric term adds to the paraxial curvature: c + 2 a2
                phi = (n[i] - n[i - 1]) * (s.curvature + 2.0 * s.coeffs.get(2, 0.0))
            u = (n[i - 1] * u - y * phi) / n[i]
        out.append((y, u))
        y = y + s.thickness * u
    return out


def paraxial(p: Prescription, indices: list[float], stop_radius: float | None = None) -> Paraxial:
    """First-order data of ``p``; ``indices[i]`` is the index after surface ``i``.

    The entrance pupil follows the file's aperture: its EPD, its F-number (at infinite
    conjugates), or the stop's semi-diameter (``stop_radius`` when given).
    """
    surfs = p.surfaces
    n = indices
    stop = p.stop_index if p.stop_index is not None else 1
    a = _trace(p, n, 1.0, 0.0)  # parallel ray
    b = _trace(p, n, 0.0, 1.0)  # ray through the first vertex
    y_last, u_last = a[-2][0], a[-2][1]  # at the last lens surface, after refraction
    efl = -1.0 / u_last if u_last else math.inf
    bfl = -y_last / u_last if u_last else math.inf
    ay, by = a[stop - 1][0], b[stop - 1][0]
    enp = by / ay if ay else 0.0  # where the chief ray crosses the axis, from surface 1
    d = p.object_distance
    if p.aperture_type == "epd":
        epd = p.aperture_value
    elif p.aperture_type == "fnum":
        epd = abs(efl) / p.aperture_value
    else:
        r = stop_radius if stop_radius is not None else surfs[stop].aperture_radius
        if r is None:
            raise ValueError("the stop has no semi-diameter or clear aperture")
        # The axial ray that reaches the stop's edge, measured at the entrance pupil.
        if math.isinf(d):
            epd = 2 * abs(r / ay)
        else:  # from the object point, slope u0: height u0 * (ay * d + by) at the stop
            u0 = r / (ay * d + by)
            epd = 2 * abs(u0 * (d + enp))
    # Marginal ray: from the axial object point to the pupil's edge.
    if math.isinf(d):
        mar = _trace(p, n, epd / 2, 0.0)
    else:
        u0 = (epd / 2) / (d + enp)
        mar = _trace(p, n, u0 * d, u0)
    stop_radius = abs(mar[stop - 1][0])
    # Chief ray: through the pupil's center at the edge field.
    # The edge field: the farthest from the axis, in x or y (the lens is rotationally symmetric).
    fy = max((math.hypot(x, y) for x, y in p.fields), default=0.0)
    if p.field_type == "angle":
        u0 = max(
            (math.hypot(math.tan(math.radians(x)), math.tan(math.radians(y))) for x, y in p.fields),
            default=0.0,
        )
    elif p.field_type == "object_height":
        u0 = fy / (d + enp) if math.isfinite(d) else 0.0
    else:  # image heights: scale a unit-slope chief ray to land at that height
        unit = _trace(p, n, -enp * 1.0, 1.0)[-1][0]
        u0 = fy / unit if unit else 0.0
    chief = _trace(p, n, -enp * u0, u0)
    return Paraxial(
        efl=efl,
        bfl=bfl,
        epd=epd,
        enp=enp,
        fnum=abs(efl) / epd if epd else math.inf,
        marginal=[abs(y) for y, _ in mar[:-1]],
        chief=[abs(y) for y, _ in chief[:-1]],
        image_height=abs(chief[-1][0]),
        stop_index=stop,
        stop_radius=stop_radius,
    )
