"""Hello CameraTwin: load a CameraSpec, inspect its DeepLens optics, generate PSFs.

Phase 0 exit criterion: DeepLensOptics loads a DeepLens model.
The runtime renderer (`CameraTwin.from_spec(...).optics`) is shown in example 03.

    python examples/00_hello_twinrobo.py
"""

from pathlib import Path

import torch

from twinrobo import CameraSpec
from twinrobo.camera import build_reference_optics
from twinrobo.registry import BUILTIN_CATALOG

ROOT = Path(__file__).resolve().parents[1]
SPEC = BUILTIN_CATALOG / "examples" / "cellphone80deg" / "camera.yaml"


def main():
    spec = CameraSpec.from_yaml(SPEC)
    optics = build_reference_optics(spec)  # DeepLensOptics
    print(f"Camera:      {spec.id} ({spec.validation.status})")
    print(f"Lens file:   {optics.model_path.name}")
    print(f"Device:      {optics.device}")
    print(
        f"Sensor:      {optics.sensor_resolution[0]}x{optics.sensor_resolution[1]} px, "
        f"pitch {optics.pixel_pitch_um:.2f} um"
    )
    print(f"Focal len:   {optics.focal_length_mm:.3f} mm, F/{optics.f_number:.2f}")

    field = [[0.0, 0.0], [0.5, 0.0], [0.9, 0.9]]
    depths_m = [0.5, 2.0, 20.0]
    psf = optics.generate_psf(field, depths_m, spp=4096)
    print(f"PSF bank:    {tuple(psf.shape)}  [depth, field, rgb, ks, ks] on {psf.device}")
    sums = psf.sum(dim=(-1, -2))
    print(f"PSF sums:    min {sums.min().item():.4f}  max {sums.max().item():.4f}")

    # Energy-weighted RMS radius (pixels) of the green PSF, per depth x field.
    ks = psf.shape[-1]
    yy, xx = torch.meshgrid(torch.arange(ks), torch.arange(ks), indexing="ij")
    yy, xx = yy.to(psf), xx.to(psf)
    g = psf[:, :, 1]
    cy = (g * yy).sum((-1, -2), keepdim=True) / g.sum((-1, -2), keepdim=True)
    cx = (g * xx).sum((-1, -2), keepdim=True) / g.sum((-1, -2), keepdim=True)
    r = ((g * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum((-1, -2)) / g.sum((-1, -2))).sqrt()
    for i, d in enumerate(depths_m):
        cells = "  ".join(f"{v:6.2f}" for v in r[i].tolist())
        print(f"RMS radius @ {d:5.1f} m (fields {field}): {cells} px")


if __name__ == "__main__":
    main()
