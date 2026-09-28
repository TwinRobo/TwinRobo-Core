"""Precompute (or refresh) the PSF bank for a CameraSpec into the PSF cache.

python tools/generate_psf/generate_psf.py twinrobo/catalog/stereolabs/zed-x/2.2mm/camera.yaml
"""

import argparse
import time

import torch

from twinrobo import CameraSpec
from twinrobo.camera import build_reference_optics, psf_bank_params
from twinrobo.optics import PSFCache


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--cache", default=None, help="cache root (default: $TWINROBO_CACHE/psf)")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec = CameraSpec.from_yaml(args.spec)
    optics = build_reference_optics(spec, device=args.device)
    if optics is None:
        raise SystemExit(f"{spec.id}: lens has no deeplens_model; nothing to generate")
    params = psf_bank_params(spec)
    cache = PSFCache(args.cache)
    key = optics.bank_key(**params)
    path = cache.path_for(key)
    if path.is_file() and not args.force:
        print(f"cached: {path}")
        return

    t0 = time.perf_counter()
    bank = optics.build_psf_bank(**params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    cache.put(key, bank)
    gw, gh = bank.grid
    print(
        f"{spec.id}: {bank.num_depths} depths x {gw}x{gh} field x {bank.channels} ch, "
        f"ks={bank.ks} built in {dt:.1f} s -> {path}"
    )


if __name__ == "__main__":
    main()
