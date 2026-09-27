#!/usr/bin/env bash
# Run TwinRobo inside Isaac Sim (Docker, headless, GPU).
#
#   docker/isaac/run.sh examples/01_isaac_camera.py
#   docker/isaac/run.sh -m pytest -q tests/isaac
#
# Arguments go to Isaac's python. The repo is mounted read-only; files written to
# /workspace/outputs appear in outputs/isaac/. Isaac's caches (shaders, Warp, PSF
# banks) persist in named Docker volumes (twinrobo-isaac-*), so later runs start fast.
# Builds the image on first use. Env: TWINROBO_ISAAC_IMAGE, ISAAC_IMAGE, DOCKER.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${TWINROBO_ISAAC_IMAGE:-twinrobo-isaac:6.1.0}
DOCKER=${DOCKER:-docker}

if ! $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; then
  $DOCKER build -t "$IMAGE" \
    --build-arg ISAAC_IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}" "$ROOT/docker/isaac"
fi

mkdir -p "$ROOT/outputs/isaac" && chmod 777 "$ROOT/outputs/isaac"  # written by the container user
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
exec $DOCKER run --rm --gpus all "${tty[@]}" \
  -v "$ROOT":/workspace/twinrobo-core:ro \
  -v "$ROOT/outputs/isaac":/workspace/outputs \
  -v twinrobo-isaac-cache:/isaac-sim/.cache \
  -v twinrobo-isaac-computecache:/isaac-sim/.nv/ComputeCache \
  -v twinrobo-isaac-ovdata:/isaac-sim/.local/share/ov/data \
  -v twinrobo-isaac-logs:/isaac-sim/.nvidia-omniverse/logs \
  -v twinrobo-isaac-config:/isaac-sim/.nvidia-omniverse/config \
  "$IMAGE" "$@"
