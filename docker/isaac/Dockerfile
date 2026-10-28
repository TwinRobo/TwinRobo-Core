# TwinRobo inside NVIDIA Isaac Sim.
#
# Isaac Sim ships its own torch (2.11 in 6.1) in extension folders that pip does not see,
# so a plain install would pull a second torch (and its CUDA libraries). DeepLens (from
# PyPI, deeplens-core>=2.5.6, which accepts torch>=2.10) and its pure-Python dependencies
# are installed without dependencies, so they run on Isaac's torch. The TwinRobo source is mounted at /workspace/twinrobo-core by
# docker/isaac/run.sh, not copied, so edits apply without a rebuild.
ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:6.1.0
FROM ${ISAAC_IMAGE}

USER root
# Cache and output directories owned by the Isaac user, so named volumes mounted on
# them inherit that ownership (Isaac writes shader, Warp and PSF caches there).
RUN mkdir -p /isaac-sim/.cache /isaac-sim/.nv/ComputeCache /isaac-sim/.local/share/ov/data \
        /isaac-sim/.nvidia-omniverse/logs /isaac-sim/.nvidia-omniverse/config /workspace/outputs \
    && chown -R isaac-sim:isaac-sim /isaac-sim/.cache /isaac-sim/.nv /isaac-sim/.local \
        /isaac-sim/.nvidia-omniverse /workspace

RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
        "deeplens-core>=2.5.6" \
        einops lpips scikit-image opencv-python-headless tqdm lazy_loader imageio tifffile \
        pyyaml pytest iniconfig pluggy pygments packaging \
    && chown -R isaac-sim:isaac-sim /isaac-sim/kit/python

USER isaac-sim
# Isaac's torch and its CUDA libraries live in extension folders, not site-packages.
ENV PYTHONPATH=/workspace/twinrobo-core:/isaac-sim/exts/isaacsim.pip.nv/pip_prebundle:/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle \
    PYTHONDONTWRITEBYTECODE=1 \
    TWINROBO_CACHE=/isaac-sim/.cache/twinrobo \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y
WORKDIR /workspace/twinrobo-core
ENTRYPOINT ["/isaac-sim/python.sh"]
