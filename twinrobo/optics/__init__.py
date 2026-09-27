"""Optical image formation backends.

`DeepLensOptics` is imported lazily so that DeepLens is only loaded when used.
"""

from .base import IdentityOptics, OpticsModel
from .cache import PSFCache
from .dense_depth import DenseDepthRenderer
from .psf import PSFBank

__all__ = [
    "OpticsModel",
    "IdentityOptics",
    "DeepLensOptics",
    "DenseDepthRenderer",
    "PSFBank",
    "PSFCache",
]


def __getattr__(name):
    if name == "DeepLensOptics":
        from .deeplens import DeepLensOptics

        return DeepLensOptics
    raise AttributeError(name)
