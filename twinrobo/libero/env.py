"""LIBERO env wrapper that swaps camera observations for CameraTwin frames."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..camera import CameraTwin
from ..mujoco import MujocoCameraTwin, RobosuiteRenderer, to_uint8


def _image_convention_flip() -> bool:
    """True if robosuite returns images bottom-up (its default ``opengl`` convention)."""
    from robosuite import macros

    return macros.IMAGE_CONVENTION == "opengl"


class CameraTwinLiberoEnv:
    """Wrap a LIBERO env (``OffScreenRenderEnv``) so its cameras are CameraTwin cameras.

    Each camera ``name`` in ``cameras`` (a robosuite camera such as
    ``"agentview"``) is rendered at the twin's resolution and FoV from the same
    camera pose, run through the twin's optics, and written back into
    ``obs[f"{name}_image"]`` at the key's original size, dtype and orientation.
    Code downstream of the env is unchanged.

    Args:
        env (OffScreenRenderEnv): LIBERO env, e.g. from `twinrobo.libero.make_env`.
        cameras: ``{robosuite camera name: CameraTwin}``.
        crop: How the twin frame (e.g. 16:10) becomes the observation (e.g.
            square): ``"center"`` crops the centered region with the output
            aspect ratio, then resizes. ``"none"`` resizes and stretches.
        match_fov: Render with the twin's paraxial FoV. ``False`` keeps LIBERO's
            camera FoV, applying only the optics.
        keep_ideal: Also return the untouched observation as ``f"{name}_image_ideal"``.
    """

    def __init__(
        self,
        env,
        cameras: dict[str, CameraTwin],
        crop: str = "center",
        match_fov: bool = True,
        keep_ideal: bool = False,
    ):
        self.env = env
        self.crop = crop
        self.keep_ideal = keep_ideal
        self.flip = _image_convention_flip()
        backend = RobosuiteRenderer(env)
        self.cameras = {
            name: MujocoCameraTwin(twin, backend, match_fov=match_fov)
            for name, twin in cameras.items()
        }
        self._attached = False

    def _attach(self) -> None:
        for name, cam in self.cameras.items():
            cam.attach(name)  # re-resolve ids: robosuite may rebuild the sim on reset
        self._attached = True

    def render_camera(self, name: str, size: tuple[int, int]) -> np.ndarray:
        """CameraTwin observation of camera ``name``: uint8 ``[h, w, 3]``, robosuite convention."""
        frame = self.cameras[name].get_frame()
        img = to_uint8(frame.rgb, size=size, crop=self.crop)
        return np.ascontiguousarray(img[::-1] if self.flip else img)

    def _replace(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._attached:
            self._attach()
        for name in self.cameras:
            key = f"{name}_image"
            if key not in obs:
                continue
            h, w = obs[key].shape[:2]
            if self.keep_ideal:
                obs[f"{name}_image_ideal"] = obs[key]
            obs[key] = self.render_camera(name, (w, h))
        return obs

    # -- LIBERO / robosuite API ---------------------------------------------------------------
    def reset(self):
        obs = self.env.reset()
        self._attached = False
        return self._replace(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._replace(obs), reward, done, info

    def set_init_state(self, init_state):
        return self._replace(self.env.set_init_state(init_state))

    def regenerate_obs_from_state(self, state):
        return self._replace(self.env.regenerate_obs_from_state(state))

    def close(self):
        return self.env.close()

    def __getattr__(self, item):
        return getattr(self.env, item)
