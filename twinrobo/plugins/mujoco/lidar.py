"""A catalog lidar in a MuJoCo scene (see `twinrobo.lidar`).

The lidar is mounted on a body (``"world"`` for a fixed one) at a pose in the body's frame;
its frame is ROS's (x forward, y left, z up), so ``rpy_deg = (0, 0, 0)`` on the world looks
along world +x with +z up. Its beams hit the geoms the cameras draw (visual groups), each
with its material colour's luminance as reflectivity.

    from twinrobo.lidar import LidarRegistry
    from twinrobo.plugins.mujoco.lidar import MujocoLidar

    lidar = MujocoLidar(LidarRegistry().load("ouster/os1-64"), model, data,
                        body="robot0_link7", pos=(0, 0, 0.1))
    frame = lidar.scan()                       # range / intensity images, points()
    frame = lidar.scan(next_qpos=q1, dt=0.05)  # FMCW lidars: radial velocity too
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ...lidar import LidarSpec, LidarTwin
from ...lidar.scan import luminance
from .lensrender import scene_mesh

WORLD = "world"
VISUAL_GROUPS = {1, 2}  # robosuite: 1 = visual, 0 = collision proxies (overlap the visuals)


def rpy_matrix(rpy_deg) -> np.ndarray:
    """Body-frame rotation of roll, pitch, yaw (degrees; x, y, z axes, applied in that order)."""
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def geom_albedo(model) -> np.ndarray:
    """Each geom's reflectivity proxy: the linear luminance of its colour (a material's, and a
    texture's mean colour when it has one)."""
    import mujoco

    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    tex_mean: dict[int, np.ndarray] = {}
    out = np.zeros(model.ngeom, np.float32)
    for g in range(model.ngeom):
        m = int(model.geom_matid[g])
        rgba = np.asarray(model.mat_rgba[m] if m >= 0 else model.geom_rgba[g], np.float64)
        if m >= 0:
            t = int(model.mat_texid[m, rgb_role])
            if t >= 0:
                if t not in tex_mean:
                    w, h, nc = (int(model.tex_width[t]), int(model.tex_height[t]),
                                int(model.tex_nchannel[t]))  # fmt: skip
                    adr = int(model.tex_adr[t])
                    data = np.asarray(model.tex_data[adr : adr + w * h * nc], np.float64)
                    px = data.reshape(-1, nc)[:, :3] / 255.0 if nc >= 3 else None
                    tex_mean[t] = px.mean(0) if px is not None else np.ones(3)
                rgba = rgba.copy()
                rgba[:3] = rgba[:3] * tex_mean[t]
        out[g] = luminance(rgba[:3])
    return out


class MujocoLidar:
    def __init__(
        self,
        spec: LidarSpec,
        model,
        data,
        body: str = WORLD,
        pos=(0.0, 0.0, 0.0),
        rpy_deg=(0.0, 0.0, 0.0),
        groups: set[int] | None = None,
        device: str | torch.device | None = None,
    ):
        import mujoco

        self.spec, self.model, self.data = spec, model, data
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.twin = LidarTwin(spec, self.device)
        self.body = 0 if body == WORLD else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self.body < 0:
            raise ValueError(f"no body {body!r} to mount the lidar on")
        self.local_p = np.asarray(pos, np.float64)
        self.local_R = rpy_matrix(rpy_deg)
        if groups is None:  # the visual geoms (all drawn ones if the model has no visual group)
            drawn = {int(g) for g in model.geom_group}
            groups = VISUAL_GROUPS & drawn or drawn
        self.groups = groups
        self.mesh = scene_mesh(model, groups, self.device)
        self.albedo = torch.as_tensor(geom_albedo(model), device=self.device)

    def pose(self, data=None) -> tuple[np.ndarray, np.ndarray]:
        """The lidar's world rotation (columns: x, y, z axes) and position."""
        data = data if data is not None else self.data
        Rb = np.asarray(data.xmat[self.body]).reshape(3, 3)
        return Rb @ self.local_R, np.asarray(data.xpos[self.body]) + Rb @ self.local_p

    def _geom_poses(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        R = torch.as_tensor(np.asarray(data.geom_xmat), dtype=torch.float32, device=self.device)
        t = torch.as_tensor(np.asarray(data.geom_xpos), dtype=torch.float32, device=self.device)
        return R.view(-1, 3, 3), t

    def scan(self, frame: int = 0, next_qpos=None, dt: float | None = None):
        """A `LidarFrame` of the current state. ``next_qpos`` (the state ``dt`` seconds later)
        gives FMCW lidars their radial velocity; the data's state is left as it was."""
        import mujoco

        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        self.mesh.update(d)
        R, p = self.pose()
        nxt = sensor1 = None
        if self.spec.fmcw and next_qpos is not None and dt:
            q0 = np.array(d.qpos)
            d.qpos[:] = next_qpos
            mujoco.mj_kinematics(m, d)
            nxt, sensor1 = self._geom_poses(d), self.pose()
            d.qpos[:] = q0
            mujoco.mj_kinematics(m, d)
        return self.twin.scan(self.mesh, self.albedo, R, p, frame, nxt, sensor1, dt)
