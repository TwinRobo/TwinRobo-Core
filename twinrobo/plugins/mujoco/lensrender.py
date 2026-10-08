"""Lens-ray rendering of MuJoCo scenes (see `twinrobo.optics.lensrender`).

The methods (``"pupil"``, ``"raycast"``) are simulator independent; this module is
their MuJoCo scene: pupil views rasterized by MuJoCo from the shifted camera, the
camera pose from ``data``, and the drawn geoms as ray-cast triangles.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch

from ...isp.color import srgb_to_linear
from ...optics.lensrender import (  # noqa: F401  (re-exported: the public names live here too)
    METHODS,
    PupilViews,
    TriangleMesh,
    _box,
    _cylinder,
    _sphere,
    lens_rays,
    pupil_centers,
)
from ...optics.lensrender import LensRayRenderer as _LensRayRenderer


@contextmanager
def _camera_fovy(model, cam_id: int, fovy_deg: float):
    old = float(model.cam_fovy[cam_id])
    model.cam_fovy[cam_id] = fovy_deg
    try:
        yield
    finally:
        model.cam_fovy[cam_id] = old


@contextmanager
def _camera_offset(model, data, cam_id: int, delta_cam):
    """Shift a camera by ``delta_cam`` (its own frame, m) for rendering; restored afterwards.

    Renderers read the camera pose from ``data.cam_xpos`` (updated by
    ``mj_camlight``), so only that is touched; the model is unchanged.
    """
    old = np.array(data.cam_xpos[cam_id], copy=True)
    R = np.asarray(data.cam_xmat[cam_id]).reshape(3, 3)
    data.cam_xpos[cam_id] = old + R @ np.asarray(delta_cam, dtype=np.float64)
    try:
        yield
    finally:
        data.cam_xpos[cam_id] = old


# -- scene triangles for ray casting (method B) ---------------------------------------------


def geom_triangles(model, g: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Local-frame triangles of geom ``g`` (as MuJoCo draws it), or None if unsupported."""
    import mujoco

    t, s = int(model.geom_type[g]), np.asarray(model.geom_size[g], np.float32)
    G = mujoco.mjtGeom
    if t == G.mjGEOM_MESH:
        m = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        fa, fn = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
        return np.asarray(model.mesh_vert[va : va + vn], np.float32), np.asarray(
            model.mesh_face[fa : fa + fn], np.int32
        )
    if t == G.mjGEOM_BOX:
        v, f = _box()
        return v * s, f
    if t == G.mjGEOM_PLANE:
        hx, hy = (s[0] or 50.0), (s[1] or 50.0)
        v = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], np.float32)
        return v, np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    if t in (G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID):
        v, f = _sphere()
        return v * (s[:3] if t == G.mjGEOM_ELLIPSOID else s[0]), f
    if t == G.mjGEOM_CYLINDER:
        v, f = _cylinder()
        return v * np.array([s[0], s[0], s[1]], np.float32), f
    if t == G.mjGEOM_CAPSULE:
        cv, cf = _cylinder()
        sv, sf = _sphere()
        top = sv * s[0] + np.array([0, 0, s[1]], np.float32)
        bot = sv * s[0] - np.array([0, 0, s[1]], np.float32)
        v = np.concatenate([cv * np.array([s[0], s[0], s[1]], np.float32), top, bot])
        f = np.concatenate([cf, sf + len(cv), sf + len(cv) + len(sv)])
        return v, f
    return None  # heightfields, SDFs: not ray cast (they are rare in manipulation scenes)


def _geom_alpha(model, g: int) -> float:
    m = int(model.geom_matid[g])
    return float(model.mat_rgba[m][3] if m >= 0 else model.geom_rgba[g][3])


class SceneMesh(TriangleMesh):
    """All drawn geoms of a MuJoCo model as one GPU triangle BVH, refit per frame."""

    def __init__(self, model, groups: set[int], device: torch.device):
        verts, faces, gidx, off = [], [], [], 0
        for g in range(model.ngeom):
            if int(model.geom_group[g]) not in groups or _geom_alpha(model, g) <= 0:
                continue
            tri = geom_triangles(model, g)
            if tri is None:
                continue
            v, f = tri
            verts.append(v)
            faces.append(f + off)
            gidx.append(np.full(len(v), g, np.int64))
            off += len(v)
        super().__init__(np.concatenate(verts), np.concatenate(faces), np.concatenate(gidx), device)
        self.model = model  # held: the cache key is id(model), which must stay unique
        self.model_id = id(model)

    def update(self, data) -> None:
        xpos = torch.as_tensor(np.asarray(data.geom_xpos), dtype=torch.float32, device=self.device)
        xmat = torch.as_tensor(
            np.asarray(data.geom_xmat), dtype=torch.float32, device=self.device
        ).view(-1, 3, 3)
        super().update(xmat, xpos)


_SCENES: OrderedDict[tuple, SceneMesh] = OrderedDict()


def scene_mesh(model, groups: set[int], device) -> SceneMesh:
    key = (id(model), int(model.ngeom), tuple(sorted(groups)), str(torch.device(device)))
    hit = _SCENES.get(key)
    if hit is not None:
        _SCENES.move_to_end(key)
        return hit
    while len(_SCENES) >= 2:
        _SCENES.popitem(last=False)
    _SCENES[key] = SceneMesh(model, groups, torch.device(device))
    return _SCENES[key]


class MujocoLensScene:
    """A MuJoCo camera as a lens scene (`twinrobo.optics.lensrender`)."""

    def __init__(self, backend, camera: str, cam_id: int, groups: set[int] | None = None):
        self.backend, self.camera, self.cam_id = backend, camera, cam_id
        self.groups = groups

    def render_views(self, views: PupilViews):
        dev = views.rays.device
        model, data = self.backend.model, self.backend.data
        rgbs, depths = [], []
        for c in views.centers.tolist():
            with (
                _camera_fovy(model, self.cam_id, views.fovy_deg),
                _camera_offset(model, data, self.cam_id, (c[0], c[1], 0.0)),
            ):
                rgb8, depth = self.backend.render(self.camera, views.width, views.height)
            rgbs.append(torch.from_numpy(rgb8).to(dev).permute(2, 0, 1).float() / 255.0)
            depths.append(torch.from_numpy(depth).to(dev)[None])
        return srgb_to_linear(torch.stack(rgbs)), torch.stack(depths)

    def camera_pose(self):
        data = self.backend.data
        return np.asarray(data.cam_xmat[self.cam_id]).reshape(3, 3), np.asarray(
            data.cam_xpos[self.cam_id]
        )

    def mesh(self, device) -> SceneMesh:
        groups = self.groups if self.groups is not None else _drawn_groups(self.backend)
        m = scene_mesh(self.backend.model, groups, device)
        m.update(self.backend.data)
        return m


class LensRayRenderer(_LensRayRenderer):
    """`twinrobo.optics.lensrender.LensRayRenderer` with a MuJoCo `render` shortcut."""

    def render(self, backend, camera: str, cam_id: int, groups: set[int] | None = None):
        """``(rgb [1, 3, H, W] linear, depth [1, 1, H, W] m)`` of the current MuJoCo state."""
        return self.render_scene(MujocoLensScene(backend, camera, cam_id, groups))


def _drawn_groups(backend) -> set[int]:
    """Geom groups the backend's rasterizer draws (so ray casting hits the same geometry)."""
    vopt = getattr(backend, "vopt", None)
    if vopt is None:
        return {0, 1, 2}  # mujoco.Renderer / MjvOption default
    return {i for i, on in enumerate(vopt.geomgroup) if on}
