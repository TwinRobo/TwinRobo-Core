"""Lens-ray rendering in Isaac Sim: a USD camera as a lens scene.

The ray-cast method (``"raycast"``) is simulator independent
(`twinrobo.optics.lensrender`); this module supplies Isaac's side:

- **shading views across the pupil:** child cameras of the camera prim (``TwinRoboPupil<i>``), each
  shifted in the camera's own frame to a pupil cell center, with the views' wide
  FoV, each with a Replicator render product (``rgb`` + ``distance_to_image_plane``).
  They render in the same Replicator step as the main view.
- **camera pose:** the camera prim's world transform (stage units -> meters).
- **scene triangles:** every visible gprim of the stage (meshes, cubes, spheres,
  cylinders, capsules, cones, planes, including instance proxies), posed per frame
  from USD transforms into one Warp BVH.

Limits: poses are read from USD, so a simulation must write them there (Isaac's
default for scripted and physics prims); deforming meshes keep their first
shape; point instancers are not ray cast.
"""

from __future__ import annotations

import numpy as np
import torch

from ...isp.color import srgb_to_linear
from ...optics.lensrender import PupilViews, TriangleMesh, _box, _cylinder, _sphere

PUPIL_PRIM = "TwinRoboPupil"
APERTURE = 20.955  # USD default horizontal aperture; only focal / aperture matters


def set_pinhole(cam, fx_px: float, width: int, height: int) -> None:
    """Give a ``UsdGeom.Camera`` a centered pinhole: ``fx_px`` pixels at ``width x height``."""
    cam.GetHorizontalApertureAttr().Set(APERTURE)
    cam.GetVerticalApertureAttr().Set(APERTURE * height / width)
    cam.GetFocalLengthAttr().Set(fx_px * APERTURE / width)


# -- USD triangles ---------------------------------------------------------------------------------


def _axis_matrix(axis: str) -> np.ndarray:
    """Rotate a shape built along +Z onto USD's ``axis`` token."""
    if axis == "X":
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32).T
    if axis == "Y":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32).T
    return np.eye(3, dtype=np.float32)


def _cone(n=32):
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(ph), np.sin(ph), -np.ones(n)]
    v = np.concatenate([ring, [[0, 0, 1], [0, 0, -1]]]).astype(np.float32)
    f = []
    for j in range(n):
        a, b = j, (j + 1) % n
        f += [[a, b, n], [n + 1, b, a]]
    return v, np.array(f, np.int32)


def prim_triangles(prim) -> tuple[np.ndarray, np.ndarray] | None:
    """Local-frame triangles of a USD gprim (as RTX draws it), or ``None`` if unsupported."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts, counts, idx = (
            m.GetPointsAttr().Get(),
            m.GetFaceVertexCountsAttr().Get(),
            m.GetFaceVertexIndicesAttr().Get(),
        )
        if not pts or not counts or not idx:
            return None
        pts = np.asarray(pts, np.float32)
        counts, idx = np.asarray(counts, np.int64), np.asarray(idx, np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tris = []  # fan-triangulate every polygon: (v0, vi, vi+1)
        for n in np.unique(counts[counts >= 3]):
            s = starts[counts == n]
            for i in range(1, int(n) - 1):
                tris.append(np.stack([idx[s], idx[s + i], idx[s + i + 1]], 1))
        if not tris:
            return None
        return pts, np.concatenate(tris).astype(np.int32)
    if prim.IsA(UsdGeom.Cube):
        v, f = _box()
        return v * (UsdGeom.Cube(prim).GetSizeAttr().Get() / 2), f
    if prim.IsA(UsdGeom.Sphere):
        v, f = _sphere()
        return v * UsdGeom.Sphere(prim).GetRadiusAttr().Get(), f
    if prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        r, h = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get()
        v, f = _cylinder()
        return (v * np.array([r, r, h / 2], np.float32)) @ _axis_matrix(c.GetAxisAttr().Get()), f
    if prim.IsA(UsdGeom.Capsule):
        c = UsdGeom.Capsule(prim)
        r, h = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get()
        cv, cf = _cylinder()
        sv, sf = _sphere()
        v = np.concatenate(
            [
                cv * np.array([r, r, h / 2], np.float32),
                sv * r + np.array([0, 0, h / 2], np.float32),
                sv * r - np.array([0, 0, h / 2], np.float32),
            ]
        )
        f = np.concatenate([cf, sf + len(cv), sf + len(cv) + len(sv)])
        return v @ _axis_matrix(c.GetAxisAttr().Get()), f
    if prim.IsA(UsdGeom.Cone):
        c = UsdGeom.Cone(prim)
        r, h = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get()
        v, f = _cone()
        return (v * np.array([r, r, h / 2], np.float32)) @ _axis_matrix(c.GetAxisAttr().Get()), f
    if hasattr(UsdGeom, "Plane") and prim.IsA(UsdGeom.Plane):
        p = UsdGeom.Plane(prim)
        hw, hl = p.GetWidthAttr().Get() / 2, p.GetLengthAttr().Get() / 2
        v = np.array([[-hw, -hl, 0], [hw, -hl, 0], [hw, hl, 0], [-hw, hl, 0]], np.float32)
        return v @ _axis_matrix(p.GetAxisAttr().Get()), np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return None


class UsdSceneMesh(TriangleMesh):
    """Every visible gprim of a USD stage as ray-cast triangles, posed per frame from USD."""

    def __init__(self, stage, device):
        from pxr import Usd, UsdGeom

        verts, faces, owner, prims, off = [], [], [], [], 0
        hidden = {UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy}
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            img = UsdGeom.Imageable(prim)
            if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if img.ComputePurposeInfo().purpose in hidden:
                continue
            tri = prim_triangles(prim)
            if tri is None:
                continue
            v, f = tri
            verts.append(v)
            faces.append(f + off)
            owner.append(np.full(len(v), len(prims), np.int64))
            prims.append(prim)
            off += len(v)
        if not prims:
            raise ValueError("no ray-castable geometry on the stage")
        super().__init__(
            np.concatenate(verts), np.concatenate(faces), np.concatenate(owner), device
        )
        self.stage, self.prims = stage, prims
        self.meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

    def update(self) -> None:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        M = np.array([cache.GetLocalToWorldTransform(p) for p in self.prims], np.float64)
        M = torch.as_tensor(M, dtype=torch.float32, device=self.device) * self.meters_per_unit
        # USD matrices act on row vectors: world = local @ M[:3, :3] + M[3, :3]
        super().update(M[:, :3, :3].transpose(1, 2), M[:, 3, :3])


# -- the lens scene --------------------------------------------------------------------------------


class IsaacLensScene:
    """A USD camera as a lens scene (`twinrobo.optics.lensrender.LensRayRenderer.render_scene`).

    Creates the pupil-view cameras and their render products under ``camera``;
    `close` removes them. The caller renders (one Replicator step renders the main
    view and every pupil view) before `LensRayRenderer.render_scene`.
    """

    def __init__(self, camera: str, views: PupilViews, near_clip_m: float = 0.01):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self.stage = omni.usd.get_context().get_stage()
        self.camera = camera
        self.meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        src = UsdGeom.Camera(self.stage.GetPrimAtPath(camera))
        far = src.GetClippingRangeAttr().Get()[1]
        scale = self._world_scale()
        self.paths, self.products, self.annotators = [], [], []
        for i, (cx, cy) in enumerate(views.centers.tolist()):
            cam = UsdGeom.Camera.Define(self.stage, f"{camera}/{PUPIL_PRIM}{i}")
            cam.GetClippingRangeAttr().Set((near_clip_m / self.meters_per_unit, far))
            set_pinhole(cam, views.f, views.width, views.height)
            offset = Gf.Vec3d(cx, cy, 0.0) / (self.meters_per_unit * scale)  # camera frame
            UsdGeom.Xformable(cam).AddTranslateOp().Set(offset)
            path = cam.GetPath().pathString
            rp = rep.create.render_product(path, (views.width, views.height))
            ann = {
                n: rep.AnnotatorRegistry.get_annotator(n, device="cuda")
                for n in ("rgb", "distance_to_image_plane")
            }
            for a in ann.values():
                a.attach([rp])
            self.paths.append(path)
            self.products.append(rp)
            self.annotators.append(ann)
        self._mesh: UsdSceneMesh | None = None

    def _world_matrix(self) -> np.ndarray:
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(self.camera)
        return np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )

    def _world_scale(self) -> float:
        return float(np.linalg.norm(self._world_matrix()[0, :3]))

    def ready(self) -> bool:
        """Whether every pupil view has rendered a frame."""
        for ann in self.annotators:
            d = ann["distance_to_image_plane"].get_data(device="cuda")
            if d is None or len(d.shape) != 2:
                return False
        return True

    def render_views(self, views: PupilViews):
        import warp as wp

        dev = views.rays.device
        rgbs, depths = [], []
        for ann in self.annotators:
            rgba = wp.to_torch(ann["rgb"].get_data(device="cuda"))
            dist = wp.to_torch(ann["distance_to_image_plane"].get_data(device="cuda"))
            rgbs.append(rgba[..., :3].to(dev).permute(2, 0, 1).float() / 255.0)
            depths.append(dist.to(dev).reshape(1, views.height, views.width).float())
        depth = torch.stack(depths) * self.meters_per_unit
        return srgb_to_linear(torch.stack(rgbs)), depth

    def camera_pose(self):
        M = self._world_matrix()  # row vectors: rows 0-2 are the camera's axes, row 3 its origin
        R = M[:3, :3].T
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        return R, M[3, :3] * self.meters_per_unit

    def mesh(self, device) -> UsdSceneMesh:
        if self._mesh is None:
            self._mesh = UsdSceneMesh(self.stage, device)
        self._mesh.update()
        return self._mesh

    def close(self) -> None:
        for ann, rp in zip(self.annotators, self.products, strict=True):
            for a in ann.values():
                a.detach()
            rp.destroy()
        for p in self.paths:
            if self.stage.GetPrimAtPath(p).IsValid():
                self.stage.RemovePrim(p)
        self.paths, self.products, self.annotators = [], [], []
