"""Lens-ray rendering: sensor -> real lens -> scene, for any simulator.

Two methods share the traced `LensRays` (every sensor pixel's rays through the
lens, per RGB wavelength) and the simulator's own shading (no PBR):

``"pupil"`` (method A, pupil-sampled rasterization)
    The entrance pupil is split into ``views`` cells. For each cell the simulator
    rasterizes one wide pinhole view (RGB + depth) from the cell's center. Every
    ray is looked up in the view of its cell; the lookup is refined with that
    view's depth so the ray's own origin, not the cell center, sets the parallax.
    Exact except where a ray's scene point is hidden from its cell center
    (disocclusion at depth edges), where it takes the occluder's color.

``"raycast"`` (method B, per-ray ray casting)
    Every ray is intersected with the scene's triangles on the GPU (NVIDIA Warp
    BVH) for its exact hit point. The hit's color comes from the pupil views, from
    the nearest view in which the hit point is visible (a depth test). This gives
    exact per-ray visibility with the simulator's shading, which is view-dependent
    only through specular highlights.

A simulator plugs in through a *lens scene* (`LensRayRenderer.render_scene`):

- ``render_views(views)`` -> ``(rgb [k, 3, h, w] linear, depth [k, 1, h, w] m)``: the
  pupil views, pinholes of ``views.fovy_deg`` at ``views.width x views.height``, from
  the camera shifted by each ``views.centers`` offset (m, camera frame);
- ``camera_pose()`` -> ``(R [3, 3], p [3])``: camera axes as columns (x right, y up,
  looking along -z) and position, world frame, meters;
- ``mesh(device)`` -> a `TriangleMesh` of the scene at the current state (ray cast only).

Adapters: `twinrobo.plugins.mujoco.lensrender` (MuJoCo) and `twinrobo.plugins.isaac.lens`
(Isaac Sim).
Rays depend on lens, focus and sensor only, so they are traced once (`lens_rays`)
and cached on the GPU.
"""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .lensrays import LensRays

METHODS = ("psf", "pupil", "raycast")
_RAYS: OrderedDict[str, LensRays] = OrderedDict()
_RAYS_MAX = 4


def lens_rays(
    reference, rays_per_pixel: int = 8, status=None, intrinsics=None, distortion=None
) -> LensRays:
    """Traced rays of a `DeepLensOptics` (cached; the first trace takes seconds).

    With ``intrinsics`` (and optionally ``distortion``), the rays take that
    calibrated geometry and keep the lens' blur (`LensRays.with_geometry`); see
    `CameraTwin.lens_geometry`. Without, the traced lens' own distortion is kept.
    """
    key = reference.lens_rays_key(rays_per_pixel)
    if intrinsics is not None:
        key["geometry"] = {
            "intrinsics": [intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy,
                           intrinsics.cx, intrinsics.cy],
            "distortion": distortion.to_dict() if distortion is not None else None,
        }  # fmt: skip
    key = json.dumps(key, sort_keys=True)
    hit = _RAYS.get(key)
    if hit is not None:
        _RAYS.move_to_end(key)
        return hit
    while len(_RAYS) >= _RAYS_MAX:
        _RAYS.popitem(last=False)
    if status is not None:
        status("tracing the lens: every sensor pixel's rays (first use of a config, ~5-20 s)")
    rays = reference.trace_lens_rays(rays_per_pixel)
    if intrinsics is not None:
        rays = rays.with_geometry(intrinsics, distortion)
    if torch.cuda.is_available():  # the trace's scratch (GBs at full res) is not needed again;
        torch.cuda.empty_cache()  # hand it back so the renderer (e.g. Isaac's RTX) can use it
    _RAYS[key] = rays
    return rays


# -- pupil views ----------------------------------------------------------------------------------


def pupil_centers(rays: LensRays, k: int, iters: int = 12) -> Tensor:
    """``k`` cell centers ``[k, 2]`` (m) on the entrance pupil: k-means of the ray origins."""
    o = rays.origin[1][rays.weight[1] > 0].float()  # green, unblocked
    if o.shape[0] > 200_000:
        o = o[
            torch.randperm(
                o.shape[0], device=o.device, generator=torch.Generator(o.device).manual_seed(0)
            )[:200_000]
        ]
    r = rays.pupil_radius_m
    # deterministic hexagonal start: center, then rings
    init = [(0.0, 0.0)]
    ring, rad = 1, 0.0
    while len(init) < k:
        rad = min(0.95, ring / (ring + 0.6))
        m = 6 * ring
        init += [
            (rad * math.cos(2 * math.pi * i / m), rad * math.sin(2 * math.pi * i / m))
            for i in range(m)
        ]
        ring += 1
    c = torch.tensor(init[:k], device=o.device, dtype=torch.float32) * r
    for _ in range(iters):
        idx = torch.cdist(o, c).argmin(1)
        for j in range(k):
            sel = o[idx == j]
            if sel.shape[0]:
                c[j] = sel.mean(0)
    return c


class PupilViews:
    """``k`` rasterized views (linear RGB + z-depth) from the pupil cell centers."""

    def __init__(self, rays: LensRays, k: int, margin: float = 0.06, oversample: float = 1.0):
        self.rays = rays
        self.centers = pupil_centers(rays, k)  # [k, 2] m
        tx, ty = rays.max_tan()
        K = rays.intrinsics
        self.f = K.fx * oversample  # source focal length (px)
        self.width = int(math.ceil(2 * (tx + margin) * self.f)) // 2 * 2 + 2
        self.height = int(math.ceil(2 * (ty + margin) * self.f)) // 2 * 2 + 2
        self.fovy_deg = math.degrees(2 * math.atan(self.height / 2 / self.f))
        # nearest cell of every ray: uint8 [C, H, W, n] (in row chunks: [.., k, 2] is large)
        C, H, W, n, _ = rays.origin.shape
        self.cell = torch.empty(C, H, W, n, dtype=torch.uint8, device=rays.device)
        rows = max(1, 2_000_000 // (W * n))
        for c in range(C):
            for r0 in range(0, H, rows):
                o = rays.origin[c, r0 : r0 + rows].float().reshape(-1, 2)
                self.cell[c, r0 : r0 + rows] = (
                    torch.cdist(o, self.centers).argmin(1).to(torch.uint8).view(-1, W, n)
                )
        self.rgb: Tensor | None = None  # [k, 3, h, w] linear
        self.depth: Tensor | None = None  # [k, 1, h, w] z-depth (m)

    def set_views(self, rgb: Tensor, depth: Tensor) -> None:
        """Install rendered views: ``rgb [k, 3, h, w]`` linear, ``depth [k, 1, h, w]`` m."""
        self.rgb, self.depth = rgb, depth
        # Silhouette pixels: rasterizers antialias color, so a pixel on a depth edge holds a
        # blend of foreground and background while its depth is one of them. Flag them.
        z = self.depth
        zmax = F.max_pool2d(z, 3, 1, 1)
        zmin = -F.max_pool2d(-z, 3, 1, 1)
        self.edge = (zmax - zmin) > (0.02 * z + 0.003)
        # flattened [channel, k * h * w] buffers: rays look up their own view in one gather
        self._rgb_flat = self.rgb.permute(1, 0, 2, 3).reshape(3, -1)
        self._z_flat = self.depth.reshape(-1)
        self._edge_flat = self.edge.reshape(-1)

    def _taps(self, k: Tensor, tan: Tensor):
        """Bilinear taps of per-ray views ``k [M]`` at slopes ``tan [M, 2]``: (index, weight) x4."""
        H, W = self.height, self.width
        px = tan[:, 0] * self.f + (W - 1) / 2
        py = -tan[:, 1] * self.f + (H - 1) / 2
        x0, y0 = px.floor(), py.floor()
        fx, fy = px - x0, py - y0
        base = k.long() * (H * W)
        taps = []
        for dx, dy, w in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            xi = (x0 + dx).clamp(0, W - 1).long()
            yi = (y0 + dy).clamp(0, H - 1).long()
            taps.append((base + yi * W + xi, w))
        nearest = base + py.round().clamp(0, H - 1).long() * W + px.round().clamp(0, W - 1).long()
        return taps, nearest

    def depth_at(self, k: Tensor, tan: Tensor) -> Tensor:
        """Nearest-tap z-depth of per-ray views ``k`` along ``tan``."""
        _, nearest = self._taps(k, tan)
        return self._z_flat[nearest]

    def shade(self, k: Tensor, tan: Tensor, channel: int, zref: Tensor):
        """Color of per-ray views ``k [M]`` along ``tan``, using only taps at depth ``zref``.

        A lookup beside a silhouette would otherwise blend the occluder into the
        surface behind it (and rasterizers antialias color, so silhouette pixels are
        themselves blends). Preference: matching depth off the silhouette ("clean"),
        then matching depth on it, then the nearest tap. Returns ``(color, clean,
        seen)``: whether a clean tap was used, and whether any tap matches ``zref``.
        """
        taps, nearest = self._taps(k, tan)
        img = self._rgb_flat[channel]
        tol = 0.01 * zref + 0.003
        acc_c = torch.zeros_like(zref)
        w_c = torch.zeros_like(zref)
        acc_m = torch.zeros_like(zref)
        w_m = torch.zeros_like(zref)
        seen = torch.zeros_like(zref, dtype=torch.bool)
        for idx, w in taps:
            col = img[idx]
            match = (self._z_flat[idx] - zref).abs() <= tol
            seen |= match
            wm = w * match
            acc_m += col * wm
            w_m += wm
            wc = wm * ~self._edge_flat[idx]
            acc_c += col * wc
            w_c += wc
        clean = w_c > 1e-6
        out = torch.where(
            clean,
            acc_c / w_c.clamp_min(1e-6),
            torch.where(w_m > 1e-6, acc_m / w_m.clamp_min(1e-6), img[nearest]),
        )
        return out, clean, seen

    def color_at(self, k: Tensor, tan: Tensor, channel: int) -> Tensor:
        """Plain bilinear color of per-ray views ``k`` along ``tan`` (no depth test)."""
        taps, _ = self._taps(k, tan)
        img = self._rgb_flat[channel]
        return sum(img[idx] * w for idx, w in taps)


# -- scene triangles for ray casting (method B) ---------------------------------------------


def _box():
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
    f = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        np.int32,
    )
    return v, f


def _sphere(nu=24, nv=12):
    th = np.linspace(0, np.pi, nv + 1)
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.array(
        [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph],
        np.float32,
    )
    f = []
    for i in range(nv):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            c, d = a + nu, b + nu
            f += [[a, c, d], [a, d, b]]
    return v, np.array(f, np.int32)


def _cylinder(n=32):
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(ph), np.sin(ph)], 1)
    v = np.concatenate(
        [np.c_[ring, -np.ones(n)], np.c_[ring, np.ones(n)], [[0, 0, -1], [0, 0, 1]]]
    ).astype(np.float32)
    f = []
    for j in range(n):
        a, b = j, (j + 1) % n
        f += [[a, b, b + n], [a, b + n, a + n], [2 * n, b, a], [2 * n + 1, a + n, b + n]]
    return v, np.array(f, np.int32)


class TriangleMesh:
    """Scene triangles as one GPU BVH (NVIDIA Warp), posed per frame and refit.

    ``local [V, 3]`` vertices in their owners' frames, ``faces [F, 3]``, ``owner [V]``:
    the index of each vertex's rigid frame. `update` poses all of them.
    """

    def __init__(self, local: np.ndarray, faces: np.ndarray, owner: np.ndarray, device):
        import warp as wp

        wp.init()
        self.device = torch.device(device)
        self.local = torch.from_numpy(np.ascontiguousarray(local, np.float32)).to(self.device)
        self.owner = torch.from_numpy(np.asarray(owner, np.int64)).to(self.device)
        self.points = torch.zeros_like(self.local)
        self.num_triangles = int(len(faces))
        self._wdev = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        self._wp_points = wp.from_torch(self.points, dtype=wp.vec3)
        self._idx = torch.from_numpy(np.asarray(faces, np.int32).reshape(-1)).to(self.device)
        self.mesh = None  # built on the first `update`, from real poses (a BVH over all-zero
        # points is degenerate, and refitting never restructures it)

    def update(self, R: Tensor, t: Tensor) -> None:
        """Pose every owner: world = ``R[owner] @ local + t[owner]``.

        ``R [P, 3, 3]`` rotations (with scale, if any) and ``t [P, 3]`` translations.
        """
        R, t = R[self.owner], t[self.owner]
        self.points.copy_(torch.einsum("vij,vj->vi", R, self.local) + t)
        if self.mesh is None:
            import warp as wp

            self.mesh = wp.Mesh(
                points=self._wp_points, indices=wp.from_torch(self._idx, dtype=wp.int32)
            )
        else:
            self.mesh.refit()

    def cast(self, origins: Tensor, dirs: Tensor, max_t: float = 100.0) -> Tensor:
        """Hit distance along unit ``dirs`` (``1e30`` for a miss), ``[M]``."""
        import warp as wp

        n = origins.shape[0]
        out = torch.empty(n, device=self.device, dtype=torch.float32)
        wp.launch(
            _raycast_kernel(),
            dim=n,
            inputs=[
                self.mesh.id,
                wp.from_torch(origins.contiguous(), dtype=wp.vec3),
                wp.from_torch(dirs.contiguous(), dtype=wp.vec3),
                float(max_t),
            ],
            outputs=[wp.from_torch(out)],
            device=self._wdev,
        )
        return out


_KERNEL = None


def _raycast_kernel():
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def raycast(
            mesh: wp.uint64,
            o: wp.array(dtype=wp.vec3),
            d: wp.array(dtype=wp.vec3),
            max_t: float,
            out: wp.array(dtype=float),
        ):
            i = wp.tid()
            q = wp.mesh_query_ray(mesh, o[i], d[i], max_t)
            if q.result:
                out[i] = q.t
            else:
                out[i] = 1.0e30

        _KERNEL = raycast
    return _KERNEL


# -- the renderer ----------------------------------------------------------------------------------


class LensRayRenderer:
    """Render a camera through a real lens with method ``"pupil"`` or ``"raycast"``.

    Args:
        rays: traced `LensRays` of the camera's lens and sensor.
        method: ``"pupil"`` (A) or ``"raycast"`` (B).
        views: number of pupil cells / rasterized views (1 = depth reprojection only).
        oversample: resolution of the views relative to the sensor. Above 1 resolves
            detail thinner than a sensor pixel (the views are the shading source).
        shading: ``"corrected"`` divides out relative illumination, as a camera ISP's
            lens-shading correction does; ``"raw"`` keeps sensor irradiance falloff.
        refine: depth-refinement iterations of the pupil lookup.
        chunk: rays per GPU batch.
    """

    def __init__(
        self,
        rays: LensRays,
        method: str = "pupil",
        views: int = 7,
        shading: str = "corrected",
        refine: int = 2,
        chunk: int = 4_000_000,
        oversample: float = 1.0,
    ):
        if method not in ("pupil", "raycast"):
            raise ValueError(f"method must be 'pupil' or 'raycast', got {method!r}")
        if shading not in ("corrected", "raw"):
            raise ValueError("shading must be 'corrected' or 'raw'")
        self.rays, self.method, self.shading = rays, method, shading
        self.refine, self.chunk = int(refine), int(chunk)
        self.views = PupilViews(rays, int(views), oversample=float(oversample))
        self.stats: dict[str, Any] = {}

    @torch.no_grad()
    def render_scene(self, scene) -> tuple[Tensor, Tensor]:
        """``(rgb [1, 3, H, W] linear, depth [1, 1, H, W] m)`` of a lens scene's current state."""
        R = self.rays
        dev = R.device
        t0 = time.perf_counter()
        self.views.set_views(*scene.render_views(self.views))
        t_views = time.perf_counter() - t0
        mesh = Rc = pc = None
        if self.method == "raycast":
            mesh = scene.mesh(dev)
            Rc, pc = scene.camera_pose()
            Rc = torch.as_tensor(Rc, dtype=torch.float32, device=dev)
            pc = torch.as_tensor(pc, dtype=torch.float32, device=dev)
        W, H = R.resolution
        out = torch.zeros(R.channels, H, W, device=dev)
        unresolved = total = 0
        rows = max(1, self.chunk // (W * R.n))
        for c in range(R.channels):
            for r0 in range(0, H, rows):
                sl = slice(r0, min(H, r0 + rows))
                o, t, w = R.rays(c, sl)
                cell = self.views.cell[c, sl].long()
                h = o.shape[0]
                o, t, w, cell = o.reshape(-1, 2), t.reshape(-1, 2), w.reshape(-1), cell.reshape(-1)
                if self.method == "pupil":
                    col = self._pupil(o, t, cell, c)
                else:
                    col, miss = self._raycast(o, t, cell, c, mesh, Rc, pc)
                    unresolved += int(miss)
                    total += o.shape[0]
                col = col.view(h, W, R.n)
                wv = w.view(h, W, R.n)
                ws = wv.sum(-1)
                out[c, sl] = torch.where(
                    ws > 0, (col * wv).sum(-1) / ws.clamp_min(1e-12), torch.zeros_like(ws)
                )
        if self.shading == "raw":
            out = out * R.illumination
        depth = self._chief_depth()
        self.stats = {
            "method": self.method,
            "views": len(self.views.centers),
            "rays_per_pixel": R.n,
            "view_size": [self.views.width, self.views.height],
            "t_views_s": t_views,
            "t_total_s": time.perf_counter() - t0,
            **(
                {"unresolved_fraction": unresolved / max(total, 1), "triangles": mesh.num_triangles}
                if mesh
                else {}
            ),
        }
        return out[None], depth[None, None]

    def _pupil(self, o, t, cell, c):
        """Method A: look each ray up in its cell's view, refined with that view's depth."""
        V = self.views
        off = o - V.centers[cell]  # origin offset from the cell's view center (m)
        look = t
        for _ in range(self.refine):
            z = V.depth_at(cell, look)
            look = t + off / z.clamp_min(1e-3)[:, None]
        col, _, _ = V.shade(cell, look, c, V.depth_at(cell, look))
        return col

    def _raycast(self, o, t, cell, c, mesh, Rc, pc):
        """Method B: exact hit per ray, shaded from the nearest pupil view that sees the hit."""
        V = self.views
        dcam = torch.cat([t, -torch.ones_like(t[:, :1])], 1)
        norm = dcam.norm(dim=1, keepdim=True)
        dworld = (dcam / norm) @ Rc.T
        oworld = torch.cat([o, torch.zeros_like(o[:, :1])], 1) @ Rc.T + pc
        s = (
            mesh.cast(oworld, dworld) / norm[:, 0]
        )  # parameter along (tx, ty, -1): the hit's z-depth
        hit = s < 1e20
        col = torch.zeros(o.shape[0], device=o.device)
        done = ~hit
        if bool((~hit).any()):  # misses (nothing in range): background along the ray
            idx = (~hit).nonzero(as_tuple=True)[0]
            col[idx] = V.color_at(cell[idx], t[idx], c)
        p = o + t * s[:, None]  # hit xy in the camera frame (z = -s)
        order = torch.cdist(o, V.centers).argsort(1)  # views by distance from the ray's origin
        # Pass 1 accepts only views that see the hit on a clean (non-silhouette) pixel: across
        # the aperture the silhouette shifts, so another view often has one. Pass 2 accepts
        # any view that sees it.
        for need_clean in (True, False):
            for rank in range(len(V.centers)):
                todo = (~done).nonzero(as_tuple=True)[0]
                if todo.numel() == 0:
                    break
                ks = order[todo, rank]
                tk = (p[todo] - V.centers[ks]) / s[todo, None]
                cv, clean, vis = V.shade(ks, tk, c, s[todo])
                ok = vis & clean if need_clean else vis
                good = todo[ok]
                col[good] = cv[ok]
                done[good] = True
        miss = int((~done).sum())
        if miss:  # hidden from every view: nearest view's color at the hit direction
            idx = (~done).nonzero(as_tuple=True)[0]
            k0 = order[idx, 0]
            col[idx] = V.color_at(k0, (p[idx] - V.centers[k0]) / s[idx, None], c)
        return col, miss

    def _chief_depth(self) -> Tensor:
        """z-depth along each pixel's chief ray, from the view nearest the pupil center."""
        V, R = self.views, self.rays
        k = int(V.centers.norm(dim=1).argmin())
        tan = R.chief_tan.reshape(-1, 2)
        z = V.depth_at(torch.full((tan.shape[0],), k, device=tan.device), tan)
        W, H = R.resolution
        return z.view(H, W)


def lens_frame(
    twin,
    renderer: LensRayRenderer,
    rgb: Tensor,
    depth: Tensor,
    ideal: Tensor,
    rectify: bool,
    timestamp=None,
    metadata: dict | None = None,
    force_depth: bool = False,
):
    """A `CameraFrame` from a lens render: sensor and ISP, then optional rectification.

    ``rgb``/``depth`` come from `LensRayRenderer.render_scene`; ``ideal`` is the paraxial
    pinhole view of the same sensor (``renderer.rays.intrinsics``), linear.
    ``rectify`` undistorts with the lens' own chief rays, as stereo SDKs deliver.
    ``force_depth``: expose depth even if the camera outputs none (see `CameraTwin.process`).
    """
    from ..frame import CameraFrame

    rgb, ideal = twin.sensor_color(rgb), twin.sensor_color(ideal)  # mono sensors: luminance
    raw = twin.sensor.capture(rgb, 1.0, None)
    out = twin.isp.process(raw, None)
    if rectify:
        grid = renderer.rays.rectify_grid()
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        depth = F.grid_sample(
            depth, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
    return CameraFrame(
        rgb=out,
        rgb_ideal=ideal,
        rgb_optical=rgb,
        raw=raw,
        depth=depth,
        timestamp=timestamp,
        metadata={
            "camera_id": twin.spec.id if twin.spec else None,
            "render": renderer.method,
            "rectified": bool(rectify),
            **(metadata or {}),
            **renderer.stats,
        },
        has_depth=twin.outputs_depth or force_depth,
    )
