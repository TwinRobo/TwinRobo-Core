"""Custom camera mounts: a camera attached to any body (robot link, gripper, ...) or the world.

A compiled MuJoCo model cannot gain cameras, and rebuilding a LIBERO scene per
edit takes seconds. So a mount is rendered by temporarily re-pointing an
existing *host* camera: its parent body, pose and FoV are replaced, the camera
is rendered, and everything is restored (`mounted`). Link-mounted cameras
therefore follow the robot through a replay.

Orientation: yaw, pitch and roll in degrees, in the parent body's frame. At
zero the camera looks along the body's +x with +z up. Yaw turns about the
body's z, pitch tilts down (positive = look down), and roll turns about the
viewing axis.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np

WORLD = "world"
#: Camera present in every LIBERO scene, borrowed while a mount renders (always restored).
HOST_CAMERA = "frontview"


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# Camera axes in the body frame at zero yaw/pitch/roll (columns x_cam, y_cam, z_cam):
# MuJoCo cameras look along -z_cam with +y_cam up, so -z_cam = +x_body and y_cam = +z_body.
_R0 = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` of rotation matrix ``R`` (Shepperd's method), ``w >= 0``."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


@dataclass
class CameraMount:
    name: str
    body: str = WORLD
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
    fovy: float = 60.0
    module: str | None = None  # stereo module id: this mount is its left eye (see expand_mount)
    baseline_mm: float | None = None  # a stereo module's baseline, if not the catalog one

    def __post_init__(self):
        self.pos = tuple(float(v) for v in self.pos)
        self.rpy_deg = tuple(float(v) for v in self.rpy_deg)
        self.fovy = float(self.fovy)
        if len(self.pos) != 3 or len(self.rpy_deg) != 3:
            raise ValueError("pos and rpy_deg need three values")
        if not 1.0 <= self.fovy <= 170.0:
            raise ValueError("fovy must be within [1, 170] degrees")
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError("camera name: letters, digits, _ and - only")
        if self.baseline_mm is not None:
            if not self.module:
                raise ValueError("baseline_mm is for stereo module mounts (set module)")
            self.baseline_mm = float(self.baseline_mm)
            if not 1.0 <= self.baseline_mm <= 2000.0:
                raise ValueError("baseline_mm must be within [1, 2000] mm")

    def rotation(self) -> np.ndarray:
        """Body-frame rotation whose columns are the camera's x, y, z axes."""
        yaw, pitch, roll = (math.radians(v) for v in self.rpy_deg)
        return _rz(yaw) @ _ry(pitch) @ _rx(roll) @ _R0

    def quat(self) -> np.ndarray:
        """Body-frame orientation as a MuJoCo quaternion (w, x, y, z), ``w >= 0``.

        Plain numpy (no MuJoCo needed): rigs are saved and shown without a simulator.
        """
        return mat_to_quat(self.rotation())

    def to_json(self) -> dict:
        d = asdict(self)
        d["pos"], d["rpy_deg"] = list(self.pos), list(self.rpy_deg)
        d["quat"] = [float(v) for v in self.quat()]  # for the 3D view
        if d["baseline_mm"] is None:  # the module's own baseline: nothing to store
            del d["baseline_mm"]
        return d

    @classmethod
    def from_json(cls, d: dict) -> CameraMount:
        return cls(
            name=d["name"],
            body=d.get("body", WORLD),
            pos=d.get("pos", (0, 0, 0)),
            rpy_deg=d.get("rpy_deg", (0, 0, 0)),
            fovy=d.get("fovy", 60.0),
            module=d.get("module") or None,
            baseline_mm=d.get("baseline_mm"),
        )


def body_id(model, body: str) -> int:
    import mujoco

    if body == WORLD:
        return 0
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0:
        raise ValueError(f"no body {body!r} in this scene")
    return b


def mountable_bodies(model) -> list[str]:
    """Bodies a camera can be attached to: the world, then robot, gripper and mount links."""
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]
    robot = [n for n in names if n and n.startswith(("robot0_", "gripper0_", "mount0_"))]
    return [WORLD] + robot


@contextmanager
def mounted(model, data, mount: CameraMount, host: str = HOST_CAMERA):
    """Temporarily turn camera ``host`` into ``mount``; yields the host camera's name."""
    import mujoco

    h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, host)
    if h < 0:
        raise ValueError(f"host camera {host!r} missing from the scene")
    fields = ("cam_bodyid", "cam_targetbodyid", "cam_mode", "cam_pos", "cam_quat", "cam_fovy")
    saved = {f: np.array(getattr(model, f)[h], copy=True) for f in fields}
    try:
        model.cam_bodyid[h] = body_id(model, mount.body)
        model.cam_targetbodyid[h] = -1
        model.cam_mode[h] = 0  # fixed to its body
        model.cam_pos[h] = mount.pos
        model.cam_quat[h] = mount.quat()
        model.cam_fovy[h] = mount.fovy
        mujoco.mj_camlight(model, data)
        yield host
    finally:
        for f, v in saved.items():
            getattr(model, f)[h] = v
        mujoco.mj_camlight(model, data)


def rpy_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `CameraMount.rotation`: body-frame camera axes -> (yaw, pitch, roll) degrees."""
    M = np.asarray(R, float) @ _R0.T  # = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    yaw = math.atan2(M[1, 0], M[0, 0])
    pitch = math.asin(float(np.clip(-M[2, 0], -1.0, 1.0)))
    roll = math.atan2(M[2, 1], M[2, 2])
    return tuple(round(math.degrees(a), 3) for a in (yaw, pitch, roll))


def mount_world_pose(model, data, mount: CameraMount) -> tuple[np.ndarray, np.ndarray]:
    """World position and camera axes (columns x, y, z; MuJoCo convention) of ``mount``."""
    b = body_id(model, mount.body)
    Rb = np.asarray(data.xmat[b]).reshape(3, 3)
    return np.asarray(data.xpos[b]) + Rb @ np.asarray(mount.pos), Rb @ mount.rotation()


def reattach(model, data, mount: CameraMount, body: str) -> CameraMount:
    """``mount`` re-parented to ``body`` with its world pose unchanged."""
    p, R = mount_world_pose(model, data, mount)
    b = body_id(model, body)
    Rb = np.asarray(data.xmat[b]).reshape(3, 3)
    pos = Rb.T @ (p - np.asarray(data.xpos[b]))
    return CameraMount(
        mount.name,
        body,
        tuple(round(float(v), 4) for v in pos),
        rpy_from_rotation(Rb.T @ R),
        mount.fovy,
    )


def _visual_geoms(model, b: int) -> list[int]:
    return [
        g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_group[g] in (1, 2)
    ]


def _support_geoms(model, b: int) -> list[int]:
    """Visual geoms that physically carry body ``b``: its own; else its descendants' (a flange
    frame carries the gripper); else its nearest ancestor's (an end-effector frame)."""
    own = _visual_geoms(model, b)
    if own:
        return own
    kids, frontier = [], [b]
    while frontier:
        frontier = [c for c in range(model.nbody) if c != 0 and model.body_parentid[c] in frontier]
        kids += frontier
    below = [g for c in kids for g in _visual_geoms(model, c)]
    if below:
        return below
    a = b
    while a > 0:
        a = int(model.body_parentid[a])
        up = _visual_geoms(model, a)
        if up and a > 0:
            return up
    return []


def _ray_to_geom(model, data, g: int, pnt: np.ndarray, vec: np.ndarray) -> float:
    import mujoco

    pnt, vec = np.asarray(pnt, float), np.asarray(vec, float)
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        return float(mujoco.mj_rayMesh(model, data, g, pnt, vec))
    return float(
        mujoco.mju_rayGeom(
            np.asarray(data.geom_xpos[g], float),
            np.asarray(data.geom_xmat[g], float),
            np.asarray(model.geom_size[g], float),
            pnt,
            vec,
            int(model.geom_type[g]),
        )
    )


def look_rotation(forward: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World camera axes (MuJoCo: looks along -z, +y up) looking along ``forward``, no roll."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight up or down
        x = np.cross(f, (1.0, 0.0, 0.0))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.stack([x, y, -f], axis=1)


def _sphere_dirs(n: int = 64) -> np.ndarray:
    """Roughly uniform unit directions (Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = math.pi * (1 + 5**0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def _clear_view(model, data, p: np.ndarray, target: np.ndarray, spread_deg: float = 10.0) -> float:
    """Fraction of a small ray cone from ``p`` to ``target`` not blocked by robot geometry."""
    import mujoco

    fwd = target - p
    dist = float(np.linalg.norm(fwd))
    R = look_rotation(fwd)
    s = math.tan(math.radians(spread_deg))
    offsets = [(0, 0), (s, 0), (-s, 0), (0, s), (0, -s)]
    group = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    robot = ("robot0_", "gripper0_", "mount0_")
    clear = 0
    for ox, oy in offsets:
        v = R @ np.array([ox, oy, -1.0])
        v /= np.linalg.norm(v)
        gid = np.array([-1], dtype=np.int32)
        t = mujoco.mj_ray(model, data, p, v, group, 1, -1, gid)
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid[0]]) or ""
            if gid[0] >= 0
            else ""
        )
        clear += not (0 <= t < dist - 0.02 and name.startswith(robot))
    return clear / len(offsets)


def snap_to_body(
    model,
    data,
    mount: CameraMount,
    body: str,
    target: np.ndarray,
    standoff_m: float = 0.015,
) -> CameraMount:
    """Put ``mount`` on the surface of ``body``, aimed at ``target`` (world).

    Candidate spots all around the link are tried; the camera goes where its view
    of ``target`` (e.g. the gripper's workspace) is least blocked by the robot,
    preferring the side of the link facing where the camera was. It sits
    ``standoff_m`` out from the surface, upright. ``world`` keeps the current
    world pose.
    """
    if body == WORLD:
        return reattach(model, data, mount, WORLD)
    b = body_id(model, body)
    target = np.asarray(target, float)
    p_now, _ = mount_world_pose(model, data, mount)
    geoms = _support_geoms(model, b)
    center = (
        np.mean([data.geom_xpos[g] for g in geoms], axis=0) if geoms else np.asarray(data.xpos[b])
    )
    pref = p_now - center
    pref = pref / np.linalg.norm(pref) if np.linalg.norm(pref) > 1e-6 else np.array([0, 0, 1.0])
    reach = 1.0  # start each ray well outside any link and shoot back at its center

    def spot(u):
        hits = [
            t
            for t in (_ray_to_geom(model, data, g, center + reach * u, -u) for g in geoms)
            if t >= 0
        ]
        return (center + (reach - min(hits)) * u if hits else center) + standoff_m * u

    best, best_score = None, -1.0
    for u in np.vstack([pref[None], _sphere_dirs()]):
        p = spot(u)
        if np.linalg.norm(target - p) < 0.05:
            continue
        score = _clear_view(model, data, p, target) + 0.25 * float(u @ pref)  # view first
        if score > best_score:
            best, best_score = p, score
    p = best if best is not None else spot(pref)
    fwd = target - p
    R = look_rotation(fwd if np.linalg.norm(fwd) > 1e-6 else -pref)
    Rb = np.asarray(data.xmat[b]).reshape(3, 3)
    pos = Rb.T @ (p - np.asarray(data.xpos[b]))
    return CameraMount(
        mount.name,
        body,
        tuple(round(float(v), 4) for v in pos),
        rpy_from_rotation(Rb.T @ R),
        mount.fovy,
    )


def mount_module(mount: CameraMount, modules=None):
    """The stereo module of a mount, at the mount's baseline (``baseline_mm``) if it sets one."""
    from ...stereo import ModuleRegistry

    module = (modules or ModuleRegistry()).load(mount.module)
    if mount.baseline_mm is not None:
        module = module.with_baseline(mount.baseline_mm / 1000)
    return module


def pair_baseline(model, left: str, right: str) -> float:
    """Distance (m) between two cameras of a compiled model, in their parent body's frame."""
    import mujoco

    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in (left, right)]
    if min(ids) < 0:
        raise ValueError(f"no cameras {left!r} / {right!r} in the model")
    return float(np.linalg.norm(model.cam_pos[ids[1]] - model.cam_pos[ids[0]]))


def set_pair_baseline(model, left: str, right: str, baseline_m: float) -> float:
    """Move a compiled stereo pair's eyes to ``baseline_m`` apart; returns the old baseline.

    The eyes move symmetrically about their midpoint, along the line between them,
    so the module stays centered where the robot mounts it. This adjusts a stereo pair
    built into a robot model (e.g. the multi-camera arms' wrist module) without
    rebuilding the model; call ``mj_forward`` (or ``sim.forward()``) before rendering.
    Both cameras must share a parent body.
    """
    import mujoco

    baseline_m = float(baseline_m)
    if not 0.001 <= baseline_m <= 2.0:
        raise ValueError(f"stereo baseline must be within 1 mm and 2 m, got {baseline_m} m")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in (left, right)]
    if min(ids) < 0:
        raise ValueError(f"no cameras {left!r} / {right!r} in the model")
    if model.cam_bodyid[ids[0]] != model.cam_bodyid[ids[1]]:
        raise ValueError(f"{left!r} and {right!r} are on different bodies")
    pl, pr = (np.array(model.cam_pos[i], dtype=np.float64) for i in ids)
    old = float(np.linalg.norm(pr - pl))
    if old == 0:
        raise ValueError(f"{left!r} and {right!r} coincide; no baseline direction")
    mid, axis = (pl + pr) / 2, (pr - pl) / old
    model.cam_pos[ids[0]] = mid - axis * baseline_m / 2
    model.cam_pos[ids[1]] = mid + axis * baseline_m / 2
    return old


def expand_mount(mount: CameraMount, modules=None) -> dict[str, CameraMount]:
    """The cameras a mount renders: itself, or a stereo module's two eyes.

    A stereo mount's pose is its left eye's; the eyes are ``<name>_L`` / ``<name>_R``
    on the same parent body, so they move together.
    """
    if not mount.module:
        return {mount.name: mount}
    from ...stereo import eye_name, eye_poses

    module = mount_module(mount, modules)
    out = {}
    for eye, (p, R) in eye_poses(module, mount.pos, mount.rotation()).items():
        n = eye_name(mount.name, eye)
        out[n] = CameraMount(
            n, mount.body, tuple(round(float(v), 6) for v in p), rpy_from_rotation(R), mount.fovy
        )
    return out
