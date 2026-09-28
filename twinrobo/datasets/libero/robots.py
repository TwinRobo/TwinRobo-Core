"""Additional robots for LIBERO scenes, registered by TwinRobo (LIBERO is not modified).

LIBERO builds ``robots=[name]`` as ``"Mounted" + name`` and only ships
``MountedPanda``. The classes here wrap robosuite's own arms the same way. They
inherit everything (init pose, gripper, mount, controller) and add only the
table-placement offsets LIBERO scenes look up.
"""

from __future__ import annotations

import numpy as np

#: Extra camera mounts of the multi-camera variants, relative to robot bodies. Wrist cameras
#: are placed relative to the arm's own ``eye_in_hand`` camera (same orientation), so the
#: recipe works for every robosuite arm; ``mast`` is a post beside the robot base.
MULTICAM_MOUNTS = {
    # A stereo module at the wrist, centered on the arm's own eye_in_hand camera and facing
    # the same way; its eyes are `<name>_L` / `<name>_R` (see twinrobo.stereo).
    "wrist": {"body": "right_hand", "like": "eye_in_hand", "module": "stereolabs/zed-x-mini/2.2mm"},
    # Over-the-shoulder camera on a post standing on the pedestal behind the arm (relative to
    # the robot base, whose origin is at the pedestal top), aimed at the table center.
    "mast": {
        "body": "base",
        "pos": (-0.25, -0.32, 0.85),
        "look_at": (0.6, 0.0, 0.0),
        "fovy": 60.0,
        "post": True,
    },
}
# Mounting hardware drawn for the added cameras: visual only (no contacts) and near massless,
# so replays and dynamics are unchanged.
_HARDWARE = {
    "group": "1",
    "contype": "0",
    "conaffinity": "0",
    "mass": "1e-5",
    "rgba": "0.12 0.12 0.13 1",
}

#: Robot catalog. ``base`` is the arm whose demos a robot can replay: a multi-camera variant
#: has the same kinematics as its base arm (cameras add no joints), so it replays its demos.
ROBOT_CATALOG = [
    {"name": "Panda", "label": "Franka Panda", "category": "Arms", "base": "Panda"},
    {"name": "Sawyer", "label": "Rethink Sawyer", "category": "Arms", "base": "Sawyer"},
    {"name": "UR5e", "label": "Universal Robots UR5e", "category": "Arms", "base": "UR5e"},
    {"name": "IIWA", "label": "KUKA LBR iiwa", "category": "Arms", "base": "IIWA"},
    {"name": "Jaco", "label": "Kinova Jaco", "category": "Arms", "base": "Jaco"},
    {"name": "Kinova3", "label": "Kinova Gen3", "category": "Arms", "base": "Kinova3"},
    {
        "name": "PandaMultiCam",
        "label": "Franka Panda · wrist stereo + mast",
        "category": "Multi-camera arms",
        "base": "Panda",
    },
    {
        "name": "UR5eMultiCam",
        "label": "UR5e · wrist stereo + mast",
        "category": "Multi-camera arms",
        "base": "UR5e",
    },
    {
        "name": "Kinova3MultiCam",
        "label": "Kinova Gen3 · wrist stereo + mast",
        "category": "Multi-camera arms",
        "base": "Kinova3",
    },
    # RoboCasa (robosuite 1.5) kitchens: a Panda on the Omron mobile base. Its own cameras
    # are the two base-mounted agent views and the wrist camera (what the demos record).
    {
        "name": "PandaOmron",
        "label": "Franka Panda on Omron mobile base",
        "category": "Mobile manipulators",
        "base": "PandaOmron",
        "simulator": "robocasa",
        "policy_size": 256,  # RoboCasa's recorded videos
        "robot_cameras": ["agentview_left", "agentview_right", "eye_in_hand"],
    },
]
# Cameras physically on the robot. robosuite's "robotview" is a free third-person viewpoint
# (it floats in front of the base), so it is a scene camera, not one of the robot's.
_ROBOT_CAMERAS = ["eye_in_hand"]


def _mount_cameras(name: str, m: dict) -> list[str]:
    """Camera names a `MULTICAM_MOUNTS` entry adds: a stereo module adds its two eyes."""
    return [f"{name}_L", f"{name}_R"] if "module" in m else [name]


for _r in ROBOT_CATALOG:
    _r.setdefault("simulator", "libero")
    multi = _r["name"].endswith("MultiCam")
    extra = [c for n, m in MULTICAM_MOUNTS.items() for c in _mount_cameras(n, m)] if multi else []
    _r["cameras"] = [f"robot0_{c}" for c in _r.get("robot_cameras", _ROBOT_CAMERAS) + extra]
    # Stereo modules on the robot: {camera group: {module, left eye, right eye}}.
    _r["stereo"] = (
        {
            f"robot0_{n}": {
                "module": m["module"],
                "left": f"robot0_{n}_L",
                "right": f"robot0_{n}_R",
            }
            for n, m in MULTICAM_MOUNTS.items()
            if "module" in m
        }
        if multi
        else {}
    )

#: Robots selectable in LIBERO scenes (the LIBERO default is "Panda").
ROBOTS = [r["name"] for r in ROBOT_CATALOG if r["simulator"] == "libero"]


def robots_for(simulator: str) -> list[str]:
    """Robots selectable in scenes of ``simulator`` ("libero" or "robocasa")."""
    return [r["name"] for r in ROBOT_CATALOG if r["simulator"] == simulator]


def base_robot(name: str) -> str:
    """The arm whose recorded demos ``name`` can replay (itself for plain arms)."""
    for r in ROBOT_CATALOG:
        if r["name"] == name:
            return r["base"]
    return name


_REGISTERED = False


def _placement(table_offset: float) -> dict:
    """LIBERO's scene-keyed base offsets, as for MountedPanda."""
    return {
        "bins": (-0.5, -0.1, 0),
        "empty": (-0.6, 0, 0),
        "table": lambda table_length: (table_offset - table_length / 2, 0, 0),
        "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
        "kitchen_table": lambda table_length: (table_offset - table_length / 2, 0, 0),
        "coffee_table": lambda table_length: (table_offset - table_length / 2, 0, 0),
        "living_room_table": lambda table_length: (table_offset - table_length / 2, 0, 0),
    }


def register_robots() -> None:
    """Register ``Mounted<Robot>`` for every non-Panda robot in `ROBOTS` (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import libero.libero.envs.robots  # noqa: F401  (registers LIBERO's MountedPanda first)
    from robosuite.models.robots import IIWA, Jaco, Kinova3, Sawyer, UR5e
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    def _shell_inertia_init(self, idn=0, _base=None):
        # robosuite 1.4.1's Sawyer head mesh is too thin for MuJoCo 3.x volume-based inertia
        # ("mesh volume is too small"); compute mesh inertia as a shell instead.
        _base.__init__(self, idn=idn)
        for mesh in self.asset.findall("mesh"):
            mesh.set("inertia", "shell")

    from libero.libero.envs.robots import MountedPanda

    mounted = {"Panda": MountedPanda}
    for base in (Sawyer, UR5e, IIWA, Jaco, Kinova3):
        name = f"Mounted{base.__name__}"
        offsets = _placement(-0.16)
        attrs = {"base_xpos_offset": property(lambda self, o=offsets: o)}
        if base is Sawyer:
            attrs["__init__"] = lambda self, idn=0, _b=base: _shell_inertia_init(self, idn, _b)
        cls = type(name, (base,), attrs)
        globals()[name] = cls  # ManipulatorModel subclasses register themselves by class name
        ROBOT_CLASS_MAPPING[name] = SingleArm
        mounted[base.__name__] = cls

    def _multicam_init(self, idn=0, _parent=None):
        _parent.__init__(self, idn=idn)
        add_camera_mounts(self)

    for r in ROBOT_CATALOG:
        if r["name"] == r["base"]:
            continue
        parent = mounted[r["base"]]
        name = f"Mounted{r['name']}"
        cls = type(
            name,
            (parent,),
            {"__init__": lambda self, idn=0, _p=parent: _multicam_init(self, idn, _p)},
        )
        globals()[name] = cls
        ROBOT_CLASS_MAPPING[name] = SingleArm
    _REGISTERED = True


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotate(q, v):
    """Rotate vector ``v`` by unit quaternion ``q`` (wxyz)."""
    qv = np.concatenate([[0.0], v])
    qc = q * np.array([1, -1, -1, -1])
    return _quat_mul(_quat_mul(q, qv), qc)[1:]


def _module_eyes(module_id: str):
    """``(StereoModule, vertical FoV of its eye)`` for a catalog stereo module."""
    from ...geometry import CameraIntrinsics
    from ...optics.distortion import RadialDistortion, field_of_view
    from ...spec import CameraSpec
    from ...stereo import ModuleRegistry

    module = ModuleRegistry().load(module_id)
    spec = CameraSpec.from_yaml(module.left.spec)
    i, res = spec.calibration["intrinsic"], spec.sensor.resolution
    intr = CameraIntrinsics(res.width, res.height, i["fx"], i["fy"], i["cx"], i["cy"])
    fov = field_of_view(intr, RadialDistortion.from_dict(spec.calibration.get("distortion")))
    return module, fov["v"]


def _add_stereo_module(model, body, cam: str, m: dict, pf: str, fmt) -> None:
    """Add a stereo module's two eye cameras and its housing to ``body``."""
    import xml.etree.ElementTree as ET

    import mujoco

    from ...stereo import eye_name, eye_poses

    ref = body.find(f"camera[@name='{pf}{m['like']}']")
    if ref is None:
        raise ValueError(f"{type(model).__name__}: no reference camera {pf}{m['like']}")
    q = np.array([float(x) for x in ref.get("quat", "1 0 0 0").split()])
    q = q / np.linalg.norm(q)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    center = np.array([float(x) for x in ref.get("pos", "0 0 0").split()])
    module, vfov = _module_eyes(m["module"])
    # Centered on the reference camera: the left eye (the module's pose) sits half a baseline left.
    eyes = eye_poses(module, np.zeros(3), np.eye(3))
    mid = (eyes["left"][0] + eyes["right"][0]) / 2
    for eye, (p_rel, R_rel) in eyes.items():
        p = center + R @ (p_rel - mid)
        qe = np.zeros(4)
        mujoco.mju_mat2Quat(qe, (R @ R_rel).reshape(-1))
        name = f"{pf}{eye_name(cam, eye)}"
        ET.SubElement(
            body,
            "camera",
            {"name": name, "mode": "fixed", "pos": fmt(p), "quat": fmt(qe), "fovy": f"{vfov:.6g}"},
        )
    w, h, d = (v / 1000 for v in (module.housing_mm or (module.baseline_m * 1000 + 40, 30, 30)))
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{pf}{cam}_housing",
            "type": "box",
            "size": fmt((w / 2, h / 2, d / 2)),
            "pos": fmt(center + R @ np.array([0.0, 0.0, d / 2 + 0.001])),  # behind the lenses
            "quat": fmt(q),
            **_HARDWARE,
        },
    )


def add_camera_mounts(model) -> None:
    """Add the `MULTICAM_MOUNTS` cameras to a robosuite robot model (after naming prefixes)."""
    import xml.etree.ElementTree as ET

    pf = model.naming_prefix
    fmt = lambda v: " ".join(f"{float(x):.6g}" for x in v)  # noqa: E731
    for cam, m in MULTICAM_MOUNTS.items():
        body = model.worldbody.find(f".//body[@name='{pf}{m['body']}']")
        if body is None:
            raise ValueError(f"{type(model).__name__}: no body {pf}{m['body']} for camera {cam}")
        attrs = {"name": f"{pf}{cam}", "mode": "fixed"}
        if "module" in m:
            _add_stereo_module(model, body, cam, m, pf, fmt)
            continue
        # Look-at: MuJoCo cameras look along -z with +y up and +x to the right.
        pos, target = np.array(m["pos"], float), np.array(m["look_at"], float)
        f = (target - pos) / np.linalg.norm(target - pos)
        x = np.cross(f, (0.0, 0.0, 1.0))
        x /= np.linalg.norm(x)
        y = np.cross(x, f)
        attrs.update(pos=fmt(pos), xyaxes=fmt((*x, *y)), fovy=f"{m['fovy']:.6g}")
        if m.get("post"):
            z = np.cross(x, y)  # camera +z: backwards from the view direction
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{pf}{cam}_post",
                    "type": "cylinder",
                    "size": "0.015",
                    "fromto": fmt((pos[0], pos[1], 0.0, pos[0], pos[1], pos[2] - 0.035)),
                    **_HARDWARE,
                },
            )
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{pf}{cam}_housing",
                    "type": "box",
                    "size": "0.04 0.02 0.022",
                    "pos": fmt(pos + 0.023 * z),
                    "xyaxes": fmt((*x, *y)),
                    **_HARDWARE,
                },
            )
        ET.SubElement(body, "camera", attrs)


#: Joint-name prefixes of the robot, its gripper and mount in robosuite scenes.
ROBOT_JOINT_PREFIXES = ("robot0_", "gripper0_", "mount0_")


def joint_layout(model) -> dict[str, tuple[int, int]]:
    """``{joint name: (qpos address, qpos size)}`` of a MuJoCo model."""
    import mujoco

    sizes = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        out[name] = (int(model.jnt_qposadr[j]), sizes[int(model.jnt_type[j])])
    return out


def remap_scene_qpos(
    src_qpos: np.ndarray,
    src_layout: dict[str, tuple[int, int]],
    dst_layout: dict[str, tuple[int, int]],
    dst_base_qpos: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Move the **scene** (non-robot) joints of a recorded trajectory into another robot's model.

    Robot, gripper and mount joints are not transferred: the destination robot
    keeps ``dst_base_qpos`` (its home pose). Returns ``(qpos [T, nq_dst],
    scene joints missing from the destination)``.
    """
    out = np.repeat(np.asarray(dst_base_qpos, dtype=np.float64)[None], len(src_qpos), axis=0)
    missing = []
    for name, (adr, n) in src_layout.items():
        if name is None or name.startswith(ROBOT_JOINT_PREFIXES):
            continue
        if name not in dst_layout:
            missing.append(name)
            continue
        dadr, dn = dst_layout[name]
        assert dn == n, f"joint {name}: size {n} vs {dn}"
        out[:, dadr : dadr + n] = src_qpos[:, adr : adr + n]
    return out, missing
