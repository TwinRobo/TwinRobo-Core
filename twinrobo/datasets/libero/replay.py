"""Exact replay of recorded LIBERO episodes.

A recorded ``states`` / ``qpos`` trajectory does not fully determine the scene.
LIBERO fixtures (cabinet, stove, ...) have no joints, and ``env.reset()``
randomizes their placement. Their pose lives only in the episode's model XML
(the demo's ``model_file`` attribute, or a run record's ``scene.xml``). That XML cannot
be loaded as is: it references the recording machine's asset paths. So the
static top-level bodies' poses are read from it and applied to the live model.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def static_body_poses(model_xml: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{body name: (pos, quat wxyz)}`` for the worldbody's direct children in an MJCF string."""
    root = ET.fromstring(model_xml)
    world = root.find("worldbody")
    out = {}
    if world is None:
        return out
    for body in world.findall("body"):
        name = body.get("name")
        if not name:
            continue
        pos = np.array([float(v) for v in (body.get("pos") or "0 0 0").split()])
        quat = np.array([float(v) for v in (body.get("quat") or "1 0 0 0").split()])
        out[name] = (pos, quat)
    return out


def apply_fixture_placement(sim, model_xml: str) -> list[str]:
    """Set the pose of every jointless top-level body from the episode's MJCF. Returns their names.

    Bodies with joints (objects, the robot) are driven by the recorded state and
    are left alone.
    """
    import mujoco

    model = sim.model._model
    changed = []
    for name, (pos, quat) in static_body_poses(model_xml).items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0 or model.body_jntnum[b] > 0 or model.body_parentid[b] != 0:
            continue
        if name.startswith(("robot0_", "mount0_", "gripper0_")):
            continue  # the robot's placement belongs to the (possibly different) robot model
        if not (np.allclose(model.body_pos[b], pos) and np.allclose(model.body_quat[b], quat)):
            model.body_pos[b] = pos
            model.body_quat[b] = quat / np.linalg.norm(quat)
            changed.append(name)
    sim.forward()
    return changed
