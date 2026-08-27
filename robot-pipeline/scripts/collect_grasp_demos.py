#!/usr/bin/env python3
"""G1-D cup-grasp demonstration collection for OpenVLA fine-tuning.

Launched inside the Isaac Sim Docker container:
    /isaac-sim/python.sh scripts/collect_grasp_demos.py \
        --episodes 50 --output-dir /workspace/outputs/grasp_demos

Each episode records the expert IK pipeline trajectory as (RGB, 7-D delta
action, instruction) tuples in a format ready for OpenVLA dataset conversion.

Data layout (one directory per episode)::

    grasp_demos/
      episode_0000/
        meta.json          # episode-level metadata
        step_0000/
          image.png        # 640×480 head-camera RGB (resize to 256×256 for OpenVLA)
          action.json      # {"dx_m": …, "dy_m": …, "dz_m": …, …}
        step_0001/
          …
      dataset.json          # global manifest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Isaac Sim bootstrap ───────────────────────────────────────────────────
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# fmt: off
# Must match the main run_g1d_simple_room_vln.py constants exactly.
ROBOT_PRIM_PATH   = "/World/G1_D"
LEFT_WHEEL_JOINT  = "Left_Wheel_Joint"
RIGHT_WHEEL_JOINT = "Right_Wheel_Joint"
WHEEL_RADIUS_M    = 0.0848
WHEEL_BASE_M      = 0.4062
PHYSICS_HZ        = 60
ROOM_FLOOR_Z_M    = -0.7695
ROBOT_ROOT_ON_FLOOR_Z_M = -0.664
CAMERA_HEIGHT_ABOVE_FLOOR_M = 1.34
CAMERA_FORWARD_OFFSET_M     = 0.18
CAMERA_DOWNWARD_PITCH_RAD   = math.radians(25.0)
# Keep the same calibrated head-camera pitch as the dashboard.  A larger
# pitch was tested and rejected because it produced black renderer frames;
# the collection gate must reject images until a visible-cup close-range view
# is established by the same live-search/alignment stack as the dashboard.
COLLECTION_CAMERA_DOWNWARD_PITCH_RAD = CAMERA_DOWNWARD_PITCH_RAD

RIGHT_PALM_LINK = "right_hand_palm_link"
RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",  "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",  "right_hand_middle_0_joint",
    "right_hand_middle_1_joint", "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RIGHT_HAND_OPEN_RAD = np.array(
    [0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08], dtype=np.float64,
)
RIGHT_HAND_CLOSED_RAD = np.array(
    [0.65, 0.45, -1.25, 1.2, 1.35, 1.2, 1.35], dtype=np.float64,
)
RIGHT_HAND_FINGERTIP_REACH_M = 0.16
RIGHT_FINGERTIP_LINKS = (
    "right_hand_thumb_2_link",  "right_hand_middle_1_link",
    "right_hand_index_1_link",
)
RIGHT_ARM_PREGRASP_SEED_RAD = np.array(
    [0.35, -0.16, 0.0, 0.87, 0.0, 0.0, 0.0], dtype=np.float64,
)
# These are the two audited positions from a successful Dashboard execution
# (20260810T015350Z), expressed in the Family Home map frame.  The first is
# where the live RGB gate sees the cup; the second is the final arm-safe pose
# after APPROACH_AND_ALIGN and micro-approach.  Do not use the old arbitrary
# dining-table pose (1.90, 2.22, 104°): it was neither a gated observation
# pose nor the verified Expert grasp pose.
VLA_OBSERVATION_POSE = (2.00829, 1.95716, math.radians(102.0))
EXPERT_MANIPULATION_POSE = (1.86689, 2.35010, math.radians(100.0))
# [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
RIGHT_ARM_LIMITS_RAD = np.array([
    (-3.0892, 2.6704),  (-2.2515, 1.5882),  (-2.6180, 2.6180),
    (-1.0472, 2.0944),  (-1.9722, 1.9722),  (-1.6144, 1.6144),
    (-1.6144, 1.6144),
], dtype=np.float64)
# fmt: on

# ── OpenVLA-compatible instructions (mixed Chinese / English for robustness) ──
DEFAULT_INSTRUCTIONS = [
    "pick up the coffee cup",
    "grasp the cup on the table",
    "grab the coffee cup",
    "reach for the cup and pick it up",
    "拿起咖啡杯",
    "抓住桌上的杯子",
    "把杯子拿起来",
    "伸手拿起杯子",
    "move the robot hand toward the coffee cup",
    "pick the cup from the dining table",
    "请拿起桌上的杯子",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of demonstration episodes (default: 100)")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "outputs/grasp_demos",
                   help="Root output directory")
    p.add_argument("--headless", action="store_true",
                   help="Run without the Isaac Sim GUI")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--record-every-n-ik", type=int, default=5,
                   help="Record one frame every N IK iterations (default: 5)")
    p.add_argument("--max-episode-length", type=int, default=300,
                   help="Maximum frames per episode before early termination")
    p.add_argument("--base-variation-xy-m", type=float, default=0.15,
                   help="Random XY jitter for robot base pose (m)")
    p.add_argument("--base-variation-yaw-deg", type=float, default=10.0,
                   help="Random yaw jitter for robot base pose (deg)")
    p.add_argument("--cup-variation-xy-m", type=float, default=0.08,
                   help="Random XY jitter for cup placement (m)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Isaac Sim / G1-D utility functions (duplicated from run_g1d_simple_room_vln.py
#  to keep this script self-contained).
# ═══════════════════════════════════════════════════════════════════════════


def link_world_position(robot, link_name: str):
    import numpy as np
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/base_link/{link_name}")
    if not prim.IsValid():
        prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/{link_name}")
    if not prim.IsValid():
        for p in stage.Traverse():
            if p.GetName() == link_name:
                prim = p
                break
    from pxr import UsdGeom
    xform = UsdGeom.Xformable(prim)
    t = xform.ComputeLocalToWorldTransform(0).ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def _prim_world_position(prim) -> np.ndarray:
    import numpy as np
    from pxr import UsdGeom, Usd
    xform = UsdGeom.Xformable(prim)
    t = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def _configure_joint_drives(robot) -> None:
    names = robot.dof_names
    stiffness = np.zeros(len(names), dtype=np.float32)
    damping = np.zeros(len(names), dtype=np.float32)
    TORSO = {"LZ_mt_Joint", "LZ_it_Joint", "Yaw_Joint", "torso_Joint"}
    for i, name in enumerate(names):
        if name in (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT):
            damping[i] = 20.0
        elif name in TORSO:
            stiffness[i] = 2000.0
            damping[i] = 150.0
        elif "hand_" in name:
            stiffness[i] = 40.0
            damping[i] = 3.0
        else:
            stiffness[i] = 80.0
            damping[i] = 8.0
    robot.set_dof_gains(stiffnesses=stiffness, dampings=damping)
    wheel_idx = robot.get_dof_indices([LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]).numpy().tolist()
    robot.set_dof_max_efforts([40.0, 40.0], dof_indices=wheel_idx)
    hand_names = [n for n in names if "hand_" in n]
    if hand_names:
        robot.set_dof_max_efforts(
            [12.0] * len(hand_names),
            dof_indices=robot.get_dof_indices(hand_names).numpy().tolist(),
        )


def _upright_torso(robot) -> None:
    TORSO = ["LZ_mt_Joint", "LZ_it_Joint", "Yaw_Joint", "torso_Joint"]
    try:
        pos = robot.get_dof_positions().numpy()[0].copy()
    except Exception:
        return
    for jn in TORSO:
        try:
            pos[int(robot.get_dof_indices([jn]).numpy()[0])] = 0.0
        except Exception:
            continue
    robot.set_dof_position_targets(pos)
    for _ in range(20):
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        simulation_app.update()


def set_assisted_robot_pose(robot, x: float, y: float, yaw: float) -> None:
    ori = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)
    robot.set_world_poses(
        positions=np.array([x, y, ROBOT_ROOT_ON_FLOOR_Z_M], dtype=np.float32),
        orientations=ori,
    )


def robot_pose(robot) -> Tuple[float, float, float]:
    pos, ori = robot.get_world_poses()
    p = pos.numpy()[0]
    q = ori.numpy()[0]
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                     1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
    return float(p[0]), float(p[1]), float(yaw)


def _base_pitch_rad(robot) -> float:
    _, ori = robot.get_world_poses()
    q = ori.numpy()[0]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sin_p = 2.0 * (w * y - z * x)
    return float(math.asin(max(-1.0, min(1.0, sin_p))))


def head_camera_pose(robot, downward_pitch_rad: float = CAMERA_DOWNWARD_PITCH_RAD):
    stage = stage_utils.get_current_stage()
    head_prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/base_link/head_link")
    if not head_prim.IsValid():
        head_prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/head_link")
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xformable(head_prim)
    m = xform.ComputeLocalToWorldTransform(0)
    t = m.ExtractTranslation()
    r = m.ExtractRotation()
    q = r.GetQuaternion()  # (real, i, j, k)
    head_w = float(q.GetReal())
    head_x = float(q.GetImaginary()[0])
    head_y = float(q.GetImaginary()[1])
    head_z = float(q.GetImaginary()[2])
    # Compose head rotation with downward pitch
    half_p = math.sin(downward_pitch_rad / 2.0)
    cp = math.cos(downward_pitch_rad / 2.0)
    cam_w = head_w * cp - head_y * half_p
    cam_x = head_x * cp - head_z * half_p
    cam_y = head_y * cp + head_w * half_p
    cam_z = head_z * cp + head_x * half_p
    norm = math.sqrt(cam_w**2 + cam_x**2 + cam_y**2 + cam_z**2)
    return (float(t[0]), float(t[1]), float(t[2])), (
        cam_w / norm, cam_x / norm, cam_y / norm, cam_z / norm,
    )


def camera_rgb(camera_prim):
    """Return the current Camera RGB frame using Isaac Sim 6's public API."""
    rgba = camera_prim.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    image = np.asarray(rgba)[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(
            image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255
        ).astype(np.uint8)
    return image


def save_camera_rgb(camera_prim, path: Path) -> bool:
    try:
        from PIL import Image
        img = Image.fromarray(camera_rgb(camera_prim))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), format="PNG")
        return True
    except Exception:
        return False


def _set_right_arm_pregrasp_seed(robot) -> dict:
    idx = robot.get_dof_indices(list(RIGHT_ARM_JOINTS)).numpy().tolist()
    start = robot.get_dof_positions().numpy()[0, idx].astype(np.float64)
    target = RIGHT_ARM_PREGRASP_SEED_RAD.astype(np.float64).copy()
    ROLL = 1
    total, roll_steps = 60, 18
    for step in range(total):
        if step < roll_steps:
            r = (step + 1) / roll_steps
            cur = start.copy()
            cur[ROLL] = start[ROLL] + r * (target[ROLL] - start[ROLL])
        else:
            r = (step + 1 - roll_steps) / (total - roll_steps)
            p2 = start.copy()
            p2[ROLL] = target[ROLL]
            cur = p2 + r * (target - p2)
        robot.set_dof_position_targets(cur, dof_indices=idx)
        simulation_app.update()
    actual = robot.get_dof_positions().numpy()[0, idx]
    return {
        "target_rad": target.tolist(),
        "actual_rad": np.asarray(actual, dtype=np.float64).tolist(),
        "max_error_rad": float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - target))),
    }


def _set_right_hand(robot, targets) -> dict:
    idx = robot.get_dof_indices(list(RIGHT_HAND_JOINTS)).numpy().tolist()
    for step in range(30):
        robot.set_dof_position_targets(targets, dof_indices=idx)
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        simulation_app.update()
    actual = robot.get_dof_positions().numpy()[0, idx]
    return {
        "target_rad": np.asarray(targets, dtype=np.float64).tolist(),
        "actual_rad": np.asarray(actual, dtype=np.float64).tolist(),
    }


def _quaternion_rotation_error_xyzw(target_xyzw, current_xyzw):
    import numpy as np
    t = np.asarray(target_xyzw, dtype=np.float64)
    c = np.asarray(current_xyzw, dtype=np.float64)
    dot = np.dot(t, c)
    if dot < 0.0:
        c = -c
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    angle = 2.0 * math.acos(dot)
    if abs(angle) < 1e-9:
        return np.zeros(3, dtype=np.float64)
    sin_half = math.sin(angle / 2.0)
    if abs(sin_half) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = (c * t[0] - t * c[0] - np.cross(c[1:], t[1:]))
    axis /= sin_half
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return axis / norm * angle


def _jacobian_link_row(robot, jacobian, link_name: str) -> int:
    # Isaac Sim 6's experimental WheeledRobot exposes ``link_names`` and
    # ``get_link_indices`` (not the legacy ``body_names`` property).  The
    # Jacobian omits the root link on some builds, hence the row adjustment.
    link_index = int(robot.get_link_indices(link_name).numpy()[0])
    if jacobian.shape[0] == len(robot.link_names):
        return link_index
    if jacobian.shape[0] == len(robot.link_names) - 1 and link_index > 0:
        return link_index - 1
    raise RuntimeError(
        f"cannot map link {link_name} index {link_index} to Jacobian "
        f"shape {jacobian.shape}"
    )


def _jacobian_dof_columns(robot, jacobian, arm_indices):
    extra = int(jacobian.shape[-1]) - int(robot.num_dofs)
    if extra not in (0, 6):
        raise RuntimeError(
            f"unexpected G1-D Jacobian columns: {jacobian.shape[-1]} "
            f"for {robot.num_dofs} DOFs"
        )
    return np.asarray([extra + int(index) for index in arm_indices], dtype=np.int64)


def move_right_palm_to(
    robot,
    target_world_m,
    *,
    maximum_cartesian_travel_m: float = 0.70,
    tolerance_m: float = 0.025,
    maximum_iterations: int = 180,
    progress_callback=None,
) -> dict:
    """Bounded DLS position-only IK for the right palm."""
    import numpy as np
    target = np.asarray(target_world_m, dtype=np.float64)
    arm_idx = robot.get_dof_indices(list(RIGHT_ARM_JOINTS)).numpy().tolist()

    start = link_world_position(robot, RIGHT_PALM_LINK)
    requested = float(np.linalg.norm(target - start))
    if requested > maximum_cartesian_travel_m:
        return {"success": False, "reason": "cartesian_target_outside_bounded_workspace",
                "requested_travel_m": requested, "maximum_cartesian_travel_m": maximum_cartesian_travel_m}

    targets = robot.get_dof_positions().numpy()[0, arm_idx].astype(np.float64)
    errors = []
    for it in range(maximum_iterations):
        cur = link_world_position(robot, RIGHT_PALM_LINK)
        err = target - cur
        en = float(np.linalg.norm(err))
        errors.append(en)
        if en <= tolerance_m:
            break

        jac = robot.get_jacobian_matrices().numpy()[0]
        row = _jacobian_link_row(robot, jac, RIGHT_PALM_LINK)
        cols = _jacobian_dof_columns(robot, jac, arm_idx)
        pos_jac = np.asarray(jac[row, :3, :][:, cols], dtype=np.float64)

        damp = 0.055
        dim = pos_jac.shape[0]
        delta = pos_jac.T @ np.linalg.solve(
            pos_jac @ pos_jac.T + (damp**2) * np.eye(dim), err,
        )
        delta = np.clip(delta, -0.025, 0.025)
        targets = np.clip(
            targets + delta,
            RIGHT_ARM_LIMITS_RAD[:, 0] + 0.03,
            RIGHT_ARM_LIMITS_RAD[:, 1] - 0.03,
        )
        robot.set_dof_position_targets(targets, dof_indices=arm_idx)
        for _ in range(3):
            simulation_app.update()
        if progress_callback:
            progress_callback(it + 1, maximum_iterations)

    final = link_world_position(robot, RIGHT_PALM_LINK)
    fe = float(np.linalg.norm(target - final))
    return {
        "success": fe <= tolerance_m,
        "target_world_m": target.tolist(),
        "start_world_m": start.tolist(),
        "final_world_m": final.tolist(),
        "requested_travel_m": requested,
        "final_error_m": fe,
        "iterations": len(errors),
        "min_iteration_error_m": min(errors) if errors else None,
    }


def _find_sim_grasp_bodies(target_world_m):
    import numpy as np
    from pxr import UsdPhysics, Usd
    stage = stage_utils.get_current_stage()
    palm = None
    candidates = []
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if name == RIGHT_PALM_LINK and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            palm = prim
        if path.startswith("/World/FamilyHomeObjects/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            pos = _prim_world_position(prim)
            candidates.append((float(np.linalg.norm(pos - target_world_m)), prim, pos))
    if palm is None:
        raise RuntimeError("palm rigid body not found")
    if not candidates:
        raise RuntimeError("no physics bodies found under FamilyHomeObjects")
    candidates.sort(key=lambda x: x[0])
    dist, obj_prim, obj_pos = candidates[0]
    return palm, obj_prim, {
        "selection": "nearest_dynamic_body_to_target_for_safety",
        "palm_prim_path": str(palm.GetPath()),
        "object_prim_path": str(obj_prim.GetPath()),
        "target_world_m": target_world_m.tolist(),
        "physics_object_world_m": obj_pos.tolist(),
        "anchor_error_m": float(dist),
    }


def _create_sim_grasp_constraint(palm_prim, object_prim, object_id: str):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage = stage_utils.get_current_stage()
    constraint_path = f"/World/G1DSimGrasp/{object_id}"
    if stage.GetPrimAtPath(constraint_path).IsValid():
        stage.RemovePrim(constraint_path)
    body = UsdPhysics.RigidBodyAPI(object_prim)
    body.GetKinematicEnabledAttr().Set(False)
    body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    constraint = UsdPhysics.FixedJoint.Define(stage, constraint_path)
    constraint.CreateBody0Rel().SetTargets([str(palm_prim.GetPath())])
    constraint.CreateBody1Rel().SetTargets([str(object_prim.GetPath())])
    time_code = Usd.TimeCode.Default()
    palm_world = UsdGeom.Xformable(palm_prim).ComputeLocalToWorldTransform(time_code)
    object_world = UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(time_code)
    palm_pos = palm_world.ExtractTranslation()
    object_anchor = object_world.GetInverse().Transform(palm_pos)
    constraint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    constraint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    constraint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(float(object_anchor[0]), float(object_anchor[1]), float(object_anchor[2]))
    )
    relative = object_world.ExtractRotationQuat().GetInverse() * palm_world.ExtractRotationQuat()
    imaginary = relative.GetImaginary()
    constraint.CreateLocalRot1Attr().Set(
        Gf.Quatf(float(relative.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2]))
    )
    constraint.CreateBreakForceAttr().Set(1e20)
    constraint.CreateBreakTorqueAttr().Set(1e20)
    simulation_app.update()
    return str(constraint_path)


def _configure_physical_grasp_friction(object_prim) -> dict:
    from pxr import UsdPhysics, UsdShade
    stage = stage_utils.get_current_stage()
    path = str(object_prim.GetPath())
    material_path = f"{path}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(1.5)
    api.CreateDynamicFrictionAttr().Set(1.2)
    api.CreateRestitutionAttr().Set(0.0)
    for descendant in [object_prim] + list(object_prim.GetAllChildren()):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(descendant).Bind(
                material, materialPurpose="physics"
            )
    return {"object_path": path, "material_path": material_path,
            "static_friction": 1.5, "dynamic_friction": 1.2}


# ═══════════════════════════════════════════════════════════════════════════
#  Scene helpers
# ═══════════════════════════════════════════════════════════════════════════

def _setup_family_home_scene():
    """Build the exact SimpleRoom + Sofa + FamilyHomeObjects scene of the UI."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
    from family_home_vln.household_objects import HOUSEHOLD_OBJECTS, require_prepared_assets
    require_prepared_assets()

    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)

    room_usd = ROOT / "Assets/room/IsaacSim/SimpleRoom.usd"
    sofa_usd = ROOT / "Assets/room/GenieSim/scenes/iros/SofaTablePlant.usd"
    if not room_usd.is_file() or not sofa_usd.is_file():
        raise RuntimeError("family-home collection assets are missing")
    stage_utils.add_reference_to_stage(str(room_usd).replace("\\", "/"), "/World/Room")
    stage_utils.add_reference_to_stage(str(sofa_usd).replace("\\", "/"), "/World/SofaSet")
    stage = stage_utils.get_current_stage()
    light = UsdLux.DomeLight.Define(stage, "/World/VLN/DomeLight")
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    sofa = stage.GetPrimAtPath("/World/SofaSet")
    UsdGeom.Xformable(sofa).AddTranslateOp().Set(Gf.Vec3d(-2.75, 1.85, 0.0))

    from family_home_vln.layout import HOME_FIXTURES, START_POSE as FH_START
    fixtures_root = UsdGeom.Xform.Define(stage, "/World/FamilyHome")
    fixtures_root.GetPrim().CreateAttribute("scene:profile", Sdf.ValueTypeNames.String).Set("family-home")
    for fixture in HOME_FIXTURES:
        cube = UsdGeom.Cube.Define(stage, f"/World/FamilyHome/{fixture.fixture_id}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*fixture.color_rgb)])
        transform = UsdGeom.Xformable(cube)
        transform.AddTranslateOp().Set(Gf.Vec3d(fixture.center_xy[0], fixture.center_xy[1], ROOM_FLOOR_Z_M + fixture.size_xyz[2] / 2.0))
        transform.AddScaleOp().Set(Gf.Vec3f(*fixture.size_xyz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    UsdGeom.Xform.Define(stage, "/World/FamilyHomeObjects")
    for index, item in enumerate(HOUSEHOLD_OBJECTS, start=1):
        root = UsdGeom.Xform.Define(stage, f"/World/FamilyHomeObjects/Item{index:02d}")
        root_transform = UsdGeom.Xformable(root)
        root_transform.AddTranslateOp().Set(Gf.Vec3d(item.position_xy[0], item.position_xy[1], ROOM_FLOOR_Z_M + item.support_height_above_floor_m - item.minimum_xyz[1]))
        root_transform.AddRotateZOp().Set(item.yaw_deg)
        frame = UsdGeom.Xform.Define(stage, f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame")
        UsdGeom.Xformable(frame).AddRotateXOp().Set(90.0)
        visual = UsdGeom.Xform.Define(stage, f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/Visual")
        visual.GetPrim().GetReferences().AddReference(str(item.prepared_usd).replace("\\", "/"))
        collision = UsdGeom.Cube.Define(stage, f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/Collision")
        minimum, maximum = np.asarray(item.minimum_xyz, dtype=np.float64), np.asarray(item.maximum_xyz, dtype=np.float64)
        collision.CreateSizeAttr(1.0)
        collision_transform = UsdGeom.Xformable(collision)
        collision_transform.AddTranslateOp().Set(Gf.Vec3d(*((minimum + maximum) / 2.0)))
        collision_transform.AddScaleOp().Set(Gf.Vec3f(*(maximum - minimum)))
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        if item.dynamic:
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            rigid_body.CreateRigidBodyEnabledAttr(True)
            rigid_body.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(item.mass_kg))

    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT],
        usd_path=str(ROOT / "Assets/g1_d_robot/g1_d.usd").replace("\\", "/"),
        positions=[FH_START.x, FH_START.y, ROOM_FLOOR_Z_M + 0.12],
    )
    _configure_joint_drives(robot)
    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    app_utils.update_app(steps=30)
    _upright_torso(robot)
    return robot, FH_START


def _create_head_camera(robot_x: float, robot_y: float, robot_yaw: float):
    """Create a head-mounted RGB camera at nominal G1-D head height."""
    # Nominal head-link pose in world frame, pitched CAMERA_DOWNWARD_PITCH_RAD
    # relative to the robot base.  The head is rigidly attached — torso joints
    # are locked upright.
    cam_x = robot_x + CAMERA_FORWARD_OFFSET_M * math.cos(robot_yaw)
    cam_y = robot_y + CAMERA_FORWARD_OFFSET_M * math.sin(robot_yaw)
    cam_z = ROOM_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M

    # Camera.set_world_pose expects a wxyz quaternion.  Keep the exact
    # world-Z(yaw) * local-Y(pitch) convention from the dashboard runner.
    cy, sy = math.cos(robot_yaw / 2.0), math.sin(robot_yaw / 2.0)
    cp, sp = (
        math.cos(COLLECTION_CAMERA_DOWNWARD_PITCH_RAD / 2.0),
        math.sin(COLLECTION_CAMERA_DOWNWARD_PITCH_RAD / 2.0),
    )
    orientation_wxyz = np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32)

    from isaacsim.sensors.camera import Camera
    cam = Camera(
        prim_path="/World/G1_D/head_camera",
        position=np.array([cam_x, cam_y, cam_z], dtype=np.float32),
        orientation=orientation_wxyz,
        frequency=30,
        resolution=(640, 480),
    )
    cam.initialize()
    app_utils.update_app(steps=10)
    return cam


# ═══════════════════════════════════════════════════════════════════════════
#  Expert IK pipeline (mirrors _execute_sim_pick)
# ═══════════════════════════════════════════════════════════════════════════

def expert_grasp_trajectory(
    robot,
    camera_prim,
    target_world,
) -> List[Dict[str, Any]]:
    """Run the multi-stage expert IK pipeline, recording every N-th step.

    Returns a list of frames, each with:
        image_path    – relative path to the step's RGB PNG
        palm_world_m  – [x, y, z] of palm before the action
        action_delta  – {"dx_m": …, …, "gripper": …} 7-D delta
    """
    frames: List[Dict[str, Any]] = []
    prev_palm = link_world_position(robot, RIGHT_PALM_LINK).copy()

    def _record_frame(phase: str, gripper: float):
        nonlocal prev_palm
        palm_now = link_world_position(robot, RIGHT_PALM_LINK)
        delta = palm_now - prev_palm
        rgb = camera_rgb(camera_prim)
        if rgb is None:
            raise RuntimeError("head camera did not produce RGB during expert collection")
        frames.append({
            "palm_world_m": prev_palm.tolist(),
            "delta_xyz_m": delta.tolist(),
            "delta_rot_rpy_rad": [0.0, 0.0, 0.0],  # position-only IK; orientation unchanged
            "gripper": gripper,  # 1=open, 0=closed
            "phase": phase,
            "rgb": rgb,
        })
        prev_palm = palm_now.copy()

    # ---- arm seed + open hand ----
    _set_right_arm_pregrasp_seed(robot)
    _set_right_hand(robot, RIGHT_HAND_OPEN_RAD)
    _record_frame("seed", 1.0)

    base_to_target = target_world[:2] - np.array([
        robot_pose(robot)[0], robot_pose(robot)[1],
    ], dtype=np.float64)
    planar = float(np.linalg.norm(base_to_target))
    direction = base_to_target / planar

    # ---- Stage 1: vertical table clearance ----
    palm_now = link_world_position(robot, RIGHT_PALM_LINK)
    clearance_z = max(float(palm_now[2]), target_world[2] + 0.18)
    vertical_target = np.array([float(palm_now[0]), float(palm_now[1]), clearance_z], dtype=np.float64)
    move_right_palm_to(robot, vertical_target, maximum_cartesian_travel_m=0.40, tolerance_m=0.035)
    _record_frame("raise", 1.0)

    # ---- Stage 2: overhead pregrasp ----
    overhead_target = target_world.copy()
    overhead_target[2] = clearance_z
    move_right_palm_to(robot, overhead_target, maximum_cartesian_travel_m=0.70, tolerance_m=0.035)
    _record_frame("horizontal", 1.0)

    # ---- Stage 3: pregrasp ----
    pregrasp = target_world.copy()
    pregrasp[:2] -= direction * 0.07
    pregrasp[2] += 0.04
    move_right_palm_to(robot, pregrasp, maximum_cartesian_travel_m=0.30, tolerance_m=0.045)
    _record_frame("pregrasp", 1.0)

    # ---- Stage 4: grasp approach ----
    grasp = target_world.copy()
    grasp[:2] -= direction * 0.05
    res = move_right_palm_to(robot, grasp, maximum_cartesian_travel_m=0.20, tolerance_m=0.060)
    _record_frame("descend", 1.0)

    # ---- Close hand + lift ----
    palm_obj = _find_sim_grasp_bodies(target_world)
    palm_prim, object_prim, body_sel = palm_obj

    _configure_physical_grasp_friction(object_prim)
    _set_right_hand(robot, RIGHT_HAND_CLOSED_RAD)
    _record_frame("close_hand", 0.0)

    constraint_path = _create_sim_grasp_constraint(palm_prim, object_prim, "cup_collect")
    app_utils.update_app(steps=5)
    _record_frame("attach", 0.0)

    lift_target = link_world_position(robot, RIGHT_PALM_LINK) + np.array([0.0, 0.0, 0.09], dtype=np.float64)
    move_right_palm_to(robot, lift_target, maximum_cartesian_travel_m=0.14, tolerance_m=0.025)
    _record_frame("lift", 0.0)

    # Verify stable hold
    for _ in range(45):
        simulation_app.update()

    object_center = _prim_world_position(object_prim)
    palm_center = _prim_world_position(palm_prim)
    lift_m = float(object_center[2] - target_world[2])
    hold_distance_m = float(np.linalg.norm(object_center - palm_center))
    success = bool(
        stage_utils.get_current_stage().GetPrimAtPath(constraint_path).IsValid()
        and lift_m >= 0.05
        and hold_distance_m <= 0.20
    )
    if not success:
        raise RuntimeError(
            f"expert verification failed: lift={lift_m:.3f} m, palm_distance={hold_distance_m:.3f} m"
        )
    return frames, success, constraint_path


# ═══════════════════════════════════════════════════════════════════════════
#  Data collection orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _randomize_cup_position(
    rng,
    variation_m: float = 0.08,
) -> np.ndarray:
    """Jitter the reviewed Family Home cup location, never the robot pose.

    The expert was calibrated at Item05's audited table position.  Deriving
    its location from a randomized robot pose moved it laterally by centimetres
    and invalidated an otherwise correct grasp.
    """
    # This is the Item05 root translation, not the visible mesh centre.
    # The grasp target below is recomputed from the USD world bound after the
    # teleport, so source-asset axis offsets cannot corrupt the action labels.
    from family_home_vln.household_objects import HOUSEHOLD_OBJECTS
    cup = HOUSEHOLD_OBJECTS[4]
    cup_x = cup.position_xy[0] + rng.uniform(-variation_m, variation_m)
    cup_y = cup.position_xy[1] + rng.uniform(-variation_m, variation_m)
    root_z = ROOM_FLOOR_Z_M + cup.support_height_above_floor_m - cup.minimum_xyz[1]
    return np.array([cup_x, cup_y, root_z], dtype=np.float64)


def _teleport_cup_to(cup_prim, x: float, y: float, z: float) -> None:
    from pxr import Gf
    attrs = cup_prim.GetAttributes()
    # Try to set translation on xform or direct attributes
    for attr_name in ("xformOp:translate",):
        attr = cup_prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(Gf.Vec3d(x, y, z))
            break
    simulation_app.update()


def _world_bbox_center(prim) -> np.ndarray:
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    center = (box.GetMin() + box.GetMax()) * 0.5
    return np.asarray([center[0], center[1], center[2]], dtype=np.float64)


def _set_collection_camera_pose(camera, x: float, y: float, yaw: float) -> None:
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = (
        math.cos(COLLECTION_CAMERA_DOWNWARD_PITCH_RAD / 2.0),
        math.sin(COLLECTION_CAMERA_DOWNWARD_PITCH_RAD / 2.0),
    )
    orientation = np.asarray([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32)
    camera.set_world_pose(
        np.asarray([
            x + CAMERA_FORWARD_OFFSET_M * math.cos(yaw),
            y + CAMERA_FORWARD_OFFSET_M * math.sin(yaw),
            ROOM_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M,
        ], dtype=np.float32),
        orientation,
        camera_axes="world",
    )
    app_utils.update_app(steps=4)


def _update_collection_camera_from_robot(robot, camera) -> None:
    """Use the physical head-link transform, exactly as the Dashboard does."""
    position, orientation = head_camera_pose(robot, CAMERA_DOWNWARD_PITCH_RAD)
    camera.set_world_pose(position, orientation, camera_axes="world")
    app_utils.update_app(steps=8)


def _aim_collection_camera_at(camera, x: float, y: float, yaw: float, target_world) -> None:
    """Keep the head-mounted camera centred on the randomized cup.

    The camera remains at its nominal G1-D head location; this only models the
    head pan/tilt needed to keep the task object in view.  It also guarantees
    that an image/action pair is not silently collected with the cup outside
    the frame.
    """
    eye = np.asarray([
        x + CAMERA_FORWARD_OFFSET_M * math.cos(yaw),
        y + CAMERA_FORWARD_OFFSET_M * math.sin(yaw),
        ROOM_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M,
    ], dtype=np.float32)
    target = np.asarray(target_world, dtype=np.float32)
    forward = target - eye
    if float(np.linalg.norm(forward)) < 1e-6:
        raise RuntimeError("camera target coincides with the camera")
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation = np.column_stack((forward, left, up))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        orientation = np.asarray([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ], dtype=np.float32)
    else:
        index = int(np.argmax(np.diag(rotation)))
        nxt, last = (index + 1) % 3, (index + 2) % 3
        scale = math.sqrt(1.0 + rotation[index, index] - rotation[nxt, nxt] - rotation[last, last]) * 2.0
        orientation = np.zeros(4, dtype=np.float32)
        orientation[index + 1] = 0.25 * scale
        orientation[0] = (rotation[last, nxt] - rotation[nxt, last]) / scale
        orientation[nxt + 1] = (rotation[nxt, index] + rotation[index, nxt]) / scale
        orientation[last + 1] = (rotation[last, index] + rotation[index, last]) / scale
    camera.set_world_pose(eye, orientation, camera_axes="world")
    app_utils.update_app(steps=8)


def _find_cup_prim_in_stage():
    """Find the scan_coffee_cup_05 prim in the loaded stage."""
    stage = stage_utils.get_current_stage()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "Item05" in path or "coffee_cup" in path.lower():
            # Try to find a transformable parent
            parent = prim
            for _ in range(3):
                parent = parent.GetParent()
                if parent and parent.HasAttribute("xformOp:translate"):
                    return parent
            return prim
    # Fallback: look for any prim under FamilyHomeObjects
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/FamilyHomeObjects/Item05"):
            return prim
    return None


def run_collection(args: argparse.Namespace) -> int:
    """Main collection loop."""
    import numpy as np

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup scene ──
    print("=" * 60)
    print("G1-D Cup-Grasp Demonstration Collection")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Seed:     {args.seed}")
    print("=" * 60)

    print("\n[1/4] Loading family-home scene …")
    robot, fh_start = _setup_family_home_scene()

    # WheeledRobot is backed by Isaac's physics tensor API.  Authoring the
    # USD prim and calling ``update`` is not enough to initialise that API:
    # attempting ``set_world_poses`` before the timeline is playing gives
    # "physics tensor entity is not valid".  This is deliberately the same
    # ordering used by the family-home runner, before any robot pose/control
    # operation or camera observation is made.
    from isaacsim.core.simulation_manager import SimulationManager
    SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ, device="cpu")
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    app_utils.play()
    app_utils.update_app(steps=24)
    _configure_joint_drives(robot)
    _upright_torso(robot)

    print("[2/4] Creating head camera …")
    cam = _create_head_camera(fh_start.x, fh_start.y, fh_start.yaw)

    print("[3/4] Locating cup prim in stage …")
    cup_prim = _find_cup_prim_in_stage()
    if cup_prim is None:
        print("WARNING: Could not find cup prim. Teleport will be skipped.")
    else:
        print(f"  Found: {cup_prim.GetPath()}")

    print(f"[4/4] Collecting {args.episodes} episodes …\n")

    dataset_meta: Dict[str, Any] = {
        "schema_version": 1,
        "task": "cup_grasping",
        "robot": "g1_d_wheeled",
        "action_space": {
            "type": "continuous_7d_delta",
            "labels": ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper"],
            "unnorm_key": "bridge_orig",
        },
        "observation_space": {
            "type": "rgb",
            "resolution": [640, 480],
            "camera": "head_camera",
        },
        "instructions": DEFAULT_INSTRUCTIONS,
        "episodes": [],
    }

    # Begin every recorded expert trajectory at the verified *operation* pose.
    # The upstream VLA observation pose is retained in metadata so a dataset
    # converter can pair the selected RGB handoff with the following grasp
    # sequence without pretending the former was captured at an arbitrary
    # dining-table location.
    DINING_BASE_X, DINING_BASE_Y, DINING_BASE_YAW = EXPERT_MANIPULATION_POSE

    successful = 0
    for ep in range(args.episodes):
        ep_dir = args.output_dir / f"episode_{ep:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Randomize base pose
        base_x = DINING_BASE_X + rng.uniform(-args.base_variation_xy_m, args.base_variation_xy_m)
        base_y = DINING_BASE_Y + rng.uniform(-args.base_variation_xy_m, args.base_variation_xy_m)
        base_yaw = DINING_BASE_YAW + math.radians(
            rng.uniform(-args.base_variation_yaw_deg, args.base_variation_yaw_deg)
        )

        # Teleport robot
        set_assisted_robot_pose(robot, base_x, base_y, base_yaw)
        _upright_torso(robot)
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        for _ in range(30):
            simulation_app.update()
        _set_collection_camera_pose(cam, base_x, base_y, base_yaw)

        # Randomize cup position
        cup_pos = _randomize_cup_position(rng, variation_m=args.cup_variation_xy_m)
        if cup_prim is not None:
            _teleport_cup_to(cup_prim, float(cup_pos[0]), float(cup_pos[1]), float(cup_pos[2]))
            target_world = _world_bbox_center(cup_prim)
        else:
            raise RuntimeError("family cup Item05 is required for collection")

        # Pick a random instruction
        instruction = rng.choice(DEFAULT_INSTRUCTIONS)

        print(f"  Ep {ep:04d} | base=({base_x:.2f},{base_y:.2f},{math.degrees(base_yaw):.0f}°) "
              f"cup=({cup_pos[0]:.2f},{cup_pos[1]:.2f}) | \"{instruction}\"", end=" ")

        try:
            frames, success, constraint_path = expert_grasp_trajectory(
                robot, cam, target_world,
            )
        except Exception as exc:
            print(f"✗ ERROR: {exc}")
            continue

        # Save frames
        saved_steps = 0
        for fi, frame in enumerate(frames):
            step_dir = ep_dir / f"step_{fi:04d}"
            step_dir.mkdir(exist_ok=True)

            # Save the RGB captured at this exact control state.  Capturing
            # here after the full trajectory would duplicate the last frame
            # and create invalid image/action pairs.
            img_path = step_dir / "image.png"
            from PIL import Image
            Image.fromarray(frame["rgb"]).save(img_path)

            # Save action
            action = {
                "dx_m": frame["delta_xyz_m"][0],
                "dy_m": frame["delta_xyz_m"][1],
                "dz_m": frame["delta_xyz_m"][2],
                "droll_rad": frame["delta_rot_rpy_rad"][0],
                "dpitch_rad": frame["delta_rot_rpy_rad"][1],
                "dyaw_rad": frame["delta_rot_rpy_rad"][2],
                "gripper": frame["gripper"],
                "phase": frame["phase"],
                "labels": ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper"],
                "unnorm_key": "bridge_orig",
            }
            (step_dir / "action.json").write_text(
                json.dumps(action, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            saved_steps += 1

        # Episode metadata
        ep_meta = {
            "episode_id": ep,
            "instruction": instruction,
            "base_pose": {"x": base_x, "y": base_y, "yaw": base_yaw},
            "vla_observation_pose": {
                "x": VLA_OBSERVATION_POSE[0],
                "y": VLA_OBSERVATION_POSE[1],
                "yaw": VLA_OBSERVATION_POSE[2],
                "role": "live_rgb_gated_staging_pose",
            },
            "expert_manipulation_pose": {
                "x": base_x,
                "y": base_y,
                "yaw": base_yaw,
                "role": "post_approach_and_align_arm_safe_pose",
            },
            "cup_root_world_m": cup_pos.tolist(),
            "cup_grasp_target_world_m": target_world.tolist(),
            "success": success,
            "frame_count": saved_steps,
            "constraint_path": constraint_path if success else None,
            "task_contract": "family_home_pick_lift_retain",
        }
        (ep_dir / "meta.json").write_text(
            json.dumps(ep_meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        if success:
            successful += 1
            print("✓")
        else:
            print("✗ (expert failed)")

        dataset_meta["episodes"].append(ep_meta)

        # Clean up constraint for next episode
        if success:
            try:
                stage_utils.get_current_stage().RemovePrim(constraint_path)
            except Exception:
                pass
        app_utils.update_app(steps=10)

    # ── Write dataset manifest ──
    dataset_meta["summary"] = {
        "total_episodes": args.episodes,
        "successful_episodes": successful,
        "total_frames": sum(ep["frame_count"] for ep in dataset_meta["episodes"]),
    }
    (args.output_dir / "dataset.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"Done. {successful}/{args.episodes} episodes succeeded.")
    print(f"Total frames: {dataset_meta['summary']['total_frames']}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "headless": args.headless,
        "width": 1920, "height": 1080,
        "active_gpu": 0, "multi_gpu": False,
    })

    import numpy as np
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

    try:
        exit_code = run_collection(args)
    finally:
        # Explicitly release the renderer/physics process after a batch.  It
        # otherwise stays alive after a completed headless collection and
        # holds the assigned GPU, preventing the dashboard from restarting.
        simulation_app.close()
    sys.exit(exit_code)
