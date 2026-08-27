#!/usr/bin/env python3
"""Collect FIRST-PERSON (ego-centric) expert demonstration trajectories for
OpenVLA-OFT training using MaChuanhao's DLS-IK expert controller.

Launched inside the Isaac Sim Docker container::

    /isaac-sim/python.sh scripts/collect_expert_demos_head.py \\
        --episodes 50 \
        --output-dir /workspace/outputs/family_home_vln/expert_demos_head

Key differences from ``collect_grasp_demos.py`` (legacy position-only IK):

* Uses the real MaChuanhao DLS-IK expert via :func:`g1d_expert_bridge.run_expert_pick`.
* Captures ego-centric head-camera RGB at 10 Hz during expert execution.
* Tracks full 6-DOF palm delta (translation + rotation), not position-only.
* Output is OpenVLA-OFT compatible: ``frame: "world"``, ``ready_for_training``,
  ``expert_evidence`` with lift-height and stable-hold gates.

Data layout (one directory per episode)::

    expert_demos_head/
      episode_0000/
        meta.json          # episode-level metadata (OFT-gated fields)
        step_0000/
          image.png        # 640×480 ego-centric head-camera RGB
          action.json      # {"dx_m": …, …, "frame": "world", …}
        step_0001/
          …
      manifest.json        # global manifest for OFT dataset builder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Isaac Sim bootstrap ─────────────────────────────────────────────────────
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import pure-Python family_home_vln modules at module level (BEFORE SimulationApp)
# to avoid a C++ abort that occurs when importing them after pxr/Fabric init.
from family_home_vln.household_objects import HOUSEHOLD_OBJECTS, require_prepared_assets
from family_home_vln.layout import HOME_FIXTURES, START_POSE as FH_START
from living_room_vln.manipulation import (
    BASE_POSE as LIVING_ROOM_BASE_POSE,
    CUP_CENTER,
    CUP_PRIM_PATH as LIVING_ROOM_CUP_PRIM_PATH,
    TABLE_PRIM_PATH as LIVING_ROOM_TABLE_PRIM_PATH,
    add_manipulation_station,
)

# ── Constants (mirror run_g1d_simple_room_vln.py exactly) ───────────────────
ROBOT_PRIM_PATH = "/World/G1_D"
LEFT_WHEEL_JOINT = "Left_Wheel_Joint"
RIGHT_WHEEL_JOINT = "Right_Wheel_Joint"
WHEEL_RADIUS_M = 0.0848
WHEEL_BASE_M = 0.4062
PHYSICS_HZ = 60
ROOM_FLOOR_Z_M = -0.7695
ROBOT_ROOT_ON_FLOOR_Z_M = -0.664
# High forward head-bracket mount used for the manipulation gaze.  It stays
# above the dining tabletop and clear of the torso while remaining attached
# to the robot's base pose; the separate audit camera is third-person only.
CAMERA_HEIGHT_ABOVE_FLOOR_M = 1.72
CAMERA_FORWARD_OFFSET_M = 0.25
# The G1-D USD optical-forward frame is rotated from the navigation base
# heading.  A rendered pose sweep in Family Home places the dining cup at the
# image centre with this fixed sensor-frame correction.
CAMERA_YAW_OFFSET_RAD = math.radians(-15.0)
CAMERA_DOWNWARD_PITCH_RAD = math.radians(61.0)
# Freeze the optical model for reproducible OpenVLA-OFT observations.  The
# USD fallback near plane is 1.0 m, which can clip the G1-D hand during close
# manipulation.  Keep the far plane unchanged and explicitly bring near to
# 10 cm.
CAMERA_FOCAL_LENGTH_MM = 50.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
CAMERA_VERTICAL_APERTURE_MM = 15.71625
CAMERA_NEAR_CLIP_M = 0.1
CAMERA_FAR_CLIP_M = 1_000_000.0
CAMERA_FX_PX = 1527.0818
CAMERA_FY_PX = 1527.0819
CAMERA_CX_PX = 320.0
CAMERA_CY_PX = 240.0

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
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "left_elbow_joint",
    "left_wrist_roll_joint",     "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
LEFT_ARM_VERTICAL_RAD = np.array(
    [0.0, 0.0, 0.0, math.pi / 2.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
RIGHT_HAND_OPEN_RAD = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64,
)
RIGHT_HAND_CLOSED_RAD = np.array(
    [0.65, 0.25, -1.32, 1.34, 1.35, 0.75, 0.75], dtype=np.float64,
)
RIGHT_ARM_LIMITS_RAD = np.array([
    (-3.0892, 2.6704),  (-2.2515, 1.5882),  (-2.6180, 2.6180),
    (-1.0472, 2.0944),  (-1.9722, 1.9722),  (-1.6144, 1.6144),
    (-1.6144, 1.6144),
], dtype=np.float64)

# High-reach arm pose that avoids the DLS-IK kinematic singularity at near-
# straight elbow.  Elbow at 1.8 rad (well-bent) places the palm above the cup
# so the expert only needs to descend, orient, and grasp.
HIGH_REACH_RAD = np.array(
    [-0.80, -0.32, -0.30, 1.80, 0.0, -0.50, 0.0],
    dtype=np.float64,
)

# Calibrated Family Home physical-grasp corridor.  The contact-only probe
# verified this local pose (with no transport joint): the cup is a few mm to
# the right of the nominal centreline and has a small upright yaw.  Collection
# jitters *around* this pose rather than sampling the edge of the arm's reach.
PHYSICAL_CUP_FORWARD_M = 0.687
PHYSICAL_CUP_RIGHT_M = 0.036
PHYSICAL_CUP_YAW_DEG = -4.0
# v13: restored the v8 baseline sampling distribution, measured from all 67
# accepted episodes of physical_v8_pick100 (2026-08-17): offsets are uniform
# +-5 mm in-plane and +-3 deg yaw around the audited nominal (fwd offset
# min=-4.97mm max=+4.94mm; right -4.79..+4.75mm; yaw -2.94..+2.85deg).
# The corridor added after v8 ([8,15] mm forward / yaw [-11,-6] deg) is
# entirely outside that cluster -- no v8 accepted episode ever sampled it --
# and v11/v12 running with it dropped acceptance ~6x (17% -> ~2.5%).
# v14 (2026-08-20): user asked for a wider but still small object pose
# variation for training diversity -- position jitter +-15 mm in-plane
# ("only a little, not too much") and yaw +-15 deg around the same nominal
# (yaw only, no roll/pitch, so the cup can never tip over).  Expect a lower
# acceptance rate than v13's ~14% because the sampling box is ~9x the area
# and the yaw band 5x wider than the v8 success cluster; keep --episodes
# large enough that --target-training-ready still terminates early.
PHYSICAL_SUCCESS_FORWARD_OFFSET_M = (-0.015, 0.015)
PHYSICAL_SUCCESS_RIGHT_OFFSET_M = (-0.015, 0.015)
PHYSICAL_SUCCESS_YAW_OFFSET_DEG = (-15.0, 15.0)
# One previously verified physical-grasp pose for deterministic repeatability
# testing.  The robot and high-reach arm seed are already fixed globally.
FIXED_TEST_FORWARD_OFFSET_M = 0.012
FIXED_TEST_RIGHT_OFFSET_M = 0.005
FIXED_TEST_YAW_OFFSET_DEG = -0.15

# Audited expert manipulation base pose from 20260810T015350Z dashboard run.
# This is the post-APPROACH_AND_ALIGN arm-safe pose, NOT the VLA observation pose.
EXPERT_MANIPULATION_POSE = (1.86689, 2.35010, math.radians(100.0))

# Set by the selected scene builder before the camera or episode loop starts.
ACTIVE_FLOOR_Z_M = ROOM_FLOOR_Z_M
ACTIVE_ROBOT_ROOT_Z_M = ROBOT_ROOT_ON_FLOOR_Z_M
ACTIVE_TABLE_PRIM_PATH = "/World/FamilyHome/dining_table"
ACTIVE_CUP_PRIM_PATH = ""
ACTIVE_BASE_POSE = EXPERT_MANIPULATION_POSE
ACTIVE_CAMERA_PITCH_RAD = CAMERA_DOWNWARD_PITCH_RAD

# ── OpenVLA-compatible instructions ─────────────────────────────────────────
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
                   help="Maximum number of collection attempts (default: 100)")
    p.add_argument(
        "--target-training-ready",
        type=int,
        default=None,
        help=(
            "Stop once this many accepted episodes exist in --output-dir. "
            "Failed attempts are quarantined and do not count."
        ),
    )
    p.add_argument(
        "--scene-profile", choices=("family-home", "living-room"),
        default="family-home",
        help="Collect in the original Family Home or scanned home_lab scene",
    )
    p.add_argument(
        "--camera-pitch-deg", type=float, default=61.0,
        help="downward pitch of the unobstructed head-mounted RGB camera",
    )
    p.add_argument(
        "--camera-calibration-only", action="store_true",
        help="render a head-camera pose sweep around the fixed cup and exit",
    )
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "outputs/family_home_vln/expert_demos_head",
                   help="Root output directory")
    p.add_argument("--headless", action="store_true",
                   help="Run without the Isaac Sim GUI")
    p.add_argument(
        "--active-gpu", type=int, default=0,
        help="Isaac Sim renderer GPU index (default: 0)",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "First episode index. By default, append after the highest existing "
            "episode_NNNN directory without overwriting accepted data."
        ),
    )
    p.add_argument("--base-variation-xy-m", type=float, default=0.0,
                   help="Deprecated: the training collector keeps the robot base fixed")
    p.add_argument("--base-variation-yaw-deg", type=float, default=0.0,
                   help="Deprecated: the training collector keeps the robot base fixed")
    p.add_argument("--cup-variation-xy-m", type=float, default=0.005,
                   help="Maximum small local XY offset of the upright cup (m)")
    p.add_argument("--cup-variation-yaw-deg", type=float, default=3.0,
                   help="Maximum rotation of the upright cup about world Z (deg)")
    p.add_argument("--cup-right-bias-m", type=float, default=0.0,
                   help="Deprecated compatibility option; no one-sided cup bias is used")
    p.add_argument(
        "--oft-manifest",
        type=Path,
        default=None,
        help=(
            "After collection, also write an OpenVLA-OFT-format manifest "
            "(instruction + image + world-frame 7-DoF action samples) to this "
            "path, reusing scripts/g1d_openvla_oft_data.build_manifest."
        ),
    )
    p.add_argument(
        "--fixed-condition",
        action="store_true",
        help=(
            "Disable all cup randomization and repeat one verified upright "
            "cup pose to measure physical-grasp repeatability."
        ),
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility functions (adapted from collect_grasp_demos.py /
#  run_g1d_simple_room_vln.py — all Isaac Sim imports are deferred to call time
#  so the module is importable without the simulation running).
# ═══════════════════════════════════════════════════════════════════════════════


def _configure_joint_drives(robot) -> None:
    import numpy as np

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
    wheel_idx = robot.get_dof_indices(
        [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]
    ).numpy().tolist()
    robot.set_dof_max_efforts([40.0, 40.0], dof_indices=wheel_idx)
    hand_names = [n for n in names if "hand_" in n]
    if hand_names:
        robot.set_dof_max_efforts(
            [12.0] * len(hand_names),
            dof_indices=robot.get_dof_indices(hand_names).numpy().tolist(),
        )


def _set_left_arm_vertical(robot, *, teleport: bool = False) -> None:
    indices = robot.get_dof_indices(list(LEFT_ARM_JOINTS)).numpy().tolist()
    if teleport:
        positions = robot.get_dof_positions().numpy()[0].copy()
        positions[indices] = LEFT_ARM_VERTICAL_RAD
        robot.set_dof_positions(positions)
    robot.set_dof_position_targets(LEFT_ARM_VERTICAL_RAD, dof_indices=indices)


def _upright_torso(
    robot, *, assisted_pose: tuple[float, float, float] | None = None
) -> None:
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
    import isaacsim.core.experimental.utils.app as _app_utils

    for _ in range(20):
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        if assisted_pose is not None:
            set_assisted_robot_pose(robot, *assisted_pose)
        _app_utils.update_app()


def set_assisted_robot_pose(robot, x: float, y: float, yaw: float) -> None:
    import numpy as np

    ori = np.array(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32
    )
    robot.set_world_poses(
        positions=np.array([x, y, ACTIVE_ROBOT_ROOT_Z_M], dtype=np.float32),
        orientations=ori,
    )
    robot.set_velocities(
        linear_velocities=[0.0, 0.0, 0.0],
        angular_velocities=[0.0, 0.0, 0.0],
    )
    _set_left_arm_vertical(robot)


def robot_pose(robot):
    import numpy as np

    pos, ori = robot.get_world_poses()
    p = pos.numpy()[0]
    q = ori.numpy()[0]
    yaw = math.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
    )
    return float(p[0]), float(p[1]), float(yaw)


def _articulation_link_transforms(robot) -> np.ndarray:
    view = robot._physics_articulation_view
    if view is None:
        raise RuntimeError("G1-D physics articulation view is not initialized")
    return view.get_link_transforms().numpy()


def link_world_position(robot, link_name: str) -> np.ndarray:
    import numpy as np

    link_index = int(robot.get_link_indices(link_name).numpy()[0])
    transforms = _articulation_link_transforms(robot)
    return np.asarray(transforms[0, link_index, :3], dtype=np.float64)


def link_world_orientation_xyzw(robot, link_name: str) -> np.ndarray:
    import numpy as np

    link_index = int(robot.get_link_indices(link_name).numpy()[0])
    transforms = _articulation_link_transforms(robot)
    return np.asarray(transforms[0, link_index, 3:7], dtype=np.float64)


def _prim_world_position(prim) -> np.ndarray:
    import numpy as np
    from pxr import UsdGeom, Usd

    xform = UsdGeom.Xformable(prim)
    t = xform.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def camera_rgb(camera_prim):
    """Return the current Camera RGB frame using Isaac Sim's public API."""
    import numpy as np

    rgba = camera_prim.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    image = np.asarray(rgba)[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(
            image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255
        ).astype(np.uint8)
    return image


def _quaternion_rotation_error_xyzw(
    target_xyzw: np.ndarray, current_xyzw: np.ndarray,
) -> np.ndarray:
    """Shortest world-frame quaternion error as a rotation vector."""
    import numpy as np

    tx, ty, tz, tw = (
        float(target_xyzw[0]),
        float(target_xyzw[1]),
        float(target_xyzw[2]),
        float(target_xyzw[3]),
    )
    cx, cy, cz, cw = (
        float(current_xyzw[0]),
        float(current_xyzw[1]),
        float(current_xyzw[2]),
        float(current_xyzw[3]),
    )
    # target * inverse(current)
    vector = np.asarray(
        [
            -tw * cx + tx * cw - ty * cz + tz * cy,
            -tw * cy + tx * cz + ty * cw - tz * cx,
            -tw * cz - tx * cy + ty * cx + tz * cw,
        ],
        dtype=np.float64,
    )
    scalar = float(tw * cw + tx * cx + ty * cy + tz * cz)
    if scalar < 0.0:
        scalar = -scalar
        vector = -vector
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, max(1e-9, scalar))
    return vector * (angle / norm)


def _black_metrics(image: np.ndarray):
    """Black-frame metrics for OFT quality gate."""
    luminance = image[..., :3].astype(np.float32).mean(axis=2)
    black = luminance < 12.0
    bottom = black[int(black.shape[0] * 0.55):]
    total_fraction = float(black.mean())
    bottom_fraction = float(bottom.mean()) if bottom.size else 1.0
    return total_fraction, bottom_fraction


# ═══════════════════════════════════════════════════════════════════════════════
#  Scene helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_family_home_scene():
    """Build the SimpleRoom + SofaTablePlant + FamilyHomeObjects scene."""
    print("  [setup] starting imports...", flush=True)
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
    print("  [setup] pxr imported", flush=True)
    # household_objects and layout are imported at module level (before SimulationApp)
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
    print("  [setup] WheeledRobot imported", flush=True)

    require_prepared_assets()
    print("  [setup] assets verified", flush=True)

    import isaacsim.core.experimental.utils.stage as _stage_utils
    import isaacsim.core.experimental.utils.app as _app_utils

    _stage_utils.create_new_stage()
    _stage_utils.set_stage_up_axis("Z")
    _stage_utils.set_stage_units(meters_per_unit=1.0)
    print("  [setup] stage created", flush=True)

    room_usd = ROOT / "Assets" / "room" / "IsaacSim" / "SimpleRoom.usd"
    sofa_usd = ROOT / "Assets" / "room" / "GenieSim" / "scenes" / "iros" / "SofaTablePlant.usd"
    if not room_usd.is_file() or not sofa_usd.is_file():
        raise RuntimeError("family-home collection assets are missing")
    _stage_utils.add_reference_to_stage(
        str(room_usd).replace("\\", "/"), "/World/Room",
    )
    print("  [setup] room added", flush=True)
    _stage_utils.add_reference_to_stage(
        str(sofa_usd).replace("\\", "/"), "/World/SofaSet",
    )
    print("  [setup] sofa added", flush=True)
    stage = _stage_utils.get_current_stage()
    light = UsdLux.DomeLight.Define(stage, "/World/VLN/DomeLight")
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    sofa = stage.GetPrimAtPath("/World/SofaSet")
    UsdGeom.Xformable(sofa).AddTranslateOp().Set(Gf.Vec3d(-2.75, 1.85, 0.0))
    print("  [setup] light + sofa moved", flush=True)

    fixtures_root = UsdGeom.Xform.Define(stage, "/World/FamilyHome")
    fixtures_root.GetPrim().CreateAttribute(
        "scene:profile", Sdf.ValueTypeNames.String
    ).Set("family-home")
    for fixture in HOME_FIXTURES:
        cube = UsdGeom.Cube.Define(
            stage, f"/World/FamilyHome/{fixture.fixture_id}"
        )
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*fixture.color_rgb)])
        transform = UsdGeom.Xformable(cube)
        transform.AddTranslateOp().Set(
            Gf.Vec3d(
                fixture.center_xy[0],
                fixture.center_xy[1],
                ROOM_FLOOR_Z_M + fixture.size_xyz[2] / 2.0,
            )
        )
        transform.AddScaleOp().Set(Gf.Vec3f(*fixture.size_xyz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    print(f"  [setup] {len(HOME_FIXTURES)} fixtures defined", flush=True)

    UsdGeom.Xform.Define(stage, "/World/FamilyHomeObjects")
    for index, item in enumerate(HOUSEHOLD_OBJECTS, start=1):
        root = UsdGeom.Xform.Define(
            stage, f"/World/FamilyHomeObjects/Item{index:02d}"
        )
        root_transform = UsdGeom.Xformable(root)
        root_transform.AddTranslateOp().Set(
            Gf.Vec3d(
                item.position_xy[0],
                item.position_xy[1],
                ROOM_FLOOR_Z_M
                + item.support_height_above_floor_m
                - item.minimum_xyz[1],
            )
        )
        root_transform.AddRotateZOp().Set(item.yaw_deg)
        frame = UsdGeom.Xform.Define(
            stage,
            f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame",
        )
        UsdGeom.Xformable(frame).AddRotateXOp().Set(90.0)
        visual = UsdGeom.Xform.Define(
            stage,
            f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/Visual",
        )
        visual.GetPrim().GetReferences().AddReference(
            str(item.prepared_usd).replace("\\", "/")
        )
        collision = UsdGeom.Cube.Define(
            stage,
            f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/Collision",
        )
        minimum = np.asarray(item.minimum_xyz, dtype=np.float64)
        maximum = np.asarray(item.maximum_xyz, dtype=np.float64)
        collision.CreateSizeAttr(1.0)
        collision_transform = UsdGeom.Xformable(collision)
        collision_transform.AddTranslateOp().Set(
            Gf.Vec3d(*((minimum + maximum) / 2.0))
        )
        collision_transform.AddScaleOp().Set(Gf.Vec3f(*(maximum - minimum)))
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        if item.dynamic:
            # v4 (regression fix): keep LOW Coulomb friction on the cup and
            # hand colliders.  The working physical_verify_v8 configuration
            # (67/400 accepted) had no friction material at all (PhysX
            # default ≈ 0.5); the later 3.0/2.5 "ceramic grip" material made
            # the middle phalanx stick to the box corner during the fold, so
            # the finger could no longer slide up the cup wall and every
            # attempt jammed at the open pose.
            # v10: STOP authoring any material — restore the exact v8
            # (physical_v8_pick100, 67/400) state where the cup and the
            # hand colliders carry NO explicit PhysX material and rely on
            # the USD/URDF import default.  The explicit μ=0.5 binding
            # (v7-v9) statistically killed the grasp: v7+v9 = 0/26 vs the
            # 17% v8 baseline (P(0/26 | 17%) ≈ 0.7%).  v9 telemetry showed
            # the same release-gap signature as v8 accepted runs (72 mm in
            # the 69-75 mm band) but the cup slipped at the micro-lift, so
            # the binding's coefficient override — not the geometry — is
            # the remaining physical differentiator.
            pass
        if item.dynamic:
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            rigid_body.CreateRigidBodyEnabledAttr(True)
            rigid_body.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(
                float(item.mass_kg)
            )
    print(f"  [setup] {len(HOUSEHOLD_OBJECTS)} household objects defined", flush=True)

    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT],
        usd_path=str(ROOT / "Assets" / "g1_d_robot" / "g1_d.usd").replace(
            "\\", "/"
        ),
        positions=[FH_START.x, FH_START.y, ROOM_FLOOR_Z_M + 0.12],
    )
    _configure_joint_drives(robot)
    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    _app_utils.update_app(steps=30)
    _upright_torso(robot)
    print("  [setup] scene complete", flush=True)
    return robot, FH_START


def _setup_living_room_scene():
    """Load the exact scanned home_lab asset and add the runtime cup station."""
    global ACTIVE_FLOOR_Z_M, ACTIVE_ROBOT_ROOT_Z_M, ACTIVE_TABLE_PRIM_PATH
    global ACTIVE_CUP_PRIM_PATH, ACTIVE_BASE_POSE
    from pxr import Gf, UsdLux
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
    import isaacsim.core.experimental.utils.app as _app_utils
    import isaacsim.core.experimental.utils.stage as _stage_utils

    scene_usd = ROOT / "scene_asset/living_room/home_lab.usda"
    robot_usd = ROOT / "Assets/g1_d_robot/g1_d.usd"
    if not scene_usd.is_file() or not robot_usd.is_file():
        raise RuntimeError("living-room scene or G1-D USD is missing")
    print("  [setup] creating living-room stage", flush=True)
    _stage_utils.create_new_stage()
    _stage_utils.set_stage_up_axis("Z")
    _stage_utils.set_stage_units(meters_per_unit=1.0)
    print("  [setup] referencing exact home_lab.usda", flush=True)
    _stage_utils.add_reference_to_stage(str(scene_usd), "/World/HomeLab")
    _app_utils.update_app(steps=12)
    print("  [setup] authoring cup station", flush=True)
    stage = _stage_utils.get_current_stage()
    light = UsdLux.DomeLight.Define(stage, "/World/LivingRoomSurveyLight")
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(0.95, 0.96, 1.0))
    station = add_manipulation_station(stage)

    ACTIVE_FLOOR_Z_M = 0.0
    ACTIVE_ROBOT_ROOT_Z_M = 0.1055
    ACTIVE_TABLE_PRIM_PATH = str(station["table_prim_path"])
    ACTIVE_CUP_PRIM_PATH = str(station["cup_prim_path"])
    ACTIVE_BASE_POSE = tuple(float(value) for value in station["base_pose"])
    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT],
        usd_path=str(robot_usd),
        positions=[ACTIVE_BASE_POSE[0], ACTIVE_BASE_POSE[1], ACTIVE_ROBOT_ROOT_Z_M],
    )
    _configure_joint_drives(robot)
    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    _app_utils.update_app(steps=30)
    _upright_torso(robot)
    print(f"  [setup] home_lab + physical cup station: {ACTIVE_CUP_PRIM_PATH}", flush=True)
    return robot, type("StartPose", (), {
        "x": ACTIVE_BASE_POSE[0], "y": ACTIVE_BASE_POSE[1], "yaw": ACTIVE_BASE_POSE[2],
    })()


# ═══════════════════════════════════════════════════════════════════════════════
#  Head camera
# ═══════════════════════════════════════════════════════════════════════════════


def _create_head_camera(robot_x: float, robot_y: float, robot_yaw: float):
    """Create a head-mounted ego-centric RGB camera at nominal G1-D head height."""
    import numpy as np

    cam_x = robot_x + CAMERA_FORWARD_OFFSET_M * math.cos(robot_yaw)
    cam_y = robot_y + CAMERA_FORWARD_OFFSET_M * math.sin(robot_yaw)
    cam_z = ACTIVE_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M
    camera_yaw = robot_yaw + CAMERA_YAW_OFFSET_RAD

    # world-Z(yaw) * local-Y(pitch) composition
    cy, sy = math.cos(camera_yaw / 2.0), math.sin(camera_yaw / 2.0)
    cp, sp = (
        math.cos(ACTIVE_CAMERA_PITCH_RAD / 2.0),
        math.sin(ACTIVE_CAMERA_PITCH_RAD / 2.0),
    )
    orientation_wxyz = np.array(
        [cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32
    )

    from isaacsim.sensors.camera import Camera

    cam = Camera(
        # Keep this camera in world space and update it from the robot head
        # pose.  A child of /World/G1_D receives the robot transform a second
        # time when Camera.set_world_pose() is used, which was the reason the
        # recorded view pointed at the sky instead of the cup.
        prim_path="/World/Sensors/G1DHeadCamera",
        position=np.array([cam_x, cam_y, cam_z], dtype=np.float32),
        orientation=orientation_wxyz,
        frequency=30,
        resolution=(640, 480),
    )
    cam.initialize()
    cam.set_clipping_range(
        near_distance=CAMERA_NEAR_CLIP_M,
        far_distance=CAMERA_FAR_CLIP_M,
    )
    import isaacsim.core.experimental.utils.app as _app_utils

    _app_utils.update_app(steps=10)
    clipping_range = tuple(float(value) for value in cam.get_clipping_range())
    if not np.allclose(
        clipping_range,
        (CAMERA_NEAR_CLIP_M, CAMERA_FAR_CLIP_M),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "head camera clipping range was not applied: "
            f"{clipping_range}"
        )
    return cam


def _create_dining_debug_camera():
    """Fixed third-person camera used only to audit robot/table geometry."""
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.experimental.utils.app as _app_utils

    eye = np.asarray([0.65, 1.20, 1.35], dtype=np.float32)
    target = np.asarray([1.95, 2.85, -0.05], dtype=np.float32)
    camera = Camera(
        prim_path="/World/Sensors/DiningTableAuditCamera",
        position=eye,
        orientation=_look_at_camera_pose(eye, target)[1],
        frequency=30,
        resolution=(640, 480),
    )
    camera.initialize()
    _app_utils.update_app(steps=10)
    return camera


def _set_head_camera_pose(camera, robot_x: float, robot_y: float, robot_yaw: float) -> None:
    """Update the head camera to track the current robot base pose."""
    import numpy as np

    cam_x = robot_x + CAMERA_FORWARD_OFFSET_M * math.cos(robot_yaw)
    cam_y = robot_y + CAMERA_FORWARD_OFFSET_M * math.sin(robot_yaw)
    cam_z = ACTIVE_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M

    camera_yaw = robot_yaw + CAMERA_YAW_OFFSET_RAD
    cy, sy = math.cos(camera_yaw / 2.0), math.sin(camera_yaw / 2.0)
    cp, sp = (
        math.cos(ACTIVE_CAMERA_PITCH_RAD / 2.0),
        math.sin(ACTIVE_CAMERA_PITCH_RAD / 2.0),
    )
    orientation = np.array(
        [cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32
    )
    camera.set_world_pose(
        np.array([cam_x, cam_y, cam_z], dtype=np.float32),
        orientation,
        camera_axes="world",
    )

    import isaacsim.core.experimental.utils.app as _app_utils

    _app_utils.update_app(steps=4)


def _set_camera_calibration_pose(
    camera,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    *,
    height_above_floor_m: float,
    forward_offset_m: float,
    yaw_offset_deg: float,
    pitch_deg: float,
) -> None:
    """Set one explicit candidate pose for head-camera visual calibration."""
    yaw = robot_yaw + math.radians(yaw_offset_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    orientation = np.asarray(
        [cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32
    )
    position = np.asarray([
        robot_x + forward_offset_m * math.cos(robot_yaw),
        robot_y + forward_offset_m * math.sin(robot_yaw),
        ACTIVE_FLOOR_Z_M + height_above_floor_m,
    ], dtype=np.float32)
    camera.set_world_pose(position, orientation, camera_axes="world")
    import isaacsim.core.experimental.utils.app as _app_utils
    _app_utils.update_app(steps=8)


def _set_head_camera_from_robot(camera, robot) -> None:
    """Attach the RGB sensor to the actual articulated head link.

    The simple-room robot root is offset from the visible articulated body.
    Computing a nominal pose from its navigation base can therefore produce a
    perfectly valid image from the wrong part of the scene.  Use the USD head
    transform so the same sensor is physically at the G1-D head for both VLN
    and VLA capture.
    """
    from pxr import UsdGeom
    import isaacsim.core.experimental.utils.app as _app_utils

    stage = _stage_utils.get_current_stage()
    head = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/base_link/head_link")
    if not head.IsValid():
        head = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/head_link")
    if not head.IsValid():
        raise RuntimeError("G1-D head_link is missing; cannot mount RGB camera")
    matrix = UsdGeom.Xformable(head).ComputeLocalToWorldTransform(0)
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotation().GetQuaternion()
    head_w = float(rotation.GetReal())
    head_i = rotation.GetImaginary()
    head_x, head_y, head_z = (float(head_i[i]) for i in range(3))
    half = ACTIVE_CAMERA_PITCH_RAD / 2.0
    cos_half, sin_half = math.cos(half), math.sin(half)
    # head quaternion × local-Y downward pitch, wxyz.
    orientation = np.asarray([
        head_w * cos_half - head_y * sin_half,
        head_x * cos_half - head_z * sin_half,
        head_y * cos_half + head_w * sin_half,
        head_z * cos_half + head_x * sin_half,
    ], dtype=np.float32)
    orientation /= max(float(np.linalg.norm(orientation)), 1e-9)
    camera.set_world_pose(
        np.asarray([translation[0], translation[1], translation[2]], dtype=np.float32),
        orientation,
        camera_axes="world",
    )
    _app_utils.update_app(steps=10)


def _look_at_camera_pose(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Isaac's +X-forward camera pose for a visible VLA target."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.asarray([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    left /= max(float(np.linalg.norm(left)), 1e-9)
    up = np.cross(forward, left)
    matrix = np.column_stack((forward, left, up))
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ], dtype=np.float32)
    else:
        index = int(np.argmax(np.diag(matrix)))
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = math.sqrt(1.0 + matrix[index, index] - matrix[j, j] - matrix[k, k]) * 2.0
        quat = np.zeros(4, dtype=np.float32)
        quat[index + 1] = 0.25 * scale
        quat[0] = (matrix[k, j] - matrix[j, k]) / scale
        quat[j + 1] = (matrix[j, index] + matrix[index, j]) / scale
        quat[k + 1] = (matrix[k, index] + matrix[index, k]) / scale
    return np.asarray(eye, dtype=np.float32), quat / np.linalg.norm(quat)


def _aim_head_camera_at_vla_target(camera, robot_x: float, robot_y: float, robot_yaw: float, target: np.ndarray) -> None:
    """Task-handoff view from the head mount, positioned ahead of the torso.

    Navigation uses the nominal fixed head pose.  At manipulation handoff the
    same camera rotates to the target, remaining 33 cm forward and 1.48 m
    above the floor so neither torso nor arm fills the lower image.
    """
    eye = np.asarray([
        robot_x + CAMERA_FORWARD_OFFSET_M * math.cos(robot_yaw),
        robot_y + CAMERA_FORWARD_OFFSET_M * math.sin(robot_yaw),
        ACTIVE_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M,
    ], dtype=np.float32)
    delta = np.asarray(target, dtype=np.float64) - eye
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    if float(np.linalg.norm(delta)) < 1e-6 or horizontal_distance < 1e-6:
        raise RuntimeError("head camera target coincides with its optical centre")
    # Camera.set_world_pose(..., camera_axes="world") uses the same
    # yaw + local-Y downward-pitch convention as the calibrated navigation
    # camera.  The generic rotation-matrix converter above is not compatible
    # with this convention for the dining-table heading (~100 degrees) and
    # produced a fully black render.
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    downward_pitch = math.atan2(float(-delta[2]), horizontal_distance)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(downward_pitch / 2.0), math.sin(downward_pitch / 2.0)
    orientation = np.asarray([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32)
    camera.set_world_pose(eye, orientation, camera_axes="world")
    import isaacsim.core.experimental.utils.app as _app_utils
    _app_utils.update_app(steps=8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cup finding / randomization
# ═══════════════════════════════════════════════════════════════════════════════


def _find_cup_prim_in_stage():
    """Find the Item05 (dining_cup) prim in the loaded stage."""
    import isaacsim.core.experimental.utils.stage as _stage_utils

    stage = _stage_utils.get_current_stage()
    if ACTIVE_CUP_PRIM_PATH:
        prim = stage.GetPrimAtPath(ACTIVE_CUP_PRIM_PATH)
        if prim.IsValid():
            return prim
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "Item05" in path or "coffee_cup" in path.lower():
            parent = prim
            for _ in range(3):
                parent = parent.GetParent()
                if parent and parent.HasAttribute("xformOp:translate"):
                    return parent
            return prim
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/FamilyHomeObjects/Item05"):
            return prim
    return None


def _randomize_cup_position(rng, variation_m: float = 0.08) -> np.ndarray:
    """Jitter Item05's location around its audited table position."""
    from family_home_vln.household_objects import HOUSEHOLD_OBJECTS

    cup = HOUSEHOLD_OBJECTS[4]
    cup_x = cup.position_xy[0] + rng.uniform(-variation_m, variation_m)
    cup_y = cup.position_xy[1] + rng.uniform(-variation_m, variation_m)
    root_z = (
        ROOM_FLOOR_Z_M + cup.support_height_above_floor_m - cup.minimum_xyz[1]
    )
    return np.array([cup_x, cup_y, root_z], dtype=np.float64)


def _teleport_cup_to(
    cup_prim, x: float, y: float, z: float, yaw_rad: float = 0.0
) -> None:
    """Reset the simulated cup to an upright, stationary world pose.

    The preceding physical grasp may have left the PhysX rigid body moving.
    Set it kinematic and clear velocity before the next attempt; the bridge
    subsequently refreshes its pose through its Isaac ``XformPrim`` wrapper.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    attr = cup_prim.GetAttribute("xformOp:translate")
    if attr.IsValid():
        attr.Set(Gf.Vec3d(x, y, z))
    orient_attr = cup_prim.GetAttribute("xformOp:orient")
    if not orient_attr.IsValid():
        orient_op = UsdGeom.Xformable(cup_prim).AddOrientOp(
            UsdGeom.XformOp.PrecisionDouble
        )
        orient_attr = orient_op.GetAttr()
    orient_attr.Set(
        Gf.Quatd(
            math.cos(yaw_rad / 2.0),
            0.0,
            0.0,
            math.sin(yaw_rad / 2.0),
        )
    )

    body = UsdPhysics.RigidBodyAPI(cup_prim)
    if body:
        body.GetKinematicEnabledAttr().Set(True)
        body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    import isaacsim.core.experimental.utils.app as _app_utils

    _app_utils.update_app(steps=10)


def _configure_right_hand_physical_friction(stage) -> int:
    """Bind finite Coulomb friction onto the right hand and the target cup.

    v4 (regression fix): the working v8 configuration had no friction
    material (PhysX default ≈ 0.5) and folded cleanly over the cup; the
    later 3.0/2.5 material made the phalanx stick to the box corner and
    jam every fold.  0.5 lets the fold slide but under-carries the
    contact-only micro-lift: the curled middle phalanx presses a downward
    component onto the far wall while the palm rises, so the cup slips
    slowly out of the clamp and falls (v5/v6 0/10 — same failure).

    v7: the earlier direct UsdPhysics.MaterialAPI.Apply loop only touched
    prims that Stage.Traverse() returns with CollisionAPI — the imported
    distal-finger collision meshes live inside instance prototypes and are
    invisible to traversal, so the hand kept its import-default friction
    and the µ change never reached the contact pair.  Bind a shared
    physics-purpose UsdShade.Material to the editable right-hand xform
    scope (USD inheritance carries it into the instance collision meshes —
    the pattern used by run_g1d_simple_room_vln._configure_physical_
    grasp_friction) and to the cup collider.

    v9: coefficient back to 0.5 (v8 parity) — the v8 accepted lifts ran at
    the default ≈0.5; the v5-v7 slips were the 12 N·m hook geometry, not
    the friction level.  The binding is kept so the hand side is explicit
    0.5 rather than the unverifiable URDF import default.
    """
    from pxr import UsdShade, UsdPhysics

    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/FamilyHomePhysicalGrasp"
    )
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.5)
    physics_material.CreateDynamicFrictionAttr(0.5)
    physics_material.CreateRestitutionAttr(0.0)

    palm_scope = ""
    for prim in stage.Traverse():
        if (
            prim.GetName() == RIGHT_PALM_LINK
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            palm_scope = str(prim.GetPath())
            break
    if not palm_scope:
        return 0

    cup_prim = _find_cup_prim_in_stage()
    object_prefix = str(cup_prim.GetPath()) + "/" if cup_prim else ""
    bound: set[str] = set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        is_object_collider = (
            bool(object_prefix)
            and path.startswith(object_prefix)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        is_right_hand_scope = (
            path.startswith(palm_scope)
            and not prim.IsInstance()
            and not prim.IsInstanceProxy()
            and (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.GetTypeName() == "Xform"
            )
        )
        if not (is_object_collider or is_right_hand_scope):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, materialPurpose="physics"
        )
        bound.add(path)
    return len(bound)


def _yaw_from_xyzw(quaternion_xyzw: np.ndarray) -> float:
    """Extract a planar heading from an Isaac XYZW quaternion."""
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _place_living_room_station_for_physics(robot, cup_prim) -> tuple[float, float, float, np.ndarray]:
    """Place the table/cup in the *live PhysX* G1-D reach envelope.

    ``home_lab.usda`` is referenced under a reconstruction transform whereas
    the G1-D articulation has an internal AGV root transform.  The high-level
    wrapper reports the requested navigation pose, but using that pose to
    place a manipulation target can put it more than a metre from the actual
    hand.  Derive the station pose from AGV_link's live physics transform
    instead.  This is the same coordinate system used by the DLS Jacobian.
    """
    from pxr import Gf
    import isaacsim.core.experimental.utils.stage as _stage_utils

    base_position = link_world_position(robot, "AGV_link")
    base_yaw = _yaw_from_xyzw(link_world_orientation_xyzw(robot, "AGV_link"))
    forward = np.asarray((math.cos(base_yaw), math.sin(base_yaw)), dtype=np.float64)
    right = np.asarray((math.sin(base_yaw), -math.cos(base_yaw)), dtype=np.float64)

    # The high-reach seed puts the right palm beside AGV_link.  Empirical DLS
    # telemetry in this scene shows its pinch centre is ~4 cm forward and
    # ~6 cm right of that palm pose.  Keep the cup in that local corridor;
    # a farther 45 cm lateral target saturated the torso/yaw joint before the
    # state machine could ever reach the descent phase.
    palm_position = link_world_position(robot, RIGHT_PALM_LINK)
    cup_xy = palm_position[:2] + 0.04 * forward + 0.06 * right
    # Centre the compact tabletop slightly behind the cup so the whole cup is
    # supported but the robot has clear access from its near edge.
    table_xy = cup_xy - 0.08 * forward
    stage = _stage_utils.get_current_stage()
    table = stage.GetPrimAtPath(ACTIVE_TABLE_PRIM_PATH)
    table_translate = table.GetAttribute("xformOp:translate")
    if not table_translate.IsValid():
        raise RuntimeError("living-room manipulation table has no translate op")
    table_translate.Set(Gf.Vec3d(float(table_xy[0]), float(table_xy[1]), 0.36))
    _teleport_cup_to(cup_prim, float(cup_xy[0]), float(cup_xy[1]), 0.795)
    return float(base_position[0]), float(base_position[1]), base_yaw, np.asarray(
        (float(cup_xy[0]), float(cup_xy[1]), 0.795), dtype=np.float64
    )


def _world_bbox_center(prim) -> np.ndarray:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    center = (box.GetMin() + box.GetMax()) * 0.5
    return np.asarray([center[0], center[1], center[2]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  Arm pre-positioning
# ═══════════════════════════════════════════════════════════════════════════════


def _set_arm_to_high_reach(
    robot, *, assisted_pose: tuple[float, float, float] | None = None
) -> np.ndarray:
    """Interpolate the right arm to HIGH_REACH_RAD (roll-first) to avoid
    the DLS-IK kinematic singularity at near-straight elbow.

    Returns the actual joint positions achieved after interpolation.
    """
    import numpy as np

    arm_indices = robot.get_dof_indices(
        list(RIGHT_ARM_JOINTS)
    ).numpy().tolist()
    current_arm = (
        robot.get_dof_positions().numpy()[0, arm_indices].astype(np.float64)
    )

    ROLL = 1  # right_shoulder_roll_joint
    total, roll_steps = 40, 12
    for step in range(total):
        if step < roll_steps:
            r = (step + 1) / roll_steps
            q = current_arm.copy()
            q[ROLL] = current_arm[ROLL] + r * (
                HIGH_REACH_RAD[ROLL] - current_arm[ROLL]
            )
        else:
            r = (step + 1 - roll_steps) / (total - roll_steps)
            phase2_start = current_arm.copy()
            phase2_start[ROLL] = HIGH_REACH_RAD[ROLL]
            q = phase2_start + r * (HIGH_REACH_RAD - phase2_start)
        robot.set_dof_position_targets(q, dof_indices=arm_indices)
        import isaacsim.core.experimental.utils.app as _app_utils

        for _ in range(2):
            if assisted_pose is not None:
                set_assisted_robot_pose(robot, *assisted_pose)
            _app_utils.update_app()

    final_arm = (
        robot.get_dof_positions().numpy()[0, arm_indices].astype(np.float64)
    )
    return final_arm


# ═══════════════════════════════════════════════════════════════════════════════
#  Main collection loop
# ═══════════════════════════════════════════════════════════════════════════════


def _meta_is_physical_training_ready(meta: dict[str, Any]) -> bool:
    evidence = meta.get("expert_evidence") or {}
    return bool(
        meta.get("success")
        and meta.get("ready_for_training")
        and evidence.get("physical_hold_verified")
        and not evidence.get("fixed_joint_created")
        and not evidence.get("fixed_joint_configured")
        and int(evidence.get("hold_contact_frames", 0)) >= 30
    )


def run_collection(args: argparse.Namespace) -> int:
    import numpy as np
    global ACTIVE_CAMERA_PITCH_RAD

    # The G1-D USD head frame uses the opposite local-Y sign to the generic
    # dashboard camera convention.  Permit signed values so a calibrated
    # physical head sensor can look down at the tabletop.
    if not -80.0 <= args.camera_pitch_deg <= 80.0:
        raise ValueError("--camera-pitch-deg must be between -80 and 80")
    ACTIVE_CAMERA_PITCH_RAD = math.radians(args.camera_pitch_deg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for pattern, prefix in (("episode_*", "episode_"), ("expert_run_*", "expert_run_")):
        for path in args.output_dir.glob(pattern):
            if path.is_dir():
                try:
                    existing_indices.append(int(path.name.removeprefix(prefix)))
                except ValueError:
                    continue
    start_index = (
        int(args.start_index)
        if args.start_index is not None
        else (max(existing_indices, default=-1) + 1)
    )
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    requested_indices = range(start_index, start_index + args.episodes)
    collisions = [
        index
        for index in requested_indices
        if (args.output_dir / f"episode_{index:04d}").exists()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing accepted episodes: "
            + ", ".join(f"episode_{index:04d}" for index in collisions)
        )
    existing_training_ready = 0
    for meta_path in args.output_dir.glob("episode_*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_training_ready += int(
                _meta_is_physical_training_ready(meta)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if (
        args.target_training_ready is not None
        and args.target_training_ready <= 0
    ):
        raise ValueError("--target-training-ready must be positive")
    if (
        args.target_training_ready is not None
        and existing_training_ready >= args.target_training_ready
    ):
        print(
            f"Already have {existing_training_ready} training-ready episodes; "
            f"target {args.target_training_ready} is met."
        )
        return 0

    # A resumed collection should not repeat the exact same randomization as
    # episode_0000 from a previous process.
    rng = np.random.default_rng(args.seed + start_index)

    print("=" * 60)
    print("G1-D First-Person Expert Demonstration Collection")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Seed:      {args.seed}")
    print(f"  Indices:   {start_index}..{start_index + args.episodes - 1}")
    if args.target_training_ready is not None:
        print(
            "  Target:    "
            f"{args.target_training_ready} accepted total "
            f"({existing_training_ready} already present)"
        )
    print(
        "  Camera:    ego-centric head-mounted "
        f"({args.camera_pitch_deg:.1f}° downward, clear of torso/hand)"
    )
    print(f"  Expert:    MaChuanhao DLS-IK via g1d_expert_bridge")
    print("=" * 60)

    # ── Setup scene ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {args.scene_profile} scene …")
    robot, fh_start = (
        _setup_living_room_scene()
        if args.scene_profile == "living-room"
        else _setup_family_home_scene()
    )

    from isaacsim.core.simulation_manager import SimulationManager
    import isaacsim.core.experimental.utils.app as _app_utils
    import isaacsim.core.experimental.utils.stage as _stage_utils

    SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ, device="cpu")
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    _app_utils.play()
    _app_utils.update_app(steps=24)
    _configure_joint_drives(robot)
    _upright_torso(robot)

    # ── Create head camera ───────────────────────────────────────────────
    print("[2/5] Creating ego-centric head camera …")
    cam = _create_head_camera(fh_start.x, fh_start.y, fh_start.yaw)
    audit_cam = _create_dining_debug_camera() if args.scene_profile == "family-home" else None

    # ── Find cup ─────────────────────────────────────────────────────────
    print("[3/5] Locating cup prim in stage …")
    cup_prim = _find_cup_prim_in_stage()
    if cup_prim is None:
        raise RuntimeError("Could not find cup prim Item05 in the stage")
    print(f"  Found: {cup_prim.GetPath()}")

    # ── Discover palm prim path ──────────────────────────────────────────
    print("[4/5] Discovering palm prim path …")
    from pxr import UsdPhysics

    stage = _stage_utils.get_current_stage()
    palm_prim_path = ""
    for prim in stage.Traverse():
        if (
            prim.GetName() == RIGHT_PALM_LINK
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            palm_prim_path = str(prim.GetPath())
            break
    if not palm_prim_path:
        raise RuntimeError("cannot find the G1-D right-palm rigid body")
    print(f"  Palm: {palm_prim_path}")
    # v10: friction binding disabled — v8 (67/400) ran with no explicit
    # material; the μ=0.5 bindings (v7-v9) coincided with 0/26.  Restore
    # v8 parity exactly.
    # friction_colliders = _configure_right_hand_physical_friction(stage)
    # print(f"  Physical-friction hand colliders: {friction_colliders}")

    # ── Collect episodes ─────────────────────────────────────────────────
    print(f"[5/5] Collecting {args.episodes} episodes …\n")

    manifest_episodes: list[dict[str, Any]] = []
    successful = 0
    attempts_this_run = 0
    training_ready_total = existing_training_ready

    for run_offset in range(args.episodes):
        if (
            args.target_training_ready is not None
            and training_ready_total >= args.target_training_ready
        ):
            break
        attempts_this_run += 1
        episode_id = start_index + run_offset
        # Keep the robot at the reviewed manipulation pose in every episode.
        # Only the upright cup is randomized, so variation does not move the
        # target outside the physical right-hand workspace.
        base_x, base_y, base_yaw = ACTIVE_BASE_POSE

        # Teleport robot
        set_assisted_robot_pose(robot, base_x, base_y, base_yaw)
        _set_left_arm_vertical(robot, teleport=True)
        _upright_torso(robot)
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        for _ in range(30):
            _app_utils.update_app()

        # Update head camera to ego-centric pose
        _set_head_camera_pose(cam, base_x, base_y, base_yaw)

        # ---- Place cup in the right-arm reachable sector ------------------
        # The G1-D right arm only converges when the cup sits ahead and to
        # the RIGHT of the base centreline: the reference's calibrated pose
        # puts it 67.5 cm forward / 3.1 cm right of facing.  Independent
        # base/cup jitter let the cup fall left of facing, where DLS-IK
        # stalls ~7 cm short of the grasp point and the close phase can
        # never make contact.  Sample the calibrated local offset with
        # jitter, then clamp onto the dining-table patch so the cup never
        # leaves the tabletop (table top spans x∈[1.375,2.725], y∈[2.64,3.46]).
        if args.fixed_condition:
            cup_local_forward_offset_m = FIXED_TEST_FORWARD_OFFSET_M
            cup_local_right_offset_m = FIXED_TEST_RIGHT_OFFSET_M
            cup_yaw_offset_deg = FIXED_TEST_YAW_OFFSET_DEG
        else:
            # Sample only the measured physical-grasp corridor.  These are
            # still independently randomized every episode, but avoid poses
            # where the cup handle makes a two-finger contact unable to
            # sustain a lift.
            cup_local_forward_offset_m = rng.uniform(
                *PHYSICAL_SUCCESS_FORWARD_OFFSET_M
            )
            cup_local_right_offset_m = rng.uniform(
                *PHYSICAL_SUCCESS_RIGHT_OFFSET_M
            )
            cup_yaw_offset_deg = rng.uniform(*PHYSICAL_SUCCESS_YAW_OFFSET_DEG)
        cup_yaw_rad = math.radians(PHYSICAL_CUP_YAW_DEG + cup_yaw_offset_deg)
        fwd_m = PHYSICAL_CUP_FORWARD_M + cup_local_forward_offset_m
        right_m = PHYSICAL_CUP_RIGHT_M + cup_local_right_offset_m
        f = (math.cos(base_yaw), math.sin(base_yaw))
        r = (math.sin(base_yaw), -math.cos(base_yaw))
        if args.scene_profile == "living-room":
            # The station is placed after the high-reach seed below, because
            # that is the frame from which the Expert starts Cartesian IK.
            cup_x, cup_y, cup_z = (float(value) for value in CUP_CENTER)
        else:
            cup_x = min(max(base_x + fwd_m * f[0] + right_m * r[0], 1.60), 1.95)
            cup_y = min(max(base_y + fwd_m * f[1] + right_m * r[1], 2.85), 3.15)
            cup_z = (
                ROOM_FLOOR_Z_M
                + HOUSEHOLD_OBJECTS[4].support_height_above_floor_m
                - HOUSEHOLD_OBJECTS[4].minimum_xyz[1]
            )
        if args.scene_profile != "living-room":
            _teleport_cup_to(cup_prim, cup_x, cup_y, cup_z, cup_yaw_rad)
        # ---- Pre-position arm to high-reach pose -------------------------
        final_arm = _set_arm_to_high_reach(robot)
        palm_now = link_world_position(robot, RIGHT_PALM_LINK)
        if args.scene_profile == "living-room":
            base_x, base_y, base_yaw, cup_pos = _place_living_room_station_for_physics(
                robot, cup_prim
            )
            cup_x, cup_y, cup_z = (float(value) for value in cup_pos)
            target_world = _world_bbox_center(cup_prim)
            _aim_head_camera_at_vla_target(cam, base_x, base_y, base_yaw, target_world)
        else:
            cup_pos = np.array([cup_x, cup_y, cup_z], dtype=np.float64)
            target_world = _world_bbox_center(cup_prim)
            # Keep the calibrated fixed head optical frame during collection.
            # A target-gaze helper exists for live operation, but it must not
            # be used here until its Isaac camera-axis calibration is verified
            # against captured RGB (the generic look-at conversion can render
            # a black frame at this dining-table heading).

        if args.camera_calibration_only:
            from PIL import Image
            from pxr import Gf, Sdf, UsdShade

            calibration_dir = args.output_dir / "camera_calibration"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            # Temporary high-visibility marker for the exact Expert target.
            # Stronger-than-descendants binding distinguishes Item05 from the
            # other cups/bowls during calibration without editing source USDs.
            marker_material = UsdShade.Material.Define(
                stage, "/World/Looks/Item05CalibrationMarker"
            )
            marker_shader = UsdShade.Shader.Define(
                stage, "/World/Looks/Item05CalibrationMarker/Shader"
            )
            marker_shader.CreateIdAttr("UsdPreviewSurface")
            marker_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(1.0, 0.01, 0.01)
            )
            marker_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.8, 0.0, 0.0)
            )
            marker_material.CreateSurfaceOutput().ConnectToSource(
                marker_shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(cup_prim).Bind(
                marker_material,
                UsdShade.Tokens.strongerThanDescendants,
            )
            _app_utils.update_app(steps=12)
            print(
                f"  Camera calibration target={target_world.tolist()} -> "
                f"{calibration_dir}",
                flush=True,
            )
            for height_m in (1.72,):
                for forward_m in (0.25,):
                    for yaw_offset_deg in (-15.0,):
                        for pitch_deg in (61.0,):
                            _set_camera_calibration_pose(
                                cam,
                                base_x,
                                base_y,
                                base_yaw,
                                height_above_floor_m=height_m,
                                forward_offset_m=forward_m,
                                yaw_offset_deg=yaw_offset_deg,
                                pitch_deg=pitch_deg,
                            )
                            rgb = camera_rgb(cam)
                            if rgb is None:
                                continue
                            name = (
                                f"h{height_m:.2f}_f{forward_m:.2f}_"
                                f"y{yaw_offset_deg:+.0f}_p{pitch_deg:.0f}.png"
                            )
                            Image.fromarray(rgb).save(calibration_dir / name)
            print("  Camera calibration sweep complete", flush=True)
            return 0
        print(
            f"  Ep {episode_id:04d} | "
            f"base=({base_x:.2f},{base_y:.2f},{math.degrees(base_yaw):.0f}°) "
            f"cup=({cup_pos[0]:.2f},{cup_pos[1]:.2f}) "
            f"palm_z={palm_now[2]:.3f} ",
            end="",
            flush=True,
        )

        # ---- Pick instruction --------------------------------------------
        instruction = str(rng.choice(DEFAULT_INSTRUCTIONS))

        # ---- Build capture state -----------------------------------------
        ticks_per_sample = max(1, PHYSICS_HZ // 10)  # 6 ticks at 60 Hz → 10 Hz
        hand_indices = robot.get_dof_indices(
            list(RIGHT_HAND_JOINTS)
        ).numpy().tolist()

        demo: dict[str, Any] = {
            "frame": 0,
            "tick": 0,
            "ticks_per_sample": ticks_per_sample,
            "image": None,
            "audit_image": None,
            "position": None,
            "orientation": None,
            "invalid_images": 0,
            "invalid_action_count": 0,
            "black_frame_fraction": 0.0,
            "bottom_black_fraction": 0.0,
            "hand_indices": hand_indices,
        }
        # Capture frames are written to a pending staging dir first.
        ep_dir = args.output_dir / f"episode_{episode_id:04d}"
        pending_dir = args.output_dir / f".pending_ep_{episode_id:04d}"
        # Clean up any stale artifacts from a previous collection run.
        import shutil
        if ep_dir.is_dir():
            shutil.rmtree(ep_dir)
        if pending_dir.is_dir():
            shutil.rmtree(pending_dir)
        pending_dir.mkdir(parents=True)

        # ---- Build control-step observer ---------------------------------
        def _capture_observer(event: str) -> None:
            nonlocal demo, pending_dir, robot, cam

            if event == "before":
                if demo["tick"] == 0:
                    # Capture head camera RGB
                    image = camera_rgb(cam)
                    audit_image = camera_rgb(audit_cam) if audit_cam is not None else None
                    demo["audit_image"] = (
                        None if audit_image is None else np.asarray(audit_image).copy()
                    )
                    if image is None or image.size == 0:
                        demo["image"] = None
                        demo["invalid_images"] += 1
                    else:
                        rgb = np.asarray(image)
                        if float(rgb.std()) < 2.0:
                            if demo["invalid_images"] == 0:
                                from PIL import Image
                                Image.fromarray(rgb).save(
                                    str(pending_dir / "debug_invalid_first.png")
                                )
                            demo["image"] = None
                            demo["invalid_images"] += 1
                        else:
                            total_black, bottom_black = _black_metrics(rgb)
                            demo["black_frame_fraction"] = max(
                                float(demo["black_frame_fraction"]),
                                float(total_black),
                            )
                            demo["bottom_black_fraction"] = max(
                                float(demo["bottom_black_fraction"]),
                                float(bottom_black),
                            )
                            # Reject individual frames with large black regions
                            if total_black > 0.18 or bottom_black > 0.45:
                                if demo["invalid_images"] == 0:
                                    from PIL import Image
                                    Image.fromarray(rgb).save(
                                        str(pending_dir / "debug_invalid_first.png")
                                    )
                                demo["image"] = None
                                demo["invalid_images"] += 1
                            else:
                                demo["image"] = rgb.copy()
                    # Snapshot palm pose BEFORE the control window
                    demo["position"] = link_world_position(
                        robot, RIGHT_PALM_LINK
                    ).copy()
                    demo["orientation"] = link_world_orientation_xyzw(
                        robot, RIGHT_PALM_LINK
                    ).copy()
                return

            if event != "after":
                raise ValueError(f"unknown Expert recorder event {event!r}")

            # "after" — one physics tick completed
            demo["tick"] += 1
            if demo["tick"] < demo["ticks_per_sample"]:
                return
            demo["tick"] = 0

            if demo["image"] is None:
                return

            # Compute action delta over the completed window
            palm_now = link_world_position(robot, RIGHT_PALM_LINK)
            orientation_now = link_world_orientation_xyzw(robot, RIGHT_PALM_LINK)
            translation = palm_now - demo["position"]
            rotation = _quaternion_rotation_error_xyzw(
                orientation_now, demo["orientation"]
            )

            # Record the expert's commanded aperture, not the contact-limited
            # realised finger positions.
            hand_target_q = (
                robot.get_dof_position_targets()
                .numpy()[0, demo["hand_indices"]]
                .astype(np.float64)
            )
            open_d = float(np.linalg.norm(hand_target_q - RIGHT_HAND_OPEN_RAD))
            closed_d = float(np.linalg.norm(hand_target_q - RIGHT_HAND_CLOSED_RAD))
            gripper = closed_d / max(open_d + closed_d, 1e-9)

            # Flag out-of-bounds actions
            if (
                float(np.max(np.abs(translation))) > 0.08
                or float(np.max(np.abs(rotation))) > 0.8
            ):
                demo["invalid_action_count"] += 1

            # Write step
            step_dir = pending_dir / f"step_{demo['frame']:04d}"
            step_dir.mkdir()
            from PIL import Image
            Image.fromarray(demo["image"]).save(str(step_dir / "image.png"))
            if demo["audit_image"] is not None:
                Image.fromarray(demo["audit_image"]).save(
                    str(step_dir / "third_person.png")
                )

            action = {
                "dx_m": float(translation[0]),
                "dy_m": float(translation[1]),
                "dz_m": float(translation[2]),
                "droll_rad": float(rotation[0]),
                "dpitch_rad": float(rotation[1]),
                "dyaw_rad": float(rotation[2]),
                "gripper": float(np.clip(gripper, 0.0, 1.0)),
                "gripper_label_source": "commanded_dof_position_target",
                "labels": [
                    "dx_m", "dy_m", "dz_m",
                    "droll_rad", "dpitch_rad", "dyaw_rad",
                    "gripper",
                ],
                "unnorm_key": "g1d_family_home_cup_head",
                "frame": "world",
                "window_sim_ticks": int(demo["ticks_per_sample"]),
            }
            (step_dir / "action.json").write_text(
                json.dumps(action, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            demo["frame"] += 1

        # ---- Build advance_fn --------------------------------------------
        def _expert_advance(steps: int) -> None:
            for _ in range(max(int(steps), 0)):
                _app_utils.update_app()

        # ---- Build config overrides --------------------------------------
        # Collect the full current articulation state as initial positions
        all_dof_names = [str(name) for name in robot.dof_names]
        all_dof_positions = robot.get_dof_positions().numpy()[0]
        pregrasp_init: dict[str, float] = {
            name: float(all_dof_positions[index])
            for index, name in enumerate(all_dof_names)
        }
        # Replace right-arm DOFs with the just-achieved high-reach positions
        for i, name in enumerate(RIGHT_ARM_JOINTS):
            pregrasp_init[name] = float(final_arm[i])

        # Keep the unused left arm out of the workspace.  In this asset the
        # zero elbow points the forearm horizontally; +pi/2 makes upper arm
        # and forearm both vertical while the left hand stays relaxed/open.
        for name in all_dof_names:
            if name.startswith("left_"):
                pregrasp_init[name] = 0.0
        left_arm_down = {
            name: float(LEFT_ARM_VERTICAL_RAD[index])
            for index, name in enumerate(LEFT_ARM_JOINTS)
        }
        pregrasp_init.update(left_arm_down)

        config_overrides: dict[str, Any] = {
            # 30 Hz still gives PhysX two settling ticks for every DLS command
            # while cutting the visible expert trajectory duration by a third.
            "control_hz": 30,
            "minimum_table_surface_world_z": ACTIVE_FLOOR_Z_M + 0.55,
            "initial_joint_positions_rad": pregrasp_init,
            "left_arm_down_joint_positions_rad": left_arm_down,
            "ik": {
                "max_joint_step_rad": 0.12,
                "max_position_step_m": 0.025,
                "approach_tilt_min_degrees": 77.5,
                "approach_tilt_max_degrees": 78.5,
            },
            "drive_overrides": {
                # Keep the arm at the pre-close IK solution while finger
                # contact builds.  Effort limits remain those imported from
                # the G1-D URDF; only PD response is made less compliant.
                "arms": {
                    "stiffness": 1200.0,
                    "damping": 120.0,
                    "maximum_effort": 80.0,
                },
                "hands": {
                    "stiffness": 80.0,
                    "damping": 8.0,
                    # The contact-calibrated aperture prevents over-closing;
                    # use enough drive effort to keep the 0.12 kg cup from
                    # opening the thumb/middle pair during vertical lift.
                    # v9: restore the v8 reference value 4.0 N·m.  At 12 N·m
                    # the middle phalanx drives DEEPER into the far wall and
                    # rides over the rim — final middle_0 flex 0.59 rad with
                    # the pad contact 12 mm BELOW the thumb's, a hook shape
                    # whose downward component ejects the cup at the micro-
                    # lift (v5-v7: 0/16, cup slips out after 5-7 mm).  At
                    # 4.0 the wall stops the phalanx earlier (middle_0 0.31,
                    # pads level at ~0.070) — a flat two-sided pinch that
                    # friction-carries the cup (v8 accepted episodes lift
                    # 40-185 mm at default μ ≈ 0.5).
                    "maximum_effort": 4.0,
                },
            },
            "expert": {
                # Enter slightly above the cup centre so the open palm clears
                # the physical cylinder while both fingertips descend along
                # its sidewall.  Reference cup-task value.
                # --- v2 diagnosis (physical_verify_v2, 0/60 accepted) ---
                # The cup's collision box sits on the table and is 111 mm
                # tall (top at world z ≈ 0.134), while the pad reachability
                # floor of this G1-D wrist at the calibrated base pose is
                # ≈ 0.05–0.06 m.  Grasping at centre + 30 mm put the pads at
                # 0.062 m: (a) only marginally above the reachability floor,
                # so the wrist either stalled 2–6 cm short (v2 dominant
                # failure) or landed on the box face where the fold arc is
                # blocked and the phalanx jams against the immovable 79 mm
                # box (v1 dominant failure); (b) the fingers could never fold
                # over the cup because the fold sweeps through the box volume
                # between pad level and the 111 mm rim.
                # v4: the friction regression (3.0/2.5 materials, see
                # _configure_right_hand_physical_friction) was the actual
                # fold blocker — with low friction the phalanx slides up the
                # cup wall and folds over the rim even from 30 mm above the
                # centre (v8 evidence).  Raising the target to 50 mm was
                # WRONG for a different reason: the pinch line lands within
                # ~3 mm of the 111 mm cup's top face, so the thumb closes
                # ABOVE the wall (thumb inner surface z > box top in 7/8 v4
                # episodes) and clamps the middle phalanx instead of the cup
                # — contact senses finger-on-finger, the cup never follows
                # the micro-lift (drift 25–41 mm, lift ≈ 0).
                # v5: restore the v8 reference offset 30 mm — pinch ~24 mm
                # below the rim, thumb on the near wall, middle wrapped over
                # the far wall (v8 accepted pinch z = 0.061; v1's single
                # accepted episode grasped even lower, z = 0.030).
                "grasp_point_z_offset_m": 0.03,
                # This G1-D wrist cannot reach the former 70 mm high staging
                # plane from the calibrated base pose: it stalls 9–10 cm
                # below it before it ever translates toward the cup.  The
                # explicit finger pads remain 20 mm above the cup/table while
                # moving laterally, which is sufficient collision clearance
                # and lets the Cartesian phases actually reach the target.
                "pregrasp_clearance_m": 0.07,
                "pregrasp_table_clearance_m": 0.02,
                "grasp_target_approach_bias_m": 0.0,
                # home_lab's runtime cup is authored directly on the live
                # stage.  Its PhysX wrapper pose differs from the USD xform
                # under this reconstructed-scene transform, so reset it
                # through the same USD xform used for target geometry.
                "object_pose_api": "usd_xform" if args.scene_profile == "living-room" else "xform_prim",
                # -- grasp phase: reference cup-expert values (code defaults).
                # The old overrides broke the close phase three ways, all
                # visible in the grasp telemetry:
                #  * grasp_object_diameter_m released the kinematic staging
                #    while the fingers were still OPEN (the 6.6–8.6 cm
                #    surface-gap band includes the open-hand gap), which
                #    switched orient off mid-preclose and let the wrist roll
                #    the pinch line to 0.44 (thumb onto the table, middle
                #    onto the rim, contact flicker, no attach);
                #  * close_cartesian_servo_until_fraction 0.95 left the
                #    surface-centring servo only 3 steps to correct the
                #    lateral landing (default 0.7 gives it the last 30% of
                #    closure at the default 2x gain);
                # The object must be carried solely by finger collision and
                # friction.  No FixedJoint or other transport attachment is
                # allowed in training demonstrations.
                "grasp_preclose_settle_steps": 8,
                "close_steps": 40,
                "close_ramp_steps": 30,
                "grasp_attach_after_steps": 20,
                "grasp_min_close_fraction": 0.7,
                # The physical finger-contact gate remains mandatory.  The
                # DLS grasp-point proxy is a palm/pad midpoint and can be
                # 3–5 cm from that proxy while both pads still contact the
                # 64 mm cup, so do not reject a valid two-sided contact on
                # this modelling approximation alone.
                "grasp_attach_max_error_m": 0.025,
                # Once both pads touch, keep the already calibrated grasp
                # target fixed.  The previous 2x centre-servo repeatedly
                # chased a pad-centre proxy that is naturally ~3--4 cm away
                # from the cup centre on this hand, pushing the cup sideways
                # before the close phase could establish a force closure.
                "grasp_max_object_drift_m": 0.012,
                # A cup is not safe to close on with the generic 2.5 cm
                # Cartesian tolerance used for transit motions.
                "grasp_entry_position_tolerance_m": 0.015,
                "close_cartesian_servo_until_fraction": 0.7,
                "contact_surface_center_servo": False,
                "contact_surface_center_gain": 0.0,
                "contact_surface_center_max_correction_m": 0.0,
                "contact_target_lead_limit_rad": 0.12,
                # Do not release the cup to dynamics until the hand has
                # reached a firm 95%-closed pinch.  At 80% the contact gate
                # could pass while the cup still rested on the table, then
                # fall out as soon as lifting started.  This remains a pure
                # collision/friction grasp: no fixed joint is ever created.
                "release_kinematic_at_close_fraction": 0.95,
                "use_fixed_joint_attach": False,
                # The pinch line stays near-horizontal (measured 0.03–0.07)
                # while the orientation lock is held through the preclose.
                # Keep modest headroom over the reference 0.10; the 0.44
                # wrist-sag state must never be admitted.
                "line_vertical_tolerance_m": 0.15,
                # At the calibrated reachable pose the wrist converges with
                # ~0.35 rad residual orientation error.  This only admits the
                # next *contact* phase; the two real fingertip contact
                # sensors, lift height, and hold-stability gates remain
                # mandatory.
                "line_alignment_tolerance_rad": 0.40,
                # Slow the loaded lift enough for the physical finger clamp
                # to retain the cup without an artificial attachment.
                "lift_target_lead_limit_rad": 0.12,
                # A loaded arm needs longer than 0.4 seconds to clear the
                # tabletop.  Keep bounded DLS control but allow a useful
                # lift and physical verification window.
                "lift_max_position_step_m": 0.004,
                # Prove a force-closure grasp before committing to the full
                # lift.  This is a physical test: both pads must remain in
                # contact while the cup follows a 2.5 cm micro-lift.
                "grasp_verify_lift_m": 0.025,
                "grasp_verify_min_lift_m": 0.015,
                "grasp_verify_max_position_step_m": 0.002,
                "grasp_verify_max_relative_drift_m": 0.012,
                "grasp_verify_stable_steps": 8,
                "grasp_verify_max_steps": 90,
                # Contact-only lifts can begin more slowly than a transport
                # joint.  Do not reject a real 2 cm initial lift before the
                # fingers have had time to carry the cup through 10 cm.
                "grasp_verify_after_steps": 150,
                "phase_max_steps": {"horizontal": 360, "lift": 360},
                # This dataset is for the VLA ``pick`` skill, not for a
                # pick-and-drop benchmark.  Keep the cup after the verified
                # lift so the quality gate can measure a real 30-frame hold
                # rather than comparing the post-release cup height with its
                # original tabletop height.
                "retain_grasp_after_lift": True,
            },
        }

        # ---- Run expert --------------------------------------------------
        ep_output_dir = args.output_dir / f"expert_run_{episode_id:04d}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)

        from g1d_expert_bridge import run_expert_pick

        try:
            evidence = run_expert_pick(
                robot=robot,
                stage=stage,
                target_prim_path=str(cup_prim.GetPath()),
                table_top_prim_path=ACTIVE_TABLE_PRIM_PATH,
                robot_prim_path=ROBOT_PRIM_PATH,
                palm_prim_path=palm_prim_path,
                arm="right",
                advance_fn=_expert_advance,
                output_dir=ep_output_dir,
                config_overrides=config_overrides,
                control_step_observer=_capture_observer,
            )
        except Exception as exc:
            print(f"\n    ✗ Expert exception: {exc}")
            _rejected = args.output_dir / "rejected"
            _rejected.mkdir(exist_ok=True)
            shutil.move(
                str(pending_dir),
                str(
                    _rejected
                    / f"rejected_ep_{episode_id:04d}_{int(time.time())}"
                ),
            )
            continue

        # ---- Verify evidence ---------------------------------------------
        expert_succeeded = bool(evidence.get("success"))
        lift_height_m = float(evidence.get("lift_height_m", 0.0))
        stable_hold = int(evidence.get("stable_hold_frames", 0))
        physical_hold_verified = bool(
            evidence.get("physical_hold_verified", False)
        )
        fixed_joint_created = bool(evidence.get("fixed_joint_created", False))
        frame_count = int(demo["frame"])
        left_indices = robot.get_dof_indices(list(LEFT_ARM_JOINTS)).numpy().tolist()
        left_actual_rad = (
            robot.get_dof_positions().numpy()[0, left_indices].astype(np.float64)
        )
        left_max_joint_error_rad = float(
            np.max(np.abs(left_actual_rad - LEFT_ARM_VERTICAL_RAD))
        )
        left_shoulder_world = link_world_position(
            robot, "left_shoulder_pitch_link"
        )
        left_elbow_world = link_world_position(robot, "left_elbow_link")
        left_wrist_world = link_world_position(robot, "left_wrist_roll_link")

        def _down_angle_deg(start: np.ndarray, end: np.ndarray) -> float:
            segment = np.asarray(end - start, dtype=np.float64)
            length = float(np.linalg.norm(segment))
            if length <= 1e-9:
                return 180.0
            cosine = float(np.clip(-segment[2] / length, -1.0, 1.0))
            return math.degrees(math.acos(cosine))

        left_upper_down_angle_deg = _down_angle_deg(
            left_shoulder_world, left_elbow_world
        )
        left_forearm_down_angle_deg = _down_angle_deg(
            left_elbow_world, left_wrist_world
        )
        left_arm_vertical_verified = (
            left_max_joint_error_rad <= 0.15
            and left_upper_down_angle_deg <= 25.0
            and left_forearm_down_angle_deg <= 25.0
        )

        accepted = (
            expert_succeeded
            and physical_hold_verified
            and not fixed_joint_created
            and frame_count >= 8
            and lift_height_m >= 0.10
            and stable_hold >= 30
            and demo["invalid_images"] == 0
            and demo["invalid_action_count"] == 0
            and left_arm_vertical_verified
        )

        if accepted:
            # Rename pending to final episode dir
            pending_dir.rename(ep_dir)
            successful += 1
            training_ready_total += 1
            meta_dir = ep_dir
            print(f"✓ ({frame_count} frames, lift={lift_height_m:.3f}m)")
        else:
            reason_parts = []
            if not expert_succeeded:
                reason_parts.append(
                    f"expert_failed: {evidence.get('reason', 'unknown')}"
                )
            if not physical_hold_verified:
                reason_parts.append("physical hold was not verified")
            if fixed_joint_created:
                reason_parts.append("forbidden fixed grasp joint was created")
            if frame_count < 8:
                reason_parts.append(f"only {frame_count} frames")
            if lift_height_m < 0.10:
                reason_parts.append(f"lift={lift_height_m:.3f}m < 0.10")
            if stable_hold < 30:
                reason_parts.append(f"hold={stable_hold} < 30")
            if demo["invalid_images"] > 0:
                reason_parts.append(f"{demo['invalid_images']} invalid images")
            if demo["invalid_action_count"] > 0:
                reason_parts.append(
                    f"{demo['invalid_action_count']} invalid actions"
                )
            if not left_arm_vertical_verified:
                reason_parts.append(
                    "left arm not vertical: "
                    f"upper={left_upper_down_angle_deg:.1f}deg "
                    f"forearm={left_forearm_down_angle_deg:.1f}deg "
                    f"joint_error={left_max_joint_error_rad:.3f}rad"
                )
            reason = "; ".join(reason_parts) if reason_parts else "unknown"
            print(f"✗ ({reason})")

            # Quarantine rejected episode — move pending dir to rejected/
            _rejected = args.output_dir / "rejected"
            _rejected.mkdir(exist_ok=True)
            _rejected_path = (
                _rejected
                / f"rejected_ep_{episode_id:04d}_{int(time.time())}"
            )
            shutil.move(str(pending_dir), str(_rejected_path))
            meta_dir = _rejected_path

        # ---- Write episode meta.json -------------------------------------
        ep_meta = {
            "episode_id": episode_id,
            "instruction": instruction,
            "object_id": (
                "living_room_coffee_cup" if args.scene_profile == "living-room"
                else "dining_cup"
            ),
            "base_pose": {"x": base_x, "y": base_y, "yaw": base_yaw},
            "unused_left_arm_pose": {
                "joint_order": list(LEFT_ARM_JOINTS),
                "target_rad": LEFT_ARM_VERTICAL_RAD.tolist(),
                "actual_rad": left_actual_rad.tolist(),
                "maximum_joint_error_rad": left_max_joint_error_rad,
                "upper_arm_down_angle_deg": left_upper_down_angle_deg,
                "forearm_down_angle_deg": left_forearm_down_angle_deg,
                "vertical_verified": left_arm_vertical_verified,
            },
            "cup_root_world_m": cup_pos.tolist(),
            "cup_grasp_target_world_m": target_world.tolist(),
            "randomization": {
                "robot_base_fixed": True,
                "base_variation_xy_m": 0.0,
                "base_variation_yaw_deg": 0.0,
                "cup_local_forward_offset_m": cup_local_forward_offset_m,
                "cup_local_right_offset_m": cup_local_right_offset_m,
                "cup_yaw_rad": cup_yaw_rad,
                "cup_nominal_yaw_deg": PHYSICAL_CUP_YAW_DEG,
                "cup_yaw_offset_deg": cup_yaw_offset_deg,
                "sampling_corridor": {
                    "forward_offset_m": list(PHYSICAL_SUCCESS_FORWARD_OFFSET_M),
                    "right_offset_m": list(PHYSICAL_SUCCESS_RIGHT_OFFSET_M),
                    "yaw_offset_deg": list(PHYSICAL_SUCCESS_YAW_OFFSET_DEG),
                },
                "fixed_condition": bool(args.fixed_condition),
                "cup_roll_rad": 0.0,
                "cup_pitch_rad": 0.0,
            },
            "success": expert_succeeded,
            "ready_for_training": accepted,
            "frame_count": frame_count,
            "capture_hz": 10,
            "total_black_fraction": float(demo["black_frame_fraction"]),
            "bottom_black_fraction": float(demo["bottom_black_fraction"]),
            "invalid_image_count": int(demo["invalid_images"]),
            "black_frame_count": int(demo["invalid_images"]),
            "invalid_action_count": int(demo["invalid_action_count"]),
            "expert_evidence": {
                "success": bool(evidence.get("success")),
                "physical_execution": bool(
                    evidence.get("physical_execution", True)
                ),
                "lift_height_m": float(lift_height_m),
                "stable_hold_frames": int(stable_hold),
                "hold_contact_frames": int(
                    evidence.get("hold_contact_frames", 0)
                ),
                "hold_max_palm_object_drift_m": float(
                    evidence.get("hold_max_palm_object_drift_m", 0.0)
                ),
                "hold_min_lift_height_m": float(
                    evidence.get("hold_min_lift_height_m", 0.0)
                ),
                "physical_hold_verified": physical_hold_verified,
                "fixed_joint_created": fixed_joint_created,
                "fixed_joint_configured": bool(
                    evidence.get("fixed_joint_configured", False)
                ),
                "state": evidence.get("state"),
                "steps": evidence.get("steps"),
                "grasp_mechanism": evidence.get("grasp_mechanism"),
                "reason": evidence.get("reason"),
                "episode_dir": evidence.get("episode_dir"),
                "object_prim_path": evidence.get("object_prim_path"),
                "table_top_prim_path": evidence.get("table_top_prim_path"),
            },
            "task_contract": f"{args.scene_profile}_pick_lift_hold_head_ego",
            "gripper_convention": "1=open,0=closed",
            "gripper_label_source": "commanded_dof_position_target",
            "camera_mode": "ego_centric_head",
            "camera_mount": {
                "prim_path": "/World/Sensors/G1DHeadCamera",
                "height_above_floor_m": CAMERA_HEIGHT_ABOVE_FLOOR_M,
                "forward_offset_m": CAMERA_FORWARD_OFFSET_M,
                "yaw_offset_deg": math.degrees(CAMERA_YAW_OFFSET_RAD),
                "downward_pitch_deg": args.camera_pitch_deg,
                "shared_for": ["vln_navigation", "vla_manipulation"],
            },
            "camera_intrinsics": {
                "model": "pinhole",
                "resolution": [640, 480],
                "focal_length_mm": CAMERA_FOCAL_LENGTH_MM,
                "horizontal_aperture_mm": CAMERA_HORIZONTAL_APERTURE_MM,
                "vertical_aperture_mm": CAMERA_VERTICAL_APERTURE_MM,
                "fx_px": CAMERA_FX_PX,
                "fy_px": CAMERA_FY_PX,
                "cx_px": CAMERA_CX_PX,
                "cy_px": CAMERA_CY_PX,
                "near_clip_m": CAMERA_NEAR_CLIP_M,
                "far_clip_m": CAMERA_FAR_CLIP_M,
                "distortion": [],
            },
            "rgb_quality_gate_version": 2,
            "expert_source": "machuanhao_dls_ik",
        }
        (meta_dir / "meta.json").write_text(
            json.dumps(ep_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_episodes.append(ep_meta)

        # ---- Clean up for next episode -----------------------------------
        # Remove the expert's stale grasp constraint if it wasn't cleaned up
        constraint_path = "/World/G1DExpertGraspConstraint"
        if stage.GetPrimAtPath(constraint_path).IsValid():
            stage.RemovePrim(constraint_path)
        _app_utils.update_app(steps=10)

    # ── Write manifest ───────────────────────────────────────────────────
    # Rebuild the aggregate list from accepted episode directories so a
    # resumed run appends to, rather than replaces, the dataset inventory.
    aggregate_episodes: list[dict[str, Any]] = []
    for episode_path in sorted(args.output_dir.glob("episode_*/meta.json")):
        try:
            episode_meta = json.loads(
                episode_path.read_text(encoding="utf-8")
            )
            if _meta_is_physical_training_ready(episode_meta):
                aggregate_episodes.append(episode_meta)
        except (OSError, json.JSONDecodeError):
            continue
    manifest = {
        "schema_version": 1,
        "task": f"{args.scene_profile}_cup_grasping_head_ego",
        "robot": "g1_d_wheeled",
        "unused_left_arm_pose": {
            "joint_order": list(LEFT_ARM_JOINTS),
            "target_rad": LEFT_ARM_VERTICAL_RAD.tolist(),
            "contract": "upper_arm_and_forearm_vertical_to_ground",
        },
        "action_space": {
            "type": "continuous_7d_delta",
            "labels": [
                "dx_m", "dy_m", "dz_m",
                "droll_rad", "dpitch_rad", "dyaw_rad",
                "gripper",
            ],
            "unnorm_key": f"g1d_{args.scene_profile.replace('-', '_')}_cup_head",
            "frame": "world",
            "gripper_convention": "1=open,0=closed",
            "gripper_label_source": "commanded_dof_position_target",
        },
        "observation_space": {
            "type": "rgb",
            "resolution": [640, 480],
            "camera": "head_camera_ego_centric",
            "shared_for": ["vln_navigation", "vla_manipulation"],
            "intrinsics": {
                "model": "pinhole",
                "focal_length_mm": CAMERA_FOCAL_LENGTH_MM,
                "fx_px": CAMERA_FX_PX,
                "fy_px": CAMERA_FY_PX,
                "cx_px": CAMERA_CX_PX,
                "cy_px": CAMERA_CY_PX,
                "near_clip_m": CAMERA_NEAR_CLIP_M,
                "far_clip_m": CAMERA_FAR_CLIP_M,
                "distortion": [],
            },
        },
        "expert_source": "machuanhao_dls_ik",
        "capture_hz": 10,
        "randomization": {
            "robot_base_fixed": True,
            "base_variation_xy_m": 0.0,
            "base_variation_yaw_deg": 0.0,
            "cup_variation_xy_m": args.cup_variation_xy_m,
            "cup_variation_yaw_deg": args.cup_variation_yaw_deg,
            "cup_nominal_yaw_deg": PHYSICAL_CUP_YAW_DEG,
            "cup_roll_rad": 0.0,
            "cup_pitch_rad": 0.0,
        },
        "instructions": DEFAULT_INSTRUCTIONS,
        "summary": {
            "collection_attempts_this_run": attempts_this_run,
            "successful_episodes_this_run": successful,
            "total_accepted_episodes": len(aggregate_episodes),
            "training_ready_episodes": sum(
                1 for ep_m in aggregate_episodes
                if _meta_is_physical_training_ready(ep_m)
            ),
            "total_frames": sum(
                ep_m["frame_count"] for ep_m in aggregate_episodes
            ),
        },
        "target_training_ready": args.target_training_ready,
        "episodes": aggregate_episodes,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.oft_manifest is not None:
        # OpenVLA-OFT-format manifest: reuses g1d_openvla_oft_data's strict
        # audit (expert evidence, camera intrinsics, action frame/unnorm key)
        # so anything that would fail downstream OFT loading is caught here.
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from g1d_openvla_oft_data import build_manifest as _build_oft_manifest

        try:
            oft_payload = _build_oft_manifest(args.output_dir, args.oft_manifest)
            print(
                f"OFT manifest: {args.oft_manifest.resolve()} "
                f"({oft_payload['episode_count']} episodes, "
                f"{oft_payload['sample_count']} action-chunk samples)"
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"OFT manifest generation FAILED: {exc}")
            return 2

    print(f"\n{'=' * 60}")
    print(
        f"Done. {successful}/{attempts_this_run} attempts succeeded "
        f"({manifest['summary']['training_ready_episodes']} training-ready)."
    )
    print(f"Total frames: {manifest['summary']['total_frames']}")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"{'=' * 60}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": args.headless,
        "width": 1920,
        "height": 1080,
        "active_gpu": args.active_gpu,
        "multi_gpu": False,
    })

    try:
        exit_code = run_collection(args)
    finally:
        simulation_app.close()
    sys.exit(exit_code)
