r"""Run a constrained semantic-navigation task with G1-D in SimpleRoom.

Examples:

    E:\robot\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat ^
        E:\robot\run_g1d_simple_room_vln.py --command "请带我到沙发旁边"

    E:\robot\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat ^
        E:\robot\run_g1d_simple_room_vln.py --headless --test

The default assisted mode validates the high-level language/map/control chain.
It is not presented as a wheel-contact or localization benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parent
LINGBOT_NAV_SRC = ROOT / "lingbot_semantic_nav/src"
if str(LINGBOT_NAV_SRC) not in sys.path:
    sys.path.insert(0, str(LINGBOT_NAV_SRC))
ROOM_USD = ROOT / "Assets/room/IsaacSim/SimpleRoom_flat.usd"
if not ROOM_USD.is_file():
    ROOM_USD = ROOT / "Assets/room/IsaacSim/SimpleRoom.usd"
ROBOT_USD = ROOT / "Assets/g1_d_robot/g1_d.usd"
SOFA_USD = ROOT / "Assets/room/GenieSim/scenes/iros/SofaTablePlant.usd"
DEFAULT_OUTPUT = ROOT / "outputs/simple_room_vln"
DEFAULT_LINGBOT_MAP = DEFAULT_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_FORMAL_PLACES = DEFAULT_OUTPUT / "places_formal.json"
DEFAULT_HOME_OUTPUT = ROOT / "outputs/family_home_vln"
DEFAULT_HOME_LINGBOT_MAP = DEFAULT_HOME_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_HOME_FORMAL_PLACES = DEFAULT_HOME_OUTPUT / "places_formal.json"
DEFAULT_HOME_FORMAL_OBJECTS = DEFAULT_HOME_OUTPUT / "objects_formal.json"
DEFAULT_OPENVLA_MODEL = ROOT / "checkpoints/openvla-7b"
DEFAULT_OPENVLA_PYTHON = ROOT / "envs/openvla/bin/python"

ROBOT_PRIM_PATH = "/World/G1_D"
LEFT_WHEEL_JOINT = "Left_Wheel_Joint"
RIGHT_WHEEL_JOINT = "Right_Wheel_Joint"
WHEEL_RADIUS_M = 0.0848
WHEEL_BASE_M = 0.4062
PHYSICS_HZ = 60
ROOM_FLOOR_Z_M = -0.7695
ROBOT_ROOT_ON_FLOOR_Z_M = -0.664
CAMERA_HEIGHT_ABOVE_FLOOR_M = 1.34
CAMERA_FORWARD_OFFSET_M = 0.18
# With the declared +X-forward, +Z-up camera frame, a positive rotation about
# local +Y pitches the optical axis toward -Z (the floor).
CAMERA_DOWNWARD_PITCH_RAD = math.radians(25.0)
CAMERA_FOCAL_LENGTH_MM = 16.0
CAMERA_HORIZONTAL_APERTURE_MM = 28.0
OVERVIEW_EYE = (3.25, -2.60, 1.75)
OVERVIEW_TARGET = (-1.25, 0.25, -0.20)
HOME_OVERVIEW_EYE = (6.25, -6.00, 6.20)
HOME_OVERVIEW_TARGET = (0.0, 0.70, -0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-profile",
        choices=("simple-room", "family-home"),
        default="simple-room",
        help="Use the original single room or the multi-zone family-home layout",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", action="store_true", help="Headless end-to-end assertion")
    parser.add_argument("--survey", action="store_true", help="Collect a LingBot-ready RGB survey")
    parser.add_argument("--command", default="请带我到沙发旁边")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map", type=Path, default=DEFAULT_LINGBOT_MAP,
                        help="ROS map.yaml built from aligned LingBot point cloud")
    parser.add_argument("--places", type=Path, default=DEFAULT_FORMAL_PLACES,
                        help="v2 place catalog containing an approved SAM3 docking pose")
    parser.add_argument(
        "--objects",
        type=Path,
        default=DEFAULT_HOME_FORMAL_OBJECTS,
        help="reviewed scan-derived household object catalog",
    )
    parser.add_argument(
        "--dual-agent",
        action="store_true",
        help="run NAVIGATE->live SEARCH_OBJECT->APPROACH_AND_ALIGN->VLA slot in one app",
    )
    parser.add_argument(
        "--family-task",
        action="store_true",
        help=(
            "compile --command as a reviewed go->pick->return family mission; "
            "requires --dual-agent"
        ),
    )
    parser.add_argument(
        "--right-arm-probe",
        action="store_true",
        help=(
            "run a bounded G1-D right-palm Jacobian IK motion probe and write "
            "measured link/joint evidence; simulation only"
        ),
    )
    parser.add_argument(
        "--target-object",
        default="houseplant",
        help="reviewed object ID/label/alias used by --dual-agent",
    )
    parser.add_argument(
        "--live-search-frames",
        type=int,
        default=9,
        help="category-free live RGB views captured by SEARCH_OBJECT",
    )
    parser.add_argument(
        "--openvla",
        action="store_true",
        help=(
            "run a real OpenVLA inference at MANIPULATE; raw Bridge action "
            "remains fail-closed until G1-D IK/collision mapping is available"
        ),
    )
    parser.add_argument(
        "--execute-sim-pick",
        action="store_true",
        help=(
            "after OpenVLA inference, execute the reviewed pick primitive with "
            "bounded G1-D position IK and a simulation-only grasp constraint; "
            "never enables physical robot output"
        ),
    )
    parser.add_argument(
        "--openvla-model",
        type=Path,
        default=DEFAULT_OPENVLA_MODEL,
    )
    parser.add_argument(
        "--openvla-python",
        type=Path,
        default=DEFAULT_OPENVLA_PYTHON,
    )
    parser.add_argument("--openvla-unnorm-key", default="bridge_orig")
    parser.add_argument(
        "--openvla-instruction",
        default="",
        help="optional English manipulation instruction for the base checkpoint",
    )
    parser.add_argument("--openvla-timeout-sec", type=float, default=900.0)
    parser.add_argument("--allow-bootstrap", action="store_true",
                        help="Explicitly allow measured Isaac geometry for demo-only navigation")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--start-hold-seconds", type=float, default=0.0,
                        help="Keep the GUI responsive at the start before navigation")
    parser.add_argument("--arrival-hold-seconds", type=float, default=0.0,
                        help="Keep the GUI responsive after reaching the goal")
    parser.add_argument("--viewport-screenshot", type=Path,
                        help="Save the final third-person Isaac viewport as a PNG")
    parser.add_argument("--record-gif", type=Path,
                        help="Record a third-person overview GIF (works in headless mode)")
    parser.add_argument("--record-fps", type=int, default=10,
                        help="Frame rate of --record-gif (default: 10)")
    parser.add_argument(
        "--live-dir",
        type=Path,
        help="Publish live state JSON and overview-camera JPEGs for a web dashboard",
    )
    parser.add_argument("--live-fps", type=int, default=10)
    parser.add_argument("--live-resolution", default="960x540")
    parser.add_argument("--wheel-physics-only", action="store_true")
    parser.add_argument("--no-camera", action="store_true", help="Disable RGB sensor outside survey mode")
    parser.add_argument("--resolution", default="640x480", help="RGB WIDTHxHEIGHT")
    return parser.parse_args()


args = parse_args()
if args.live_dir is not None and not args.live_dir.is_absolute():
    args.live_dir = ROOT / args.live_dir
for openvla_path_name in ("openvla_model", "openvla_python"):
    openvla_path = getattr(args, openvla_path_name)
    if not openvla_path.is_absolute():
        setattr(args, openvla_path_name, ROOT / openvla_path)
if args.scene_profile == "family-home":
    if args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_HOME_OUTPUT
    if args.map == DEFAULT_LINGBOT_MAP:
        args.map = DEFAULT_HOME_LINGBOT_MAP
    if args.places == DEFAULT_FORMAL_PLACES:
        args.places = DEFAULT_HOME_FORMAL_PLACES
    if args.objects == DEFAULT_HOME_FORMAL_OBJECTS:
        args.objects = DEFAULT_HOME_FORMAL_OBJECTS
if args.test:
    args.headless = True
    if args.steps <= 0:
        args.steps = 1800
if args.survey:
    args.no_camera = False
    if args.steps <= 0:
        args.steps = 6000
if args.dual_agent:
    args.no_camera = False
if args.openvla and not args.dual_agent:
    raise SystemExit("--openvla requires --dual-agent")
if args.execute_sim_pick and not args.openvla:
    raise SystemExit("--execute-sim-pick requires --openvla")
if args.family_task and not args.dual_agent:
    raise SystemExit("--family-task requires --dual-agent")
if args.right_arm_probe and args.dual_agent:
    raise SystemExit("--right-arm-probe cannot be combined with --dual-agent")
if args.openvla_timeout_sec <= 0.0:
    raise SystemExit("--openvla-timeout-sec must be positive")

try:
    camera_width, camera_height = (int(item) for item in args.resolution.lower().split("x", 1))
except (TypeError, ValueError) as exc:
    raise SystemExit("--resolution must be WIDTHxHEIGHT") from exc
try:
    live_width, live_height = (
        int(item) for item in args.live_resolution.lower().split("x", 1)
    )
except (TypeError, ValueError) as exc:
    raise SystemExit("--live-resolution must be WIDTHxHEIGHT") from exc
if not 1 <= args.live_fps <= 30:
    raise SystemExit("--live-fps must be between 1 and 30")
if live_width <= 0 or live_height <= 0:
    raise SystemExit("--live-resolution dimensions must be positive")
if args.live_search_frames < 3 or args.live_search_frames > 24:
    raise SystemExit("--live-search-frames must be between 3 and 24")
if args.dual_agent and args.scene_profile != "family-home":
    raise SystemExit("--dual-agent currently requires --scene-profile family-home")

for required in (ROOM_USD, ROBOT_USD, SOFA_USD):
    if not required.is_file():
        raise FileNotFoundError(required)
if args.scene_profile == "family-home":
    from family_home_vln.household_objects import require_prepared_assets

    require_prepared_assets()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1440,
        "height": 900,
    }
)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from family_home_vln.layout import (
    HOME_FIXTURES,
    SCENE_NAME as FAMILY_HOME_SCENE_NAME,
    START_POSE as FAMILY_HOME_START,
    build_bootstrap_artifacts as build_family_home_bootstrap_artifacts,
    build_survey_path as build_family_home_survey_path,
)
from family_home_vln.household_objects import (
    HOUSEHOLD_OBJECTS,
    OBJECT_SET_SIGNATURE,
)
from family_home_vln.formal_mapping import plan_object_approach
from family_home_vln.live_object_search import load_reviewed_object
from hospital_vln.live import LivePublisher, publish_failure
from simple_room_vln.artifacts import (
    SOFA_SET_TRANSLATION,
    build_bootstrap_artifacts,
    load_lingbot_artifacts,
)
from simple_room_vln.core import PathFollower, Pose2D, path_length, resolve_place


def command_to_wheel_velocities(linear_speed: float, angular_speed: float) -> np.ndarray:
    left = (linear_speed - angular_speed * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
    right = -(linear_speed + angular_speed * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
    return np.array([left, right], dtype=np.float32)


def configure_joint_drives(robot: WheeledRobot) -> None:
    names = robot.dof_names
    stiffness = np.zeros(len(names), dtype=np.float32)
    damping = np.zeros(len(names), dtype=np.float32)
    for index, name in enumerate(names):
        if name in (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT):
            damping[index] = 20.0
        elif name in ("LZ_mt_Joint", "LZ_it_Joint"):
            stiffness[index] = 2000.0
            damping[index] = 150.0
        elif "hand_" in name:
            stiffness[index] = 10.0
            damping[index] = 1.0
        else:
            stiffness[index] = 80.0
            damping[index] = 8.0
    robot.set_dof_gains(stiffnesses=stiffness, dampings=damping)
    wheel_indices = robot.get_dof_indices([LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]).numpy().tolist()
    robot.set_dof_max_efforts([40.0, 40.0], dof_indices=wheel_indices)
    robot.set_dof_position_targets(robot.get_dof_positions().numpy()[0])


def assisted_step(pose: Pose2D, linear: float, angular: float) -> Pose2D:
    dt = 1.0 / PHYSICS_HZ
    yaw = pose.yaw + angular * dt
    return Pose2D(
        pose.x + linear * math.cos(yaw) * dt,
        pose.y + linear * math.sin(yaw) * dt,
        yaw,
    )


def set_assisted_robot_pose(robot: WheeledRobot, pose: Pose2D, linear: float, angular: float) -> None:
    orientation = np.array(
        [math.cos(pose.yaw / 2.0), 0.0, 0.0, math.sin(pose.yaw / 2.0)],
        dtype=np.float32,
    )
    robot.set_world_poses(
        positions=np.array([pose.x, pose.y, ROBOT_ROOT_ON_FLOOR_Z_M], dtype=np.float32),
        orientations=orientation,
    )
    robot.set_velocities(
        linear_velocities=[linear * math.cos(pose.yaw), linear * math.sin(pose.yaw), 0.0],
        angular_velocities=[0.0, 0.0, angular],
    )


def robot_pose(robot: WheeledRobot) -> Pose2D:
    positions, orientations = robot.get_world_poses()
    position = positions.numpy()[0]
    quaternion = orientations.numpy()[0]
    yaw = math.atan2(
        2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
        1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
    )
    return Pose2D(float(position[0]), float(position[1]), float(yaw))


RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RIGHT_ARM_LIMITS_RAD = np.asarray(
    [
        (-3.0892, 2.6704),
        (-2.2515, 1.5882),
        (-2.6180, 2.6180),
        (-1.0472, 2.0944),
        (-1.972222054, 1.972222054),
        (-1.614429558, 1.614429558),
        (-1.614429558, 1.614429558),
    ],
    dtype=np.float64,
)
RIGHT_PALM_LINK = "right_hand_palm_link"
RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
RIGHT_HAND_OPEN_RAD = np.asarray(
    [0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08],
    dtype=np.float64,
)
RIGHT_HAND_CLOSED_RAD = np.asarray(
    [0.65, 0.45, -1.25, 1.20, 1.35, 1.20, 1.35],
    dtype=np.float64,
)
# URDF palm-to-fingertip envelope: 0.0777 + 0.0458 + 0.0263 m.
RIGHT_HAND_FINGERTIP_REACH_M = 0.16


def articulation_link_transforms(robot: WheeledRobot) -> np.ndarray:
    """Read physics link transforms as xyz + xyzw quaternion."""

    view = robot._physics_articulation_view
    if view is None:
        raise RuntimeError("G1-D physics articulation view is not initialized")
    return view.get_link_transforms().numpy()


def link_world_position(robot: WheeledRobot, link_name: str) -> np.ndarray:
    link_index = int(robot.get_link_indices(link_name).numpy()[0])
    transforms = articulation_link_transforms(robot)
    return np.asarray(transforms[0, link_index, :3], dtype=np.float64)


def _jacobian_dof_columns(
    robot: WheeledRobot,
    jacobian: np.ndarray,
    dof_indices: Sequence[int],
) -> list[int]:
    extra = int(jacobian.shape[-1]) - int(robot.num_dofs)
    if extra not in (0, 6):
        raise RuntimeError(
            f"unexpected G1-D Jacobian columns: {jacobian.shape[-1]} "
            f"for {robot.num_dofs} DOFs"
        )
    return [extra + int(index) for index in dof_indices]


def _jacobian_link_row(
    robot: WheeledRobot,
    jacobian: np.ndarray,
    link_name: str,
) -> int:
    link_index = int(robot.get_link_indices(link_name).numpy()[0])
    if jacobian.shape[0] == len(robot.link_names):
        return link_index
    if jacobian.shape[0] == len(robot.link_names) - 1 and link_index > 0:
        return link_index - 1
    raise RuntimeError(
        f"cannot map link {link_name} index {link_index} to "
        f"Jacobian shape {jacobian.shape}"
    )


def run_right_arm_position_probe(
    robot: WheeledRobot,
    *,
    output_path: Path,
    target_offset_world_m: Sequence[float] = (0.06, 0.0, 0.05),
) -> dict:
    """Move the right palm by a small Cartesian offset with bounded DLS IK."""

    print("Arm probe: resolving joint indices", flush=True)
    arm_indices = (
        robot.get_dof_indices(list(RIGHT_ARM_JOINTS)).numpy().tolist()
    )
    limits = RIGHT_ARM_LIMITS_RAD.copy()
    print("Arm probe: reading joint positions", flush=True)
    start_joints = robot.get_dof_positions().numpy()[0, arm_indices].astype(
        np.float64
    )
    print("Arm probe: reading palm transform", flush=True)
    start = link_world_position(robot, RIGHT_PALM_LINK)
    target = start + np.asarray(target_offset_world_m, dtype=np.float64)
    target_norm = float(np.linalg.norm(target - start))
    if target_norm > 0.10:
        raise ValueError("right-arm probe offset must not exceed 0.10 m")

    targets = start_joints.copy()
    errors: list[float] = []
    converged = False
    jacobian_shape: list[int] = []
    for _iteration in range(80):
        current = link_world_position(robot, RIGHT_PALM_LINK)
        error = target - current
        error_norm = float(np.linalg.norm(error))
        errors.append(error_norm)
        if error_norm <= 0.012:
            converged = True
            break
        jacobian = robot.get_jacobian_matrices().numpy()[0]
        jacobian_shape = list(jacobian.shape)
        row = _jacobian_link_row(robot, jacobian, RIGHT_PALM_LINK)
        columns = _jacobian_dof_columns(robot, jacobian, arm_indices)
        position_jacobian = np.asarray(
            jacobian[row, :3, :][:, columns],
            dtype=np.float64,
        )
        damping = 0.04
        delta = position_jacobian.T @ np.linalg.solve(
            position_jacobian @ position_jacobian.T
            + (damping**2) * np.eye(3),
            error,
        )
        delta = np.clip(delta, -0.035, 0.035)
        targets = np.clip(
            targets + delta,
            limits[:, 0] + 0.02,
            limits[:, 1] - 0.02,
        )
        robot.set_dof_position_targets(targets, dof_indices=arm_indices)
        app_utils.update_app(steps=4)

    app_utils.update_app(steps=30)
    final = link_world_position(robot, RIGHT_PALM_LINK)
    final_joints = robot.get_dof_positions().numpy()[0, arm_indices]
    final_error = float(np.linalg.norm(target - final))
    maximum_joint_delta = float(
        np.max(np.abs(final_joints - start_joints))
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "g1d_right_arm_position_ik_probe",
        "success": bool(
            converged
            and final_error <= 0.02
            and maximum_joint_delta <= 0.8
        ),
        "simulation_only": True,
        "controller": "damped_least_squares_position_ik",
        "joint_order": list(RIGHT_ARM_JOINTS),
        "jacobian_shape": jacobian_shape,
        "start_palm_world_m": start.tolist(),
        "target_palm_world_m": target.tolist(),
        "final_palm_world_m": final.tolist(),
        "target_offset_world_m": list(target_offset_world_m),
        "final_position_error_m": final_error,
        "start_joint_position_rad": start_joints.tolist(),
        "final_joint_position_rad": final_joints.tolist(),
        "maximum_joint_delta_rad": maximum_joint_delta,
        "iterations": len(errors),
        "minimum_iteration_error_m": min(errors) if errors else None,
        "joint_limits_rad": limits.tolist(),
        "safety": {
            "maximum_requested_cartesian_offset_m": 0.10,
            "maximum_per_iteration_joint_delta_rad": 0.035,
            "collision_checked": False,
            "object_contact": False,
            "hardware_output": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def move_right_palm_to(
    robot: WheeledRobot,
    target_world_m: Sequence[float],
    *,
    maximum_cartesian_travel_m: float = 0.70,
    tolerance_m: float = 0.025,
    maximum_iterations: int = 180,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Move the right palm with bounded position-only DLS IK."""

    target = np.asarray(target_world_m, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("right-palm target must be a finite xyz vector")
    arm_indices = robot.get_dof_indices(list(RIGHT_ARM_JOINTS)).numpy().tolist()
    start = link_world_position(robot, RIGHT_PALM_LINK)
    requested_travel = float(np.linalg.norm(target - start))
    if requested_travel > maximum_cartesian_travel_m:
        return {
            "success": False,
            "reason": "cartesian_target_outside_bounded_workspace",
            "requested_travel_m": requested_travel,
            "maximum_cartesian_travel_m": maximum_cartesian_travel_m,
        }
    targets = robot.get_dof_positions().numpy()[0, arm_indices].astype(
        np.float64
    )
    errors: list[float] = []
    maximum_step = 0.0
    for _iteration in range(maximum_iterations):
        current = link_world_position(robot, RIGHT_PALM_LINK)
        error = target - current
        error_norm = float(np.linalg.norm(error))
        errors.append(error_norm)
        if error_norm <= tolerance_m:
            break
        jacobian = robot.get_jacobian_matrices().numpy()[0]
        row = _jacobian_link_row(robot, jacobian, RIGHT_PALM_LINK)
        columns = _jacobian_dof_columns(robot, jacobian, arm_indices)
        position_jacobian = np.asarray(
            jacobian[row, :3, :][:, columns],
            dtype=np.float64,
        )
        damping = 0.055
        delta = position_jacobian.T @ np.linalg.solve(
            position_jacobian @ position_jacobian.T
            + (damping**2) * np.eye(3),
            error,
        )
        delta = np.clip(delta, -0.025, 0.025)
        maximum_step = max(maximum_step, float(np.max(np.abs(delta))))
        targets = np.clip(
            targets + delta,
            RIGHT_ARM_LIMITS_RAD[:, 0] + 0.03,
            RIGHT_ARM_LIMITS_RAD[:, 1] - 0.03,
        )
        robot.set_dof_position_targets(targets, dof_indices=arm_indices)
        for _ in range(3):
            simulation_app.update()
            if progress_callback is not None:
                progress_callback(_iteration + 1, maximum_iterations)
    final = link_world_position(robot, RIGHT_PALM_LINK)
    final_error = float(np.linalg.norm(target - final))
    return {
        "success": final_error <= tolerance_m,
        "controller": "bounded_damped_least_squares_position_ik",
        "target_world_m": target.tolist(),
        "start_world_m": start.tolist(),
        "final_world_m": final.tolist(),
        "requested_travel_m": requested_travel,
        "final_error_m": final_error,
        "iterations": len(errors),
        "minimum_iteration_error_m": min(errors) if errors else None,
        "maximum_joint_step_rad": maximum_step,
        "joint_limits_checked": True,
        "orientation_controlled": False,
        "scene_collision_query": False,
    }


def _set_right_hand(
    robot: WheeledRobot,
    targets_rad: np.ndarray,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    indices = robot.get_dof_indices(list(RIGHT_HAND_JOINTS)).numpy().tolist()
    robot.set_dof_position_targets(targets_rad, dof_indices=indices)
    for step in range(45):
        simulation_app.update()
        if progress_callback is not None:
            progress_callback(step + 1, 45)
    actual = robot.get_dof_positions().numpy()[0, indices]
    return {
        "joint_order": list(RIGHT_HAND_JOINTS),
        "target_rad": targets_rad.tolist(),
        "actual_rad": np.asarray(actual, dtype=np.float64).tolist(),
        "maximum_error_rad": float(
            np.max(np.abs(np.asarray(actual) - targets_rad))
        ),
    }


def _prim_world_position(prim) -> np.ndarray:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return np.asarray(
        [translation[0], translation[1], translation[2]],
        dtype=np.float64,
    )


def _find_sim_grasp_bodies(
    target_world_m: np.ndarray,
    *,
    maximum_object_anchor_error_m: float = 0.18,
) -> tuple[object, object, dict]:
    """Find physics bodies by proximity only; no semantic prim name is read."""

    stage = stage_utils.get_current_stage()
    palm = None
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            prim.GetName() == RIGHT_PALM_LINK
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            palm = prim
        if (
            path.startswith("/World/FamilyHomeObjects/")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            position = _prim_world_position(prim)
            candidates.append(
                (
                    float(np.linalg.norm(position - target_world_m)),
                    prim,
                    position,
                )
            )
    if palm is None:
        raise RuntimeError("cannot find the G1-D right-palm rigid body")
    if not candidates:
        raise RuntimeError("no dynamic family object is available for picking")
    candidates.sort(key=lambda item: item[0])
    anchor_error, object_prim, object_position = candidates[0]
    if anchor_error > maximum_object_anchor_error_m:
        raise RuntimeError(
            "nearest dynamic object's physics pose disagrees with the "
            f"scan-derived anchor by {anchor_error:.3f} m"
        )
    return palm, object_prim, {
        "selection": "nearest_dynamic_body_to_scan_anchor_for_safety_verification",
        "palm_prim_path": str(palm.GetPath()),
        "object_prim_path": str(object_prim.GetPath()),
        "scan_anchor_world_m": target_world_m.tolist(),
        "physics_object_world_m": object_position.tolist(),
        "anchor_error_m": anchor_error,
        "maximum_anchor_error_m": maximum_object_anchor_error_m,
        "simulator_semantic_label_read": False,
    }


def _create_sim_grasp_constraint(palm_prim, object_prim, object_id: str) -> str:
    """Attach a nearby object with an explicitly declared simulation joint."""

    stage = stage_utils.get_current_stage()
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in object_id
    )
    path = f"/World/G1DSimGrasp/{safe_name}"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([palm_prim.GetPath()])
    joint.CreateBody1Rel().SetTargets([object_prim.GetPath()])
    palm_world = UsdGeom.Xformable(palm_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    object_world = UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    palm_position = palm_world.ExtractTranslation()
    object_local_anchor = object_world.GetInverse().Transform(palm_position)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(
            float(object_local_anchor[0]),
            float(object_local_anchor[1]),
            float(object_local_anchor[2]),
        )
    )
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    relative_rotation = (
        object_world.ExtractRotationQuat().GetInverse()
        * palm_world.ExtractRotationQuat()
    )
    relative_imaginary = relative_rotation.GetImaginary()
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(
            float(relative_rotation.GetReal()),
            float(relative_imaginary[0]),
            float(relative_imaginary[1]),
            float(relative_imaginary[2]),
        )
    )
    joint.CreateBreakForceAttr(float("inf"))
    joint.CreateBreakTorqueAttr(float("inf"))
    UsdPhysics.RigidBodyAPI(object_prim).GetKinematicEnabledAttr().Set(False)
    app_utils.update_app(steps=10)
    return path


def camera_world_pose(
    pose: Pose2D,
    downward_pitch_rad: float = CAMERA_DOWNWARD_PITCH_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(
        [
            pose.x + CAMERA_FORWARD_OFFSET_M * math.cos(pose.yaw),
            pose.y + CAMERA_FORWARD_OFFSET_M * math.sin(pose.yaw),
            ROOM_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M,
        ],
        dtype=np.float32,
    )
    cy = math.cos(pose.yaw / 2.0)
    sy = math.sin(pose.yaw / 2.0)
    cp = math.cos(downward_pitch_rad / 2.0)
    sp = math.sin(downward_pitch_rad / 2.0)
    # world_Z(yaw) * local_Y(pitch); positive Y pitch points +X down.
    orientation = np.array(
        [cy * cp, -sy * sp, cy * sp, sy * cp],
        dtype=np.float32,
    )
    return position, orientation


def look_at_camera_pose(
    eye: Sequence[float], target: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a +X-forward, +Z-up camera pose looking at *target*."""

    eye_array = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_array
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation = np.column_stack((forward, left, up))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float32,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = math.sqrt(1.0 + rotation[index, index] - rotation[next_index, next_index] - rotation[last_index, last_index]) * 2.0
        quaternion = np.zeros(4, dtype=np.float32)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
        quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        quaternion[last_index + 1] = (rotation[last_index, index] + rotation[index, last_index]) / scale
    return eye_array.astype(np.float32), quaternion


def home_chase_camera_pose(pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
    """Follow G1-D from above the walls so household partitions do not occlude it."""

    eye = (
        max(-3.60, min(3.60, pose.x + 2.25)),
        max(-2.40, min(3.85, pose.y - 2.25)),
        ROOM_FLOOR_Z_M + 4.40,
    )
    target = (
        pose.x,
        pose.y,
        ROOM_FLOOR_Z_M + 0.45,
    )
    return look_at_camera_pose(eye, target)


def save_camera_rgb(camera, path: Path) -> bool:
    image = camera_rgb(camera)
    if image is None:
        return False
    from PIL import Image

    Image.fromarray(image).save(path)
    return True


def camera_rgb(camera):
    rgba = camera.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    image = np.asarray(rgba)[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(
            image * (255.0 if image.max() <= 1.0 else 1.0),
            0,
            255,
        ).astype(np.uint8)
    return image


def add_composed_scene(
    path: Sequence[tuple[float, float]], target: Pose2D | None
) -> None:
    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage_utils.add_reference_to_stage(str(ROOM_USD).replace("\\", "/"), "/World/Room")
    stage_utils.add_reference_to_stage(str(SOFA_USD).replace("\\", "/"), "/World/SofaSet")
    stage = stage_utils.get_current_stage()
    light = UsdLux.DomeLight.Define(stage, "/World/VLN/DomeLight")
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    sofa = stage.GetPrimAtPath("/World/SofaSet")
    UsdGeom.Xformable(sofa).AddTranslateOp().Set(Gf.Vec3d(*SOFA_SET_TRANSLATION))
    if args.scene_profile == "family-home":
        fixtures_root = UsdGeom.Xform.Define(stage, "/World/FamilyHome")
        fixtures_root.GetPrim().CreateAttribute(
            "scene:profile",
            Sdf.ValueTypeNames.String,
        ).Set("family-home")
        for fixture in HOME_FIXTURES:
            cube = UsdGeom.Cube.Define(
                stage,
                f"/World/FamilyHome/{fixture.fixture_id}",
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
            cube.GetPrim().CreateAttribute(
                "semantic:category",
                Sdf.ValueTypeNames.String,
            ).Set(fixture.category)
        objects_root = UsdGeom.Xform.Define(stage, "/World/FamilyHomeObjects")
        objects_root.GetPrim().CreateAttribute(
            "perception:rgbOnly",
            Sdf.ValueTypeNames.Bool,
        ).Set(True)
        objects_root.GetPrim().CreateAttribute(
            "perception:categoryLabelsSupplied",
            Sdf.ValueTypeNames.Bool,
        ).Set(False)
        for index, item in enumerate(HOUSEHOLD_OBJECTS, start=1):
            # Opaque prim names ensure simulator truth is not exposed as a
            # semantic category. The discovery model receives rendered RGB only.
            root = UsdGeom.Xform.Define(
                stage,
                f"/World/FamilyHomeObjects/Item{index:02d}",
            )
            source_min_y = item.minimum_xyz[1]
            root_transform = UsdGeom.Xformable(root)
            root_transform.AddTranslateOp().Set(
                Gf.Vec3d(
                    item.position_xy[0],
                    item.position_xy[1],
                    ROOM_FLOOR_Z_M
                    + item.support_height_above_floor_m
                    - source_min_y,
                )
            )
            root_transform.AddRotateZOp().Set(item.yaw_deg)
            asset_frame = UsdGeom.Xform.Define(
                stage,
                f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame",
            )
            UsdGeom.Xformable(asset_frame).AddRotateXOp().Set(90.0)
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
            center = (minimum + maximum) / 2.0
            size = maximum - minimum
            collision.CreateSizeAttr(1.0)
            collision_transform = UsdGeom.Xformable(collision)
            collision_transform.AddTranslateOp().Set(Gf.Vec3d(*center))
            collision_transform.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
            if item.dynamic:
                rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
                rigid_body.CreateRigidBodyEnabledAttr(True)
                # Portable props remain stable throughout long base navigation.
                # The grasp backend releases kinematic staging only after the
                # hand closes and the explicit PhysX grasp joint is authored.
                rigid_body.CreateKinematicEnabledAttr(True)
                mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
                mass.CreateMassAttr(float(item.mass_kg))
    if target is not None:
        marker = UsdGeom.Cylinder.Define(stage, "/World/VLN/Goal")
        marker.CreateAxisAttr("Z")
        marker.CreateRadiusAttr(0.16)
        marker.CreateHeightAttr(0.035)
        marker.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.05, 0.05)])
        UsdGeom.Xformable(marker).AddTranslateOp().Set(
            Gf.Vec3d(target.x, target.y, ROOM_FLOOR_Z_M + 0.025)
        )
    route = UsdGeom.BasisCurves.Define(stage, "/World/VLN/PlannedPath")
    route.CreateTypeAttr("linear")
    route.CreateCurveVertexCountsAttr([len(path)])
    route.CreatePointsAttr(
        [Gf.Vec3f(x, y, ROOM_FLOOR_Z_M + 0.045) for x, y in path]
    )
    route.CreateWidthsAttr([0.045])
    route.SetWidthsInterpolation("constant")
    route.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])


def build_simple_survey_path(grid) -> list[tuple[float, float]]:
    waypoints = [
        (0.0, 0.0),
        (0.0, -2.20),
        (-3.20, -2.20),
        (-3.45, -0.45),
        (-1.10, 0.10),
        (0.00, 3.35),
        (2.20, 3.35),
        (2.15, 1.75),
        (1.20, 0.00),
        (0.0, 0.0),
    ]
    combined = [waypoints[0]]
    for start, goal in zip(waypoints, waypoints[1:]):
        segment = grid.plan(start, goal)
        combined.extend(segment[1:])
    return combined


class SurveyRecorder:
    def __init__(self, output_dir: Path, camera, intrinsics: np.ndarray) -> None:
        self.root = output_dir / "survey"
        self.rgb = self.root / "rgb"
        self.rgb.mkdir(parents=True, exist_ok=True)
        # A survey directory is one immutable sequence. Remove only frames
        # generated by an earlier run so manifest and RGB count cannot diverge.
        stale_manifest = self.root / "capture_manifest.json"
        if stale_manifest.is_file():
            stale_manifest.unlink()
        for stale_frame in self.rgb.glob("*.png"):
            stale_frame.unlink()
        self.camera = camera
        self.intrinsics = np.asarray(intrinsics).tolist()
        self.frames: list[dict] = []
        self.last_pose: Pose2D | None = None

    def maybe_capture(self, pose: Pose2D, *, force: bool = False) -> bool:
        if self.last_pose is not None and not force:
            distance = math.dist((pose.x, pose.y), (self.last_pose.x, self.last_pose.y))
            angle = abs(math.atan2(math.sin(pose.yaw - self.last_pose.yaw), math.cos(pose.yaw - self.last_pose.yaw)))
            if distance < 0.12 and angle < math.radians(10.0):
                return False
        index = len(self.frames)
        image_name = f"{index:06d}.png"
        if not save_camera_rgb(self.camera, self.rgb / image_name):
            return False
        camera_position, camera_orientation = camera_world_pose(pose)
        self.frames.append(
            {
                "frame": index,
                "image": f"rgb/{image_name}",
                "robot_pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
                "camera_pose": {
                    "position": camera_position.tolist(),
                    "orientation_wxyz": camera_orientation.tolist(),
                },
                "consumer": "lingbot_rgb_only",
            }
        )
        self.last_pose = pose
        return True

    def finish(self) -> Path:
        manifest = {
            "schema_version": 1,
            "scene": (
                FAMILY_HOME_SCENE_NAME
                if args.scene_profile == "family-home"
                else "SimpleRoom+SofaTablePlant"
            ),
            "rgb_is_only_model_input": True,
            "pose_consumer": "offline_metric_alignment_and_evaluation_only",
            "household_object_set_signature": (
                OBJECT_SET_SIGNATURE
                if args.scene_profile == "family-home"
                else None
            ),
            "object_category_labels_supplied_to_perception": False,
            "camera": {
                "resolution": [camera_width, camera_height],
                "intrinsics": self.intrinsics,
                "horizontal_fov_deg": math.degrees(
                    2.0 * math.atan(CAMERA_HORIZONTAL_APERTURE_MM / (2.0 * CAMERA_FOCAL_LENGTH_MM))
                ),
                "height_above_floor_m": CAMERA_HEIGHT_ABOVE_FLOOR_M,
                "axes": "+X forward, +Z up",
            },
            "frames": self.frames,
        }
        path = self.root / "capture_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class FamilyHomeDualAgentSession:
    """In-process skill backend that keeps one Isaac SimulationApp alive."""

    def __init__(
        self, robot, camera, grid, places, output_dir: Path,
        *, live=None, overview_camera=None,
    ) -> None:
        self.robot = robot
        self.camera = camera
        self.grid = grid
        self.places = places
        self.output_dir = output_dir
        self.pose = robot_pose(robot)
        self.application_id = f"isaac-sim-{os.getpid()}"
        self.segments: list[dict] = []
        self.last_manipulation_evidence: dict = {}
        self.carried_physics_prim_path = ""
        self.carried_reference_palm_distance_m: float | None = None
        self.manipulation_camera_pitch_rad = CAMERA_DOWNWARD_PITCH_RAD
        self.last_handoff_image_path = ""
        self.last_handoff_gate: dict = {}
        self.search_handoff_cache: dict[str, dict] = {}
        self.live = live
        self.overview_camera = overview_camera
        self.live_frame = 0

    def _manipulation_progress(self, phase: str, step: int, total: int) -> None:
        self._publish_live(
            "OPENVLA_PICK",
            f"OPENVLA_PICK：{phase} {step}/{total}",
            waypoint=step,
            waypoint_count=total,
        )

    def _rotate_in_place(self, target_yaw: float, action: str) -> None:
        """Continuously rotate the base; never teleport its heading for RGB scans."""

        for step in range(240):
            error = math.atan2(
                math.sin(target_yaw - self.pose.yaw),
                math.cos(target_yaw - self.pose.yaw),
            )
            if abs(error) <= 0.025:
                break
            angular = max(-0.55, min(0.55, 2.0 * error))
            self.robot.apply_wheel_actions(
                command_to_wheel_velocities(0.0, angular)
            )
            if not args.wheel_physics_only:
                self.pose = assisted_step(self.pose, 0.0, angular)
                set_assisted_robot_pose(self.robot, self.pose, 0.0, angular)
            if self.camera is not None:
                self.camera.set_world_pose(
                    *camera_world_pose(self.pose), camera_axes="world"
                )
            simulation_app.update()
            self._publish_live(
                action,
                f"{action}：底盘原地正向转向扫描 {step + 1}/240",
                angular=angular,
            )
        self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        self.pose = (
            robot_pose(self.robot) if args.wheel_physics_only else self.pose
        )

    def _publish_live(
        self, action: str, message: str, *, linear: float = 0.0,
        angular: float = 0.0, waypoint: int = 0, waypoint_count: int = 0,
        force: bool = False, result: dict | None = None,
    ) -> None:
        if self.live is None:
            return
        due = force or self.live_frame % max(1, PHYSICS_HZ // args.live_fps) == 0
        if self.overview_camera is not None:
            self.overview_camera.set_world_pose(
                *home_chase_camera_pose(self.pose), camera_axes="world"
            )
        if due:
            source = self.overview_camera or self.camera
            image = camera_rgb(source) if source is not None else None
            if image is not None:
                self.live.publish_image(image)
            self.live.publish_state(
                state="running", message=message, frame=self.live_frame,
                action=action, pose=self.pose, linear=linear, angular=angular,
                waypoint=waypoint, waypoint_count=waypoint_count, result=result,
            )
        self.live_frame += 1

    def _drive(
        self,
        path: list[tuple[float, float]],
        goal_yaw: float,
        *,
        precision: bool = False,
        phase: str = "NAVIGATE",
    ) -> dict:
        follower = PathFollower(
            path,
            goal_yaw=goal_yaw,
            max_linear=0.24 if precision else 0.35,
            max_angular=0.7 if precision else 0.9,
            position_tolerance=0.03 if precision else 0.12,
            yaw_tolerance=0.05 if precision else 0.12,
            waypoint_tolerance=0.12 if precision else 0.18,
        )
        frame = 0
        reverse_motion_frames = 0
        previous_xy = np.asarray([self.pose.x, self.pose.y], dtype=np.float64)
        while simulation_app.is_running() and not follower.done:
            observed = robot_pose(self.robot) if args.wheel_physics_only else self.pose
            linear, angular, _label = follower.command(observed)
            self.robot.apply_wheel_actions(
                command_to_wheel_velocities(linear, angular)
            )
            if not args.wheel_physics_only:
                self.pose = assisted_step(self.pose, linear, angular)
                set_assisted_robot_pose(self.robot, self.pose, linear, angular)
            if self.camera is not None:
                self.camera.set_world_pose(
                    *camera_world_pose(
                        observed if args.wheel_physics_only else self.pose
                    ),
                    camera_axes="world",
                )
            simulation_app.update()
            motion_pose = (
                robot_pose(self.robot) if args.wheel_physics_only else self.pose
            )
            current_xy = np.asarray(
                [motion_pose.x, motion_pose.y], dtype=np.float64
            )
            displacement = current_xy - previous_xy
            if np.linalg.norm(displacement) > 1e-5:
                forward = np.asarray(
                    [math.cos(motion_pose.yaw), math.sin(motion_pose.yaw)],
                    dtype=np.float64,
                )
                if float(np.dot(displacement, forward)) < -1e-5:
                    reverse_motion_frames += 1
            previous_xy = current_xy
            self._publish_live(
                phase,
                f"{phase}：航点 {follower.index}/{max(0, len(path) - 1)}",
                linear=linear,
                angular=angular,
                waypoint=follower.index,
                waypoint_count=max(0, len(path) - 1),
            )
            frame += 1
            if args.steps > 0 and frame >= args.steps:
                break
        self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        app_utils.update_app(steps=5)
        self.pose = robot_pose(self.robot) if args.wheel_physics_only else self.pose
        error = math.dist((self.pose.x, self.pose.y), path[-1])
        result = {
            "success": (
                follower.done and error <= 0.20 and reverse_motion_frames == 0
            ),
            "frames": frame,
            "path_length_m": path_length(path),
            "position_error_m": error,
            "reverse_motion_frames": reverse_motion_frames,
            "forward_only_verified": reverse_motion_frames == 0,
            "final_pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "yaw": self.pose.yaw,
            },
        }
        self.segments.append(result)
        self._publish_live(
            phase, f"{phase} 完成。", waypoint=max(0, len(path) - 1),
            waypoint_count=max(0, len(path) - 1), force=True, result=result,
        )
        return result

    def navigate(self, command, memory):
        from g1d_dual_brain_agent.models import (
            FailureCode,
            SkillResult,
            SkillStatus,
        )

        if (
            command.payload_object_id
            and memory.blackboard.carried_object_id
            != command.payload_object_id
        ):
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                (
                    f"返回导航要求携带 {command.payload_object_id}，"
                    "但对象未通过拿取验证。"
                ),
                FailureCode.OBJECT_SLIPPED,
                {"application_id": self.application_id},
            )
        target = resolve_place(command.instruction, self.places)
        path = self.grid.plan(
            (self.pose.x, self.pose.y), (target.pose.x, target.pose.y)
        )
        phase = "RETURN" if command.payload_object_id else "NAVIGATE"
        result = self._drive(path, target.pose.yaw, phase=phase)
        carry_check = None
        if command.payload_object_id and result["success"]:
            stage = stage_utils.get_current_stage()
            object_prim = stage.GetPrimAtPath(self.carried_physics_prim_path)
            palm_candidates = [
                prim
                for prim in stage.Traverse()
                if prim.GetName() == RIGHT_PALM_LINK
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if not object_prim.IsValid() or len(palm_candidates) != 1:
                carry_check = {
                    "success": False,
                    "reason": "carried_physics_body_or_palm_missing",
                }
            else:
                current_distance = float(
                    np.linalg.norm(
                        _prim_world_position(object_prim)
                        - _prim_world_position(palm_candidates[0])
                    )
                )
                reference = self.carried_reference_palm_distance_m
                drift = (
                    abs(current_distance - reference)
                    if reference is not None
                    else float("inf")
                )
                carry_check = {
                    "success": drift <= 0.025,
                    "palm_object_distance_m": current_distance,
                    "reference_distance_m": reference,
                    "distance_drift_m": drift,
                }
            result["carry_check"] = carry_check
            result["success"] = (
                result["success"] and carry_check["success"]
            )
        return SkillResult(
            command.command_id,
            SkillStatus.SUCCEEDED if result["success"] else SkillStatus.FAILED,
            (
                f"同一 Isaac 会话已到达 {target.place_id}。"
                if result["success"]
                else f"导航未到达 {target.place_id}。"
            ),
            (
                FailureCode.NONE
                if result["success"]
                else (
                    FailureCode.OBJECT_SLIPPED
                    if carry_check is not None
                    else FailureCode.PATH_BLOCKED
                )
            ),
            {
                "application_id": self.application_id,
                "target_place": target.place_id,
                "navigation": result,
            },
        )

    def verify_task(self, command, _memory):
        """Accept a pick only after a physical lift-and-hold backend proves it."""

        from g1d_dual_brain_agent.models import (
            FailureCode,
            SkillResult,
            SkillStatus,
        )

        self._publish_live(
            "VERIFY",
            f"VERIFY：检查 {command.target_id} 的抬升高度和稳定保持。",
            force=True,
        )
        evidence = dict(self.last_manipulation_evidence)
        success = (
            evidence.get("object_id") == command.target_id
            and evidence.get("physical_execution") is True
            and float(evidence.get("lift_height_m", 0.0)) >= 0.05
            and int(evidence.get("stable_hold_frames", 0)) >= 30
        )
        if success:
            body_selection = evidence.get("body_selection", {})
            self.carried_physics_prim_path = str(
                body_selection.get("object_prim_path", "")
            )
            self.carried_reference_palm_distance_m = float(
                evidence.get("final_palm_object_distance_m", 0.0)
            )
        return SkillResult(
            command.command_id,
            SkillStatus.SUCCEEDED if success else SkillStatus.FAILED,
            (
                f"{command.target_id} 已物理抬升并稳定保持。"
                if success
                else (
                    f"{command.target_id} 没有满足 0.05 m 抬升和 "
                    "30 帧稳定保持门槛。"
                )
            ),
            FailureCode.NONE if success else FailureCode.VERIFY_FAILED,
            {
                "application_id": self.application_id,
                "verification": evidence,
                "required_lift_height_m": 0.05,
                "required_stable_hold_frames": 30,
            },
        )

    def _capture_live_views(
        self,
        target: dict,
        *,
        frame_count: int | None = None,
        span_deg: float = 70.0,
        purpose: str = "search",
        pitch_sweep_deg: Sequence[float] | None = None,
    ) -> tuple[Path, Path]:
        if self.camera is None:
            raise RuntimeError("live SEARCH_OBJECT requires the G1-D RGB camera")
        if args.wheel_physics_only:
            raise RuntimeError(
                "live panoramic search does not teleport in --wheel-physics-only mode"
            )
        count = frame_count or args.live_search_frames
        if count < 1 or not 0.0 <= span_deg <= 90.0:
            raise ValueError("invalid live-view frame count or yaw span")
        root = (
            self.output_dir
            / "live_search"
            / (
                f"{time.strftime('%Y%m%dT%H%M%SZ')}-"
                f"{time.time_ns() % 1_000_000_000:09d}-{purpose}"
            )
        )
        rgb_dir = root / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        anchor = target["map_position"]
        center = math.atan2(
            float(anchor["y"]) - self.pose.y,
            float(anchor["x"]) - self.pose.x,
        )
        pitches = tuple(pitch_sweep_deg or (math.degrees(CAMERA_DOWNWARD_PITCH_RAD),))
        if not pitches or any(not 0.0 <= pitch <= 70.0 for pitch in pitches):
            raise ValueError("camera pitch sweep must stay in [0, 70] degrees")
        yaw_count = max(1, math.ceil(count / len(pitches)))
        yaws = [
            center
            + math.radians(
                -span_deg
                + 2.0 * span_deg * index / max(1, yaw_count - 1)
            )
            for index in range(yaw_count)
        ]
        samples = [
            (yaw, math.radians(pitch))
            for pitch in pitches
            for yaw in yaws
        ][:count]
        frames = []
        for index, (yaw, pitch_rad) in enumerate(samples):
            self._rotate_in_place(yaw, "SEARCH_OBJECT")
            self.camera.set_world_pose(
                *camera_world_pose(self.pose, pitch_rad), camera_axes="world"
            )
            app_utils.update_app(steps=4)
            self._publish_live(
                "SEARCH_OBJECT",
                f"SEARCH_OBJECT：机载 RGB 扫描 {index + 1}/{len(samples)}",
                waypoint=index + 1,
                waypoint_count=len(samples),
                force=True,
            )
            image_name = f"{index:06d}.png"
            if not save_camera_rgb(self.camera, rgb_dir / image_name):
                raise RuntimeError(f"RGB camera produced no image at live view {index}")
            frames.append(
                {
                    "frame": index,
                    "image": f"rgb/{image_name}",
                    "robot_pose": {
                        "x": self.pose.x,
                        "y": self.pose.y,
                        "yaw": self.pose.yaw,
                    },
                    "camera_downward_pitch_deg": math.degrees(pitch_rad),
                }
            )
        manifest = {
            "schema_version": 1,
            "scene": FAMILY_HOME_SCENE_NAME,
            "rgb_is_only_model_input": True,
            "object_category_labels_supplied_to_perception": False,
            "pose_consumer": "audit_only_not_live_model_input",
            "purpose": purpose,
            "camera": {"resolution": [camera_width, camera_height]},
            "frames": frames,
        }
        manifest_path = root / "capture_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, rgb_dir

    def _run_live_search_sidecar(
        self,
        target: dict,
        manifest: Path,
        rgb_dir: Path,
        *,
        maximum_frames: int,
    ) -> tuple[dict, Path, Path]:
        result_path = manifest.parent / "search_result.json"
        log_path = manifest.parent / "sidecar.log"
        python = ROOT / "envs/lingbot-map/bin/python"
        process_args = [
            str(python),
            str(ROOT / "scripts/search_live_household_object.py"),
            "--manifest",
            str(manifest),
            "--rgb-dir",
            str(rgb_dir),
            "--catalog",
            str(args.objects),
            "--target",
            target["object_id"],
            "--output",
            str(result_path),
            "--maximum-frames",
            str(maximum_frames),
        ]
        child_env = os.environ.copy()
        # SimulationApp sets Python 3.12 runtime variables in-process. They
        # must not leak into the isolated LingBot Python 3.10 sidecar.
        for key in (
            "PYTHONHOME",
            "PYTHONEXECUTABLE",
            "__PYVENV_LAUNCHER__",
            "_PYTHON_SYSCONFIGDATA_NAME",
        ):
            child_env.pop(key, None)
        child_env["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT),
                str(ROOT / "lingbot_semantic_nav/src"),
                str(ROOT / "lingbot_semantic_nav/third_party/lingbot-map"),
            ]
        )
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["PATH"] = os.pathsep.join(
            [str(python.parent), "/usr/local/bin", "/usr/bin", "/bin"]
        )
        conda_lib = str(python.parent.parent / "lib")
        child_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                conda_lib,
                *[
                    item
                    for item in child_env.get("LD_LIBRARY_PATH", "").split(
                        os.pathsep
                    )
                    if item and "isaacsim/kit" not in item
                ],
            ]
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                process_args,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            started = time.monotonic()
            while process.poll() is None and simulation_app.is_running():
                self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
                simulation_app.update()
                self._publish_live(
                    "SEARCH_OBJECT", "SEARCH_OBJECT：正在分析机器人实时 RGB…"
                )
                if time.monotonic() - started > 240.0:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break
        if not result_path.is_file():
            raise RuntimeError(
                "live RGB search sidecar did not produce a result; "
                f"see {log_path}"
            )
        return (
            json.loads(result_path.read_text(encoding="utf-8")),
            result_path,
            log_path,
        )

    def search_object(self, command, _memory):
        from g1d_dual_brain_agent.models import (
            FailureCode,
            SkillResult,
            SkillStatus,
        )

        target = load_reviewed_object(args.objects, command.target_id)
        try:
            manifest, rgb_dir = self._capture_live_views(target)
        except RuntimeError as exc:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                str(exc),
                FailureCode.UNSUPPORTED_SKILL,
            )
        try:
            result, result_path, log_path = self._run_live_search_sidecar(
                target,
                manifest,
                rgb_dir,
                maximum_frames=args.live_search_frames,
            )
        except RuntimeError as exc:
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                str(exc),
                FailureCode.ADAPTER_ERROR,
                {"application_id": self.application_id},
            )
        matches = result.get("live_matches", [])
        success = result.get("success") is True and bool(matches)
        if success:
            try:
                from family_home_vln.live_object_search import (
                    manipulation_view_gate,
                )

                capture_payload = json.loads(
                    manifest.read_text(encoding="utf-8")
                )
                search_gate = manipulation_view_gate(
                    result,
                    capture_payload,
                    image_size=(camera_width, camera_height),
                )
                selected = search_gate.get("selected")
                if search_gate["ready"] and isinstance(selected, dict):
                    selected_frame = next(
                        frame
                        for frame in capture_payload.get("frames", [])
                        if int(frame.get("frame", -1))
                        == int(selected["frame_index"])
                    )
                    selected_image = manifest.parent / str(
                        selected_frame["image"]
                    )
                    if selected_image.is_file():
                        self.search_handoff_cache[target["object_id"]] = {
                            "gate": search_gate,
                            "image": str(selected_image),
                            "monotonic_sec": time.monotonic(),
                            "source": "search_object_live_rgb",
                        }
            except (KeyError, RuntimeError, StopIteration, TypeError, ValueError):
                pass
        anchor = target["map_position"]
        distance = math.dist(
            (self.pose.x, self.pose.y),
            (float(anchor["x"]), float(anchor["y"])),
        )
        update = {
            "object_id": target["object_id"],
            "labels": list(target.get("aliases", [])),
            "global_pose": dict(anchor),
            "local_pose": {
                "range_m": distance,
                "bearing_rad": math.atan2(
                    float(anchor["y"]) - self.pose.y,
                    float(anchor["x"]) - self.pose.x,
                )
                - self.pose.yaw,
                "source": "reviewed_static_map_anchor_after_live_rgb_confirmation",
            },
            "visible": success,
            "detection_confidence": (
                min(
                    1.0,
                    max(
                        float(item.get("frame_occurrences", 1))
                        for item in matches
                    )
                    / max(1, args.live_search_frames),
                )
                if success
                else 0.0
            ),
            "last_seen_monotonic_sec": time.monotonic(),
            "observation_source": "live_florence2_category_free_rgb",
            "map_revision": str(
                json.loads(args.objects.read_text(encoding="utf-8"))["map"][
                    "sha256"
                ]
            ),
            "reachable": None,
        }
        return SkillResult(
            command.command_id,
            SkillStatus.SUCCEEDED if success else SkillStatus.FAILED,
            (
                f"live RGB confirmed {target['object_id']}."
                if success
                else f"live RGB did not confirm {target['object_id']}."
            ),
            FailureCode.NONE if success else FailureCode.TARGET_NOT_FOUND,
            {
                "application_id": self.application_id,
                "result": str(result_path),
                "log": str(log_path),
                "category_list_supplied_to_model": False,
            },
            (update,),
        )

    def approach_and_align(self, command, memory):
        from g1d_dual_brain_agent.models import (
            FailureCode,
            SkillResult,
            SkillStatus,
        )

        target = load_reviewed_object(args.objects, command.target_id)
        anchor_payload = target["map_position"]
        anchor = (float(anchor_payload["x"]), float(anchor_payload["y"]))
        approach = target["approach"]
        preferred_bearing = (
            float(approach["preferred_view_bearing_rad"])
            if approach.get("preferred_view_bearing_rad") is not None
            else None
        )
        tolerance = float(approach["alignment_tolerance_m"])
        visibility_standoff = float(
            approach.get("visibility_standoff_m", approach["stand_off_m"])
        )
        manipulation_standoff = float(approach["stand_off_m"])
        self.last_handoff_image_path = ""
        self.last_handoff_gate = {}
        try:
            visibility_goal, visibility_path = plan_object_approach(
                self.grid,
                self.pose,
                anchor,
                stand_off_m=visibility_standoff,
                tolerance_m=max(tolerance, 0.08),
                preferred_view_bearing_rad=preferred_bearing,
            )
        except ValueError as exc:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                str(exc),
                FailureCode.OUT_OF_REACH,
            )
        visibility_result = self._drive(
            visibility_path,
            visibility_goal.yaw,
            precision=True,
            phase="APPROACH_AND_ALIGN",
        )
        visibility_distance = math.dist((self.pose.x, self.pose.y), anchor)
        visibility_yaw_error = abs(
            math.atan2(
                math.sin(visibility_goal.yaw - self.pose.yaw),
                math.cos(visibility_goal.yaw - self.pose.yaw),
            )
        )
        visibility_aligned = (
            visibility_result["success"]
            and abs(visibility_distance - visibility_standoff)
            <= max(tolerance, 0.08) + 0.02
            and visibility_yaw_error <= 0.18
        )
        record = memory.get_object(target["object_id"])
        visibility_aligned = visibility_aligned and bool(
            record and record.visible
        )
        handoff_gate: dict = {
            "ready": False,
            "reason": "geometric_alignment_or_search_visibility_failed",
        }
        handoff_result_path: Path | None = None
        if visibility_aligned:
            try:
                from family_home_vln.live_object_search import (
                    manipulation_view_gate,
                )

                handoff_manifest, handoff_rgb = self._capture_live_views(
                    target,
                    frame_count=9,
                    span_deg=18.0,
                    purpose="vla-handoff",
                    pitch_sweep_deg=(25.0, 35.0, 45.0),
                )
                handoff_search, handoff_result_path, _handoff_log = (
                    self._run_live_search_sidecar(
                        target,
                        handoff_manifest,
                        handoff_rgb,
                        maximum_frames=9,
                    )
                )
                capture_payload = json.loads(
                    handoff_manifest.read_text(encoding="utf-8")
                )
                handoff_gate = manipulation_view_gate(
                    handoff_search,
                    capture_payload,
                    image_size=(camera_width, camera_height),
                )
                selected = handoff_gate.get("selected")
                if handoff_gate["ready"] and isinstance(selected, dict):
                    selected_frame = next(
                        frame
                        for frame in capture_payload.get("frames", [])
                        if int(frame.get("frame", -1))
                        == int(selected["frame_index"])
                    )
                    selected_image = (
                        handoff_manifest.parent / str(selected_frame["image"])
                    )
                    if not selected_image.is_file():
                        raise RuntimeError(
                            "selected VLA handoff image is missing"
                        )
                    selected_pose = selected.get("robot_pose", {})
                    self.pose = Pose2D(
                        float(selected_pose["x"]),
                        float(selected_pose["y"]),
                        float(selected_pose["yaw"]),
                    )
                    self.manipulation_camera_pitch_rad = math.radians(
                        float(
                            selected.get(
                                "camera_downward_pitch_deg",
                                math.degrees(CAMERA_DOWNWARD_PITCH_RAD),
                            )
                        )
                    )
                    set_assisted_robot_pose(self.robot, self.pose, 0.0, 0.0)
                    self.camera.set_world_pose(
                        *camera_world_pose(
                            self.pose,
                            self.manipulation_camera_pitch_rad,
                        ),
                        camera_axes="world",
                    )
                    app_utils.update_app(steps=5)
                    self.last_handoff_image_path = str(selected_image)
                    handoff_gate["observation_phase"] = (
                        "visible_staging_before_arm_reach_alignment"
                    )
                    handoff_gate["selected_image"] = str(selected_image)
                    self.last_handoff_gate = dict(handoff_gate)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                handoff_gate = {
                    "ready": False,
                    "reason": f"live_handoff_revalidation_failed: {exc}",
                }
        if not handoff_gate["ready"]:
            cached = self.search_handoff_cache.get(target["object_id"], {})
            cache_age = time.monotonic() - float(
                cached.get("monotonic_sec", float("-inf"))
            )
            cached_image = Path(str(cached.get("image", "")))
            cached_gate = cached.get("gate", {})
            if (
                0.0 <= cache_age <= 120.0
                and cached_image.is_file()
                and isinstance(cached_gate, dict)
                and cached_gate.get("ready") is True
            ):
                handoff_gate = dict(cached_gate)
                handoff_gate["reason"] = (
                    "fresh_search_object_rgb_gate_reused_after_scan_bearing_alignment"
                )
                handoff_gate["observation_phase"] = "search_object"
                handoff_gate["observation_age_sec"] = cache_age
                handoff_gate["selected_image"] = str(cached_image)
                handoff_gate["freshness_limit_sec"] = 120.0
                self.last_handoff_image_path = str(cached_image)
                self.last_handoff_gate = dict(handoff_gate)
        manipulation_result: dict | None = None
        distance = visibility_distance
        yaw_error = visibility_yaw_error
        reach_aligned = False
        if handoff_gate["ready"]:
            try:
                reach_goal, reach_path = plan_object_approach(
                    self.grid,
                    self.pose,
                    anchor,
                    stand_off_m=manipulation_standoff,
                    tolerance_m=tolerance,
                    preferred_view_bearing_rad=preferred_bearing,
                )
                manipulation_result = self._drive(
                    reach_path,
                    reach_goal.yaw,
                    precision=True,
                    phase="APPROACH_AND_ALIGN",
                )
                distance = math.dist((self.pose.x, self.pose.y), anchor)
                yaw_error = abs(
                    math.atan2(
                        math.sin(reach_goal.yaw - self.pose.yaw),
                        math.cos(reach_goal.yaw - self.pose.yaw),
                    )
                )
                reach_aligned = (
                    manipulation_result["success"]
                    and abs(distance - manipulation_standoff)
                    <= tolerance + 0.03
                    and yaw_error <= 0.18
                )
            except ValueError as exc:
                handoff_gate["reach_alignment_error"] = str(exc)
        aligned = visibility_aligned and handoff_gate["ready"] and reach_aligned
        visible = bool(handoff_gate["ready"])
        update = {
            "object_id": target["object_id"],
            "visible": visible,
            "reachable": aligned,
            "reachability_context": {
                "application_id": self.application_id,
                "stand_off_m": distance,
                "required_stand_off_m": manipulation_standoff,
                "distance_tolerance_m": tolerance,
                "yaw_error_rad": yaw_error,
                "base_stopped": True,
                "formal_occupancy_checked": True,
                "manipulation_ready": target["manipulation_ready"],
                "live_rgb_handoff_gate": handoff_gate,
                "visibility_standoff_m": visibility_distance,
                "visibility_verified_before_close_alignment": visible,
            },
            "last_result": "aligned" if aligned else "alignment_failed",
        }
        return SkillResult(
            command.command_id,
            SkillStatus.SUCCEEDED if aligned else SkillStatus.FAILED,
            (
                f"G1-D aligned to {target['object_id']} at {distance:.3f} m."
                if aligned
                else f"G1-D object alignment failed at {distance:.3f} m."
            ),
            FailureCode.NONE if aligned else FailureCode.BAD_VIEWPOINT,
            {
                "application_id": self.application_id,
                "visibility_navigation": visibility_result,
                "manipulation_reach_navigation": manipulation_result,
                "distance_m": distance,
                "yaw_error_rad": yaw_error,
                "live_rgb_handoff_gate": handoff_gate,
                "live_rgb_handoff_result": (
                    str(handoff_result_path)
                    if handoff_result_path is not None
                    else ""
                ),
            },
            (update,),
        )

    def _execute_sim_pick(self, target: dict) -> dict:
        """Run the simulation-only semantic-IK pick primitive."""

        anchor = target.get("map_position", {})
        if "z" not in anchor:
            return {
                "success": False,
                "reason": "reviewed_scan_anchor_has_no_metric_z",
                "physical_execution": False,
            }
        target_world = np.asarray(
            [
                float(anchor["x"]),
                float(anchor["y"]),
                ROOM_FLOOR_Z_M + float(anchor["z"]),
            ],
            dtype=np.float64,
        )
        base_to_target = target_world[:2] - np.asarray(
            [self.pose.x, self.pose.y],
            dtype=np.float64,
        )
        planar_distance = float(np.linalg.norm(base_to_target))
        if not 0.35 <= planar_distance <= 0.78:
            return {
                "success": False,
                "reason": "scan_anchor_outside_reviewed_right_arm_standoff",
                "planar_distance_m": planar_distance,
                "physical_execution": False,
            }
        direction = base_to_target / planar_distance
        open_result = _set_right_hand(
            self.robot,
            RIGHT_HAND_OPEN_RAD,
            progress_callback=lambda step, total: self._manipulation_progress(
                "张开右手", step, total
            ),
        )
        pregrasp = target_world.copy()
        pregrasp[:2] -= direction * 0.11
        pregrasp[2] += 0.025
        pregrasp_result = move_right_palm_to(
            self.robot,
            pregrasp,
            maximum_cartesian_travel_m=0.75,
            tolerance_m=0.035,
            progress_callback=lambda step, total: self._manipulation_progress(
                "移动到预抓取位", step, total
            ),
        )
        if not pregrasp_result["success"]:
            return {
                "success": False,
                "reason": "pregrasp_ik_failed",
                "physical_execution": True,
                "open_hand": open_result,
                "pregrasp": pregrasp_result,
            }

        grasp = target_world.copy()
        # The G1-D hand closes around the object from the fingertip envelope;
        # requiring the palm center itself to reach the cup center overdrives
        # the seven-axis arm near its forward workspace boundary.
        grasp[:2] -= direction * 0.075
        grasp_result = move_right_palm_to(
            self.robot,
            grasp,
            maximum_cartesian_travel_m=0.20,
            tolerance_m=0.030,
            progress_callback=lambda step, total: self._manipulation_progress(
                "靠近杯子", step, total
            ),
        )
        if not grasp_result["success"]:
            return {
                "success": False,
                "reason": "grasp_ik_failed",
                "physical_execution": True,
                "open_hand": open_result,
                "pregrasp": pregrasp_result,
                "grasp": grasp_result,
            }
        try:
            palm_prim, object_prim, body_selection = _find_sim_grasp_bodies(
                target_world
            )
        except RuntimeError as exc:
            return {
                "success": False,
                "reason": f"physics_object_resolution_failed: {exc}",
                "physical_execution": True,
                "pregrasp": pregrasp_result,
                "grasp": grasp_result,
            }
        object_before = _prim_world_position(object_prim)
        palm_before = _prim_world_position(palm_prim)
        palm_object_distance = float(
            np.linalg.norm(palm_before - object_before)
        )
        if palm_object_distance > RIGHT_HAND_FINGERTIP_REACH_M:
            return {
                "success": False,
                "reason": "palm_not_close_enough_to_physics_object",
                "physical_execution": True,
                "palm_object_distance_m": palm_object_distance,
                "maximum_palm_object_distance_m": (
                    RIGHT_HAND_FINGERTIP_REACH_M
                ),
                "body_selection": body_selection,
                "pregrasp": pregrasp_result,
                "grasp": grasp_result,
            }

        close_result = _set_right_hand(
            self.robot,
            RIGHT_HAND_CLOSED_RAD,
            progress_callback=lambda step, total: self._manipulation_progress(
                "闭合手指", step, total
            ),
        )
        self._publish_live(
            "OPENVLA_PICK",
            "OPENVLA_PICK：手指已闭合，建立物理抓取约束。",
            force=True,
        )
        constraint_path = _create_sim_grasp_constraint(
            palm_prim,
            object_prim,
            target["object_id"],
        )
        object_after_attachment = _prim_world_position(object_prim)
        attachment_displacement = float(
            np.linalg.norm(object_after_attachment - object_before)
        )
        if attachment_displacement > 0.04:
            stage_utils.get_current_stage().RemovePrim(constraint_path)
            app_utils.update_app(steps=8)
            return {
                "success": False,
                "reason": "simulation_constraint_caused_excessive_object_snap",
                "physical_execution": True,
                "attachment_displacement_m": attachment_displacement,
                "constraint_removed": True,
                "body_selection": body_selection,
                "close_hand": close_result,
            }

        palm_lift_target = link_world_position(
            self.robot, RIGHT_PALM_LINK
        ) + np.asarray([0.0, 0.0, 0.09], dtype=np.float64)
        lift_result = move_right_palm_to(
            self.robot,
            palm_lift_target,
            maximum_cartesian_travel_m=0.14,
            tolerance_m=0.025,
            progress_callback=lambda step, total: self._manipulation_progress(
                "连续抬升杯子", step, total
            ),
        )
        relative_distances = []
        object_heights = []
        for _frame in range(45):
            simulation_app.update()
            self._manipulation_progress("稳定保持", _frame + 1, 45)
            object_now = _prim_world_position(object_prim)
            palm_now = _prim_world_position(palm_prim)
            object_heights.append(float(object_now[2]))
            relative_distances.append(
                float(np.linalg.norm(object_now - palm_now))
            )
        object_after = _prim_world_position(object_prim)
        lift_height = float(object_after[2] - object_before[2])
        relative_drift = (
            max(relative_distances) - min(relative_distances)
            if relative_distances
            else float("inf")
        )
        stable_window_frames = 30
        stable_heights = object_heights[-stable_window_frames:]
        stable_distances = relative_distances[-stable_window_frames:]
        stable_height_range = (
            max(stable_heights) - min(stable_heights)
            if stable_heights
            else float("inf")
        )
        stable_distance_range = (
            max(stable_distances) - min(stable_distances)
            if stable_distances
            else float("inf")
        )
        stable_hold_frames = (
            len(stable_heights)
            if len(stable_heights) == stable_window_frames
            and stable_height_range <= 0.015
            and stable_distance_range <= 0.015
            else 0
        )
        success = (
            lift_result["success"]
            and lift_height >= 0.05
            and stable_hold_frames >= 30
        )
        return {
            "success": success,
            "reason": (
                "scan_guided_simulation_constraint_pick_verified"
                if success
                else "lift_or_stable_hold_gate_failed"
            ),
            "physical_execution": True,
            "execution_environment": "isaac_sim_only",
            "hardware_output": False,
            "grasp_mechanism": (
                "g1d_hand_close_then_explicit_physx_fixed_joint"
            ),
            "openvla_role": (
                "live_rgb_policy_inference_advisory; uncalibrated Cartesian "
                "delta is not applied to joints"
            ),
            "object_id": target["object_id"],
            "constraint_path": constraint_path,
            "body_selection": body_selection,
            "open_hand": open_result,
            "pregrasp": pregrasp_result,
            "grasp": grasp_result,
            "palm_object_distance_m": palm_object_distance,
            "maximum_palm_object_distance_m": RIGHT_HAND_FINGERTIP_REACH_M,
            "close_hand": close_result,
            "attachment_displacement_m": attachment_displacement,
            "lift": lift_result,
            "object_start_world_m": object_before.tolist(),
            "object_final_world_m": object_after.tolist(),
            "lift_height_m": lift_height,
            "stable_hold_frames": stable_hold_frames,
            "maximum_relative_hold_drift_m": relative_drift,
            "stable_window_height_range_m": stable_height_range,
            "stable_window_relative_distance_range_m": stable_distance_range,
            "final_palm_object_distance_m": (
                relative_distances[-1] if relative_distances else None
            ),
            "scene_collision_query": False,
        }

    def manipulate_openvla(self, command, _memory):
        """Infer OpenVLA, then optionally run the explicit sim pick primitive."""

        from g1d_dual_brain_agent.models import (
            FailureCode,
            SkillResult,
            SkillStatus,
        )
        from g1d_openvla import (
            OpenVlaAction,
            build_g1d_right_arm_handoff,
            inspect_checkpoint,
        )

        self._publish_live(
            "OPENVLA_PICK",
            f"OPENVLA_PICK：准备对 {command.target_id} 执行视觉动作推理。",
            force=True,
        )
        if self.camera is None:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                "OpenVLA requires the live G1-D RGB camera.",
                FailureCode.UNSUPPORTED_SKILL,
            )
        if not args.openvla_python.is_file():
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                f"OpenVLA Python is missing: {args.openvla_python}",
                FailureCode.VLA_UNAVAILABLE,
            )
        checkpoint = inspect_checkpoint(args.openvla_model)
        if not checkpoint.ready:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                (
                    "OpenVLA checkpoint is incomplete: "
                    f"{checkpoint.actual_bytes}/{checkpoint.expected_bytes} bytes"
                ),
                FailureCode.VLA_UNAVAILABLE,
                {
                    "model": str(args.openvla_model),
                    "missing_files": list(checkpoint.missing_files),
                },
            )

        target = load_reviewed_object(args.objects, command.target_id)
        root = self.output_dir / "openvla" / time.strftime("%Y%m%dT%H%M%SZ")
        root.mkdir(parents=True, exist_ok=True)
        image_path = root / "head_rgb.png"
        result_path = root / "inference.json"
        handoff_path = root / "g1d_right_arm_handoff.json"
        log_path = root / "sidecar.log"
        self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        app_utils.update_app(steps=5)
        staged_observation = Path(self.last_handoff_image_path)
        if not staged_observation.is_file():
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                "APPROACH_AND_ALIGN produced no gated RGB handoff image.",
                FailureCode.BAD_VIEWPOINT,
                {"application_id": self.application_id},
            )
        shutil.copy2(staged_observation, image_path)

        instruction = args.openvla_instruction.strip() or (
            f"move the robot hand toward the {target['source_label']}"
        )
        process_args = [
            str(args.openvla_python),
            str(ROOT / "scripts/run_openvla_inference.py"),
            "--model",
            str(args.openvla_model),
            "--image",
            str(image_path),
            "--instruction",
            instruction,
            "--unnorm-key",
            args.openvla_unnorm_key,
            "--output",
            str(result_path),
        ]
        child_env = os.environ.copy()
        for key in (
            "PYTHONHOME",
            "PYTHONEXECUTABLE",
            "__PYVENV_LAUNCHER__",
            "_PYTHON_SYSCONFIGDATA_NAME",
        ):
            child_env.pop(key, None)
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["HF_HOME"] = str(ROOT / ".cache/huggingface")
        child_env["PYTHONPATH"] = str(ROOT)
        child_env["PATH"] = os.pathsep.join(
            [str(args.openvla_python.parent), "/usr/local/bin", "/usr/bin", "/bin"]
        )
        conda_lib = str(args.openvla_python.parent.parent / "lib")
        child_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                conda_lib,
                *[
                    item
                    for item in child_env.get("LD_LIBRARY_PATH", "").split(
                        os.pathsep
                    )
                    if item and "isaacsim/kit" not in item
                ],
            ]
        )
        timed_out = False
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                process_args,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            started = time.monotonic()
            while process.poll() is None and simulation_app.is_running():
                self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
                simulation_app.update()
                if time.monotonic() - started > args.openvla_timeout_sec:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
                    timed_out = True
                    break
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
        if timed_out or process.returncode != 0 or not result_path.is_file():
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                (
                    "OpenVLA inference timed out."
                    if timed_out
                    else f"OpenVLA sidecar failed with code {process.returncode}."
                ),
                FailureCode.ADAPTER_ERROR,
                {
                    "application_id": self.application_id,
                    "log": str(log_path),
                    "image": str(image_path),
                },
            )

        inference = json.loads(result_path.read_text(encoding="utf-8"))
        action = OpenVlaAction.from_values(
            inference.get("action", []),
            unnorm_key=args.openvla_unnorm_key,
        )
        handoff = build_g1d_right_arm_handoff(action)
        handoff.update(
            {
                "application_id": self.application_id,
                "inference_artifact": str(result_path),
                "observation_image": str(image_path),
                "instruction": instruction,
                "observation_phase": self.last_handoff_gate.get(
                    "observation_phase",
                    "visible_staging_before_final_arm_reach_base_alignment",
                ),
                "observation_age_sec": self.last_handoff_gate.get(
                    "observation_age_sec", 0.0
                ),
                "final_reach_alignment_uses": (
                    "reviewed_metric_object_anchor_and_formal_occupancy"
                ),
                "target_object": {
                    "object_id": target["object_id"],
                    "source_label": target["source_label"],
                    "manipulation_ready": bool(target["manipulation_ready"]),
                },
            }
        )
        visibility_block = (
            "target_visibility_not_revalidated_in_final_openvla_frame"
        )
        if visibility_block in handoff["blocked_reasons"]:
            handoff["blocked_reasons"].remove(visibility_block)
        handoff["validated_gates"] = [
            "base_stopped",
            "fresh_live_category_free_rgb_target_view_available",
            "target_bbox_clears_edge_and_scale_gates",
            "formal_occupancy_arm_reach_base_alignment_completed_after_staging",
        ]
        if not target["manipulation_ready"]:
            handoff["blocked_reasons"].append(
                "reviewed_object_is_search_and_docking_only_not_manipulation_ready"
            )
        handoff_path.write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.execute_sim_pick:
            if not target["manipulation_ready"]:
                return SkillResult(
                    command.command_id,
                    SkillStatus.BLOCKED,
                    (
                        f"{target['object_id']} 未通过 manipulation_ready "
                        "审核，拒绝执行仿真拿取。"
                    ),
                    FailureCode.UNSUPPORTED_SKILL,
                    {
                        "application_id": self.application_id,
                        "openvla_inference_succeeded": True,
                        "handoff": str(handoff_path),
                    },
                )
            evidence = self._execute_sim_pick(target)
            evidence["openvla_inference"] = str(result_path)
            evidence["openvla_action"] = action.to_dict()
            self.last_manipulation_evidence = evidence
            evidence_path = root / "sim_pick_evidence.json"
            evidence_path.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return SkillResult(
                command.command_id,
                (
                    SkillStatus.SUCCEEDED
                    if evidence["success"]
                    else SkillStatus.FAILED
                ),
                (
                    f"{target['object_id']} 已在 Isaac 中抬升并稳定保持。"
                    if evidence["success"]
                    else (
                        f"{target['object_id']} 仿真拿取未通过："
                        f"{evidence['reason']}"
                    )
                ),
                (
                    FailureCode.NONE
                    if evidence["success"]
                    else FailureCode.GRASP_FAILED
                ),
                {
                    "application_id": self.application_id,
                    "openvla_inference_succeeded": True,
                    "inference": str(result_path),
                    "handoff": str(handoff_path),
                    "sim_pick_evidence": str(evidence_path),
                    "manipulation": evidence,
                },
            )
        return SkillResult(
            command.command_id,
            SkillStatus.BLOCKED,
            (
                "OpenVLA produced a real 7-D action from live RGB, but the "
                "Bridge action was not sent to G1-D joints because calibrated "
                "right-arm IK, collision checking and hand mapping are absent."
            ),
            FailureCode.UNSUPPORTED_SKILL,
            {
                "application_id": self.application_id,
                "openvla_inference_succeeded": True,
                "inference": str(result_path),
                "handoff": str(handoff_path),
                "log": str(log_path),
                "action": action.to_dict(),
            },
        )


def run_dual_agent_session(session: FamilyHomeDualAgentSession) -> dict:
    from g1d_dual_brain_agent.executive import DualBrainExecutive
    from g1d_dual_brain_agent.memory import SharedWorldMemory
    from g1d_dual_brain_agent.models import GoalKind, Mission, TaskGoal
    from g1d_dual_brain_agent.skills import (
        CallableSkillExecutor,
        SkillRegistry,
        UnavailableSkillExecutor,
    )
    from g1d_dual_brain_agent.models import SkillKind

    memory_path = args.output_dir / "dual_agent_world_memory.json"
    memory = SharedWorldMemory(memory_path)
    skills = SkillRegistry()
    skills.register(
        SkillKind.NAVIGATE, CallableSkillExecutor(session.navigate)
    )
    skills.register(
        SkillKind.SEARCH_OBJECT, CallableSkillExecutor(session.search_object)
    )
    skills.register(
        SkillKind.APPROACH_ALIGN,
        CallableSkillExecutor(session.approach_and_align),
    )
    if args.openvla:
        skills.register(
            SkillKind.MANIPULATE,
            CallableSkillExecutor(session.manipulate_openvla),
        )
    else:
        skills.register(
            SkillKind.MANIPULATE,
            UnavailableSkillExecutor(
                SkillKind.MANIPULATE,
                "OpenVLA is disabled; pass --openvla to run diagnostic inference",
            ),
        )
    skills.register(
        SkillKind.VERIFY,
        CallableSkillExecutor(session.verify_task),
    )
    if args.family_task:
        from g1d_dual_brain_agent.planner import compile_family_home_command

        places_catalog = json.loads(args.places.read_text(encoding="utf-8"))
        objects_catalog = json.loads(args.objects.read_text(encoding="utf-8"))
        mission = compile_family_home_command(
            args.command,
            places_catalog=places_catalog,
            objects_catalog=objects_catalog,
            mission_id=f"family-home-{int(time.time())}",
        )
    else:
        target = load_reviewed_object(args.objects, args.target_object)
        mission = Mission(
            mission_id=f"family-home-{int(time.time())}",
            instruction=(
                f"{args.command}; 搜索并对齐 {target['object_id']}，随后交给 VLA"
            ),
            goals=(
                TaskGoal(
                    goal_id="mobile-manipulation-1",
                    kind=GoalKind.INTERACT,
                    instruction=f"操作 {target['object_id']}",
                    target_id=target["object_id"],
                    action="future_vla_manipulation",
                    region_hint=args.command,
                    success_condition="VLA execution and independent verification succeed",
                ),
            ),
            maximum_attempts_per_skill=1,
        )
    result = DualBrainExecutive(
        skills,
        memory,
        maximum_object_observation_age_sec=300.0,
    ).execute(mission)
    payload = result.to_dict()
    succeeded_skills = {
        event.get("payload", {}).get("result", {}).get("details", {}).get(
            "application_id"
        ): event.get("payload", {}).get("result", {}).get("status")
        for event in payload["events"]
        if event.get("type") == "skill_finished"
    }
    payload["same_simulation_app"] = (
        set(key for key in succeeded_skills if key) == {session.application_id}
    )
    payload["application_id"] = session.application_id
    payload["navigation_segments"] = session.segments
    payload["pre_vla_pipeline_succeeded"] = all(
        any(
            event.get("type") == "skill_finished"
            and event.get("payload", {}).get("result", {}).get("status")
            == "succeeded"
            and event.get("payload", {}).get("result", {}).get("details", {}).get(
                "application_id"
            )
            == session.application_id
            and event.get("payload", {})
            .get("result", {})
            .get("message", "")
            .startswith(prefix)
            for event in payload["events"]
        )
        for prefix in ("同一 Isaac 会话已到达", "live RGB confirmed", "G1-D aligned")
    )
    payload["openvla_inference_succeeded"] = any(
        event.get("type") == "skill_finished"
        and event.get("payload", {})
        .get("result", {})
        .get("details", {})
        .get("openvla_inference_succeeded")
        is True
        for event in payload["events"]
    )
    return payload


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.right_arm_probe:
        print("Mode: bounded G1-D right-arm kinematics probe")
    if args.survey or args.allow_bootstrap:
        if args.scene_profile == "family-home":
            grid, places = build_family_home_bootstrap_artifacts(args.output_dir)
            map_source = "reviewed_procedural_family_home_bootstrap"
        else:
            grid, places = build_bootstrap_artifacts(args.output_dir)
            map_source = "isaac_geometry_bootstrap"
    else:
        required_artifacts = [args.map, args.places]
        if args.dual_agent:
            required_artifacts.append(args.objects)
        missing = [str(path) for path in required_artifacts if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "正式导航拒绝退化到 Isaac 几何；缺少 LingBot/SAM3 工件："
                + ", ".join(missing)
                + "。请先运行 build_simple_room_vln.ps1，或仅调试时显式传 --allow-bootstrap。"
            )
        if args.scene_profile == "family-home":
            formal_catalog = json.loads(args.places.read_text(encoding="utf-8"))
            actual_signature = formal_catalog.get("map", {}).get(
                "household_object_set_signature"
            )
            if actual_signature != OBJECT_SET_SIGNATURE:
                raise ValueError(
                    "家庭正式地图与当前自主发现物品集不一致；必须重新运行 "
                    "home-survey -> home-map，禁止在旧 occupancy 上导航。"
                )
        grid, places = load_lingbot_artifacts(args.map, args.places)
        map_source = (
            "lingbot_rgb_depth+isaac_survey_pose_offline_diagnostic"
            if args.map.parent.name == "lingbot_pose_fused_map"
            else "lingbot_map_rgb_only_pointcloud"
        )
    if args.survey:
        target = None
        path = (
            build_family_home_survey_path(grid)
            if args.scene_profile == "family-home"
            else build_simple_survey_path(grid)
        )
        final_yaw = 0.0
        task_name = "rgb_survey"
    else:
        target = resolve_place(args.command, places)
        start = (
            FAMILY_HOME_START
            if args.scene_profile == "family-home"
            else Pose2D(0.0, 0.0, 0.0)
        )
        path = grid.plan((start.x, start.y), (target.pose.x, target.pose.y))
        final_yaw = target.pose.yaw
        task_name = target.place_id

    print(f"Map source: {map_source}")
    print(f"Task: {task_name}")
    if target is not None:
        print(f"Resolved command {args.command!r} -> {target.place_id}")
    print(f"Planned {len(path)} waypoints, length={path_length(path):.3f} m")

    start_pose = Pose2D(path[0][0], path[0][1], 0.0)
    live = None
    if args.live_dir is not None:
        live = LivePublisher(
            args.live_dir,
            command="" if args.survey else args.command,
            task=task_name,
            map_source=map_source,
            path=path,
            mission_mode=("dual_brain_task" if args.family_task else "vln_navigation"),
            mission_steps=(
                (
                    "NAVIGATE",
                    "SEARCH_OBJECT",
                    "APPROACH_AND_ALIGN",
                    "OPENVLA_PICK",
                    "VERIFY",
                    "RETURN",
                )
                if args.family_task
                else ("NAVIGATE", "ARRIVE")
            ),
        )
        live.publish_state(
            state="loading",
            message="正在加载家庭场景、G1-D 和 RTX 总览相机…",
            frame=0,
            action="loading",
            pose=start_pose,
            linear=0.0,
            angular=0.0,
            waypoint=0,
            waypoint_count=max(0, len(path) - 1),
        )

    add_composed_scene(path, None if target is None else target.pose)
    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT],
        usd_path=str(ROBOT_USD).replace("\\", "/"),
        positions=[path[0][0], path[0][1], ROOM_FLOOR_Z_M + 0.12],
    )

    camera = None
    recorder = None
    if not args.no_camera:
        from isaacsim.sensors.camera import Camera

        camera_position, camera_orientation = camera_world_pose(
            Pose2D(path[0][0], path[0][1], 0.0)
        )
        camera = Camera(
            prim_path="/World/G1DRgbCamera",
            position=camera_position,
            orientation=camera_orientation,
            frequency=20,
            resolution=(camera_width, camera_height),
        )

    overview_camera = None
    gif_frames = []
    if args.record_gif is not None or live is not None:
        if args.record_fps <= 0 or args.record_fps > PHYSICS_HZ:
            raise ValueError("--record-fps must be between 1 and 60")
        from isaacsim.sensors.camera import Camera

        overview_eye = (
            HOME_OVERVIEW_EYE
            if args.scene_profile == "family-home"
            else OVERVIEW_EYE
        )
        overview_target = (
            HOME_OVERVIEW_TARGET
            if args.scene_profile == "family-home"
            else OVERVIEW_TARGET
        )
        overview_position, overview_orientation = (
            home_chase_camera_pose(start_pose)
            if live is not None and args.scene_profile == "family-home"
            else look_at_camera_pose(overview_eye, overview_target)
        )
        overview_camera = Camera(
            prim_path="/World/VLNOverviewCamera",
            position=overview_position,
            orientation=overview_orientation,
            frequency=max(
                args.record_fps if args.record_gif is not None else 1,
                args.live_fps if live is not None else 1,
            ),
            resolution=(
                (live_width, live_height)
                if live is not None
                else (640, 480)
            ),
        )

    if not args.headless:
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=(
                [5.5, -5.0, 5.5]
                if args.scene_profile == "family-home"
                else [1.5, -1.5, 1.8]
            ),
            target=(
                [0.0, 0.7, 0.0]
                if args.scene_profile == "family-home"
                else [-1.8, 1.4, 0.2]
            ),
        )

    SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ, device="cpu")
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    app_utils.play()
    app_utils.update_app(steps=24)
    configure_joint_drives(robot)

    pose = Pose2D(path[0][0], path[0][1], 0.0)
    if not args.wheel_physics_only:
        set_assisted_robot_pose(robot, pose, 0.0, 0.0)

    if camera is not None:
        camera.initialize()
        camera.set_focal_length(CAMERA_FOCAL_LENGTH_MM)
        camera.set_horizontal_aperture(CAMERA_HORIZONTAL_APERTURE_MM)
        camera.set_world_pose(*camera_world_pose(pose), camera_axes="world")
        app_utils.update_app(steps=30)
        if args.survey:
            recorder = SurveyRecorder(args.output_dir, camera, camera.get_intrinsics_matrix())
            recorder.maybe_capture(pose, force=True)
    if overview_camera is not None:
        overview_camera.initialize()
        overview_camera.set_focal_length(14.0)
        overview_camera.set_horizontal_aperture(28.0)
        overview_eye = (
            HOME_OVERVIEW_EYE
            if args.scene_profile == "family-home"
            else OVERVIEW_EYE
        )
        overview_target = (
            HOME_OVERVIEW_TARGET
            if args.scene_profile == "family-home"
            else OVERVIEW_TARGET
        )
        overview_camera.set_world_pose(
            *(
                home_chase_camera_pose(pose)
                if live is not None and args.scene_profile == "family-home"
                else look_at_camera_pose(overview_eye, overview_target)
            ),
            camera_axes="world",
        )
        app_utils.update_app(steps=20)

    if args.right_arm_probe:
        probe_path = args.output_dir / "g1d_right_arm_probe.json"
        payload = run_right_arm_position_probe(
            robot,
            output_path=probe_path,
        )
        print(
            "Right-arm probe: "
            f"success={payload['success']} "
            f"error={payload['final_position_error_m']:.4f} m "
            f"max_joint_delta={payload['maximum_joint_delta_rad']:.4f} rad"
        )
        print(f"Summary: {probe_path}")
        simulation_app.close()
        return 0 if payload["success"] else 1

    if args.dual_agent:
        session = FamilyHomeDualAgentSession(
            robot,
            camera,
            grid,
            places,
            args.output_dir,
            live=live,
            overview_camera=overview_camera,
        )
        payload = run_dual_agent_session(session)
        summary_path = args.output_dir / "dual_agent_run_summary.json"
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        app_utils.update_app(steps=5)
        if live is not None:
            succeeded = payload["status"] == "succeeded"
            live.publish_state(
                state="succeeded" if succeeded else "failed",
                message=(
                    "任务完成：机器人已完成导航、找物、对齐、拿取验证并返回。"
                    if succeeded
                    else f"任务未完成：{payload['message']}"
                ),
                frame=session.live_frame,
                action="COMPLETE" if succeeded else "FAILED",
                pose=session.pose,
                linear=0.0,
                angular=0.0,
                waypoint=max(0, len(path) - 1),
                waypoint_count=max(0, len(path) - 1),
                result=payload,
            )
            source = overview_camera or camera
            final_image = camera_rgb(source) if source is not None else None
            if final_image is not None:
                live.publish_image(final_image)
        app_utils.stop()
        print(
            "Dual-agent result: "
            f"status={payload['status']} "
            f"pre_vla_pipeline_succeeded={payload['pre_vla_pipeline_succeeded']} "
            f"openvla_inference_succeeded={payload['openvla_inference_succeeded']} "
            f"same_simulation_app={payload['same_simulation_app']}"
        )
        print(f"Summary: {summary_path}")
        if args.openvla:
            return (
                0
                if payload["pre_vla_pipeline_succeeded"]
                and payload["openvla_inference_succeeded"]
                else 5
            )
        return 0 if payload["pre_vla_pipeline_succeeded"] else 4

    follower = PathFollower(
        path,
        goal_yaw=final_yaw,
        max_linear=0.42 if args.survey else 0.45,
        max_angular=1.10,
    )
    if live is not None:
        live.publish_state(
            state="running",
            message="家庭场景已就绪，机器人开始导航。",
            frame=0,
            action="start",
            pose=pose,
            linear=0.0,
            angular=0.0,
            waypoint=follower.index,
            waypoint_count=max(0, len(path) - 1),
        )
    if args.start_hold_seconds > 0:
        hold_until = time.monotonic() + args.start_hold_seconds
        while simulation_app.is_running() and time.monotonic() < hold_until:
            simulation_app.update()

    frame = 0
    last_label = "start"
    reverse_motion_frames = 0
    previous_xy = np.asarray([pose.x, pose.y], dtype=np.float64)
    while simulation_app.is_running():
        observed = robot_pose(robot) if args.wheel_physics_only else pose
        linear, angular, label = follower.command(observed)
        robot.apply_wheel_actions(command_to_wheel_velocities(linear, angular))
        if not args.wheel_physics_only:
            pose = assisted_step(pose, linear, angular)
            set_assisted_robot_pose(robot, pose, linear, angular)

        if camera is not None:
            camera.set_world_pose(*camera_world_pose(observed if args.wheel_physics_only else pose), camera_axes="world")
        if live is not None and overview_camera is not None:
            overview_camera.set_world_pose(
                *home_chase_camera_pose(
                    observed if args.wheel_physics_only else pose
                ),
                camera_axes="world",
            )
        simulation_app.update()

        current = robot_pose(robot) if args.wheel_physics_only else pose
        current_xy = np.asarray([current.x, current.y], dtype=np.float64)
        displacement = current_xy - previous_xy
        if np.linalg.norm(displacement) > 1e-5:
            forward = np.asarray(
                [math.cos(current.yaw), math.sin(current.yaw)],
                dtype=np.float64,
            )
            if float(np.dot(displacement, forward)) < -1e-5:
                reverse_motion_frames += 1
        previous_xy = current_xy
        live_due = live is not None and frame % max(
            1, PHYSICS_HZ // args.live_fps
        ) == 0
        gif_due = (
            args.record_gif is not None
            and frame % max(1, PHYSICS_HZ // args.record_fps) == 0
        )
        if live_due:
            live_image = camera_rgb(camera if camera is not None else overview_camera)
            if live_image is not None:
                live.publish_image(live_image)
        if overview_camera is not None and gif_due:
            image = camera_rgb(overview_camera)
            if image is not None:
                from PIL import Image

                gif_frames.append(Image.fromarray(image).copy())
        if live_due:
            live.publish_state(
                state="running",
                message=f"正在导航：{label}，航点 {follower.index}/{len(path) - 1}",
                frame=frame,
                action=label,
                pose=current,
                linear=linear,
                angular=angular,
                waypoint=follower.index,
                waypoint_count=max(0, len(path) - 1),
            )

        if recorder is not None:
            recorder.maybe_capture(observed if args.wheel_physics_only else pose)
        if label != last_label or frame % 180 == 0:
            current = robot_pose(robot) if args.wheel_physics_only else pose
            print(
                f"frame={frame:4d} state={label:7s} waypoint={follower.index}/{len(path)-1} "
                f"pose=({current.x:+.2f},{current.y:+.2f},{current.yaw:+.2f}) "
                f"cmd=({linear:+.2f},{angular:+.2f})"
            )
            last_label = label
        frame += 1
        if follower.done:
            break
        if args.steps > 0 and frame >= args.steps:
            break

    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    app_utils.update_app(steps=5)
    if args.viewport_screenshot is not None and not args.headless:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            print("WARNING: no active Isaac viewport; final screenshot was not captured")
        else:
            args.viewport_screenshot.parent.mkdir(parents=True, exist_ok=True)
            capture_task = asyncio.ensure_future(
                capture_viewport_to_file(
                    viewport,
                    file_path=str(args.viewport_screenshot),
                    is_hdr=False,
                ).wait_for_result()
            )
            while simulation_app.is_running() and not capture_task.done():
                simulation_app.update()
            if capture_task.done() and capture_task.exception() is None:
                print(f"Arrival viewport: {args.viewport_screenshot}")
            else:
                print("WARNING: final Isaac viewport capture failed")
    if args.arrival_hold_seconds > 0:
        hold_until = time.monotonic() + args.arrival_hold_seconds
        while simulation_app.is_running() and time.monotonic() < hold_until:
            simulation_app.update()
    final_pose = robot_pose(robot) if args.wheel_physics_only else pose
    goal = path[-1]
    position_error = math.dist((final_pose.x, final_pose.y), goal)
    yaw_error = abs(math.atan2(math.sin(final_yaw - final_pose.yaw), math.cos(final_yaw - final_pose.yaw)))
    manifest = None
    if recorder is not None:
        recorder.maybe_capture(final_pose, force=True)
        manifest = recorder.finish()
        print(f"Survey frames: {len(recorder.frames)} -> {manifest}")
    elif camera is not None:
        arrival_image = args.output_dir / "arrival_rgb.png"
        if save_camera_rgb(camera, arrival_image):
            print(f"Arrival RGB: {arrival_image}")

    result = {
        "scene_profile": args.scene_profile,
        "scene": (
            FAMILY_HOME_SCENE_NAME
            if args.scene_profile == "family-home"
            else "SimpleRoom+SofaTablePlant"
        ),
        "task": task_name,
        "command": None if args.survey else args.command,
        "map_source": map_source,
        "map_path": str(args.map) if map_source.startswith("lingbot") else None,
        "places_path": str(args.places) if map_source.startswith("lingbot") else None,
        "execution_mode": "wheel_physics_only" if args.wheel_physics_only else "stable_assisted",
        "success": follower.done and reverse_motion_frames == 0,
        "frames": frame,
        "path_length_m": path_length(path),
        "final_pose": {"x": final_pose.x, "y": final_pose.y, "yaw": final_pose.yaw},
        "position_error_m": position_error,
        "yaw_error_rad": yaw_error,
        "reverse_motion_frames": reverse_motion_frames,
        "forward_only_verified": reverse_motion_frames == 0,
        "survey_manifest": str(manifest) if manifest else None,
        "navigation_gif": str(args.record_gif) if args.record_gif else None,
    }
    if args.record_gif is not None:
        if not gif_frames:
            raise RuntimeError("overview camera produced no frames for --record-gif")
        args.record_gif.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            args.record_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=round(1000.0 / args.record_fps),
            loop=0,
            optimize=False,
        )
        print(f"Navigation GIF: {args.record_gif} ({len(gif_frames)} frames)")
    summary_path = args.output_dir / ("survey_summary.json" if args.survey else "run_summary.json")
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if live is not None:
        live.publish_state(
            state=(
                "succeeded"
                if follower.done and reverse_motion_frames == 0
                else "failed"
            ),
            message=(
                f"已到达 {task_name}，位置误差 {position_error:.3f} m。"
                if follower.done and reverse_motion_frames == 0
                else "检测到反向运动，正向一致性门已拒绝本次导航。"
                if reverse_motion_frames
                else f"任务结束但未到达目标，位置误差 {position_error:.3f} m。"
            ),
            frame=frame,
            action=(
                "arrived"
                if follower.done and reverse_motion_frames == 0
                else "failed"
            ),
            pose=final_pose,
            linear=0.0,
            angular=0.0,
            waypoint=follower.index,
            waypoint_count=max(0, len(path) - 1),
            result=result,
        )
    print(
        f"Result: success={result['success']} position_error={position_error:.3f} m "
        f"yaw_error={yaw_error:.3f} rad"
    )
    print(f"Summary: {summary_path}")

    app_utils.stop()
    if args.test:
        if not follower.done:
            print("TEST FAILED: navigation did not finish", file=sys.stderr)
            return 2
        if position_error > 0.20:
            print("TEST FAILED: final position error is too large", file=sys.stderr)
            return 3
        if reverse_motion_frames:
            print(
                "TEST FAILED: forward-only gate detected reverse motion",
                file=sys.stderr,
            )
            return 6
        print(
            "TEST PASSED: G1-D reached the reviewed "
            f"{args.scene_profile} destination"
        )
    return 0


try:
    exit_code = main()
except Exception as exc:
    if args.live_dir is not None:
        publish_failure(
            args.live_dir,
            command=args.command,
            message=f"{type(exc).__name__}: {exc}",
            pose=FAMILY_HOME_START,
        )
    raise
finally:
    simulation_app.close()

raise SystemExit(exit_code)
