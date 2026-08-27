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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
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
# The original family asset was copied without its `textures/` directory.  The
# Expert package contains a USDA wrapper that clears only those broken texture
# inputs while retaining the same SofaTablePlant geometry and transforms.
_SOFA_USD_FIXED = (
    ROOT
    / "g1d-expert-MaChuanhao/g1d-expert/scene/assets"
    / "SofaTablePlant_no_missing_textures.usda"
)
SOFA_USD = _SOFA_USD_FIXED if _SOFA_USD_FIXED.is_file() else (
    ROOT / "Assets/room/GenieSim/scenes/iros/SofaTablePlant.usd"
)
DEFAULT_OUTPUT = ROOT / "outputs/simple_room_vln"
DEFAULT_LINGBOT_MAP = DEFAULT_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_FORMAL_PLACES = DEFAULT_OUTPUT / "places_formal.json"
DEFAULT_HOME_OUTPUT = ROOT / "outputs/family_home_vln"
DEFAULT_HOME_LINGBOT_MAP = DEFAULT_HOME_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_HOME_FORMAL_PLACES = DEFAULT_HOME_OUTPUT / "places_formal.json"
DEFAULT_HOME_FORMAL_OBJECTS = DEFAULT_HOME_OUTPUT / "objects_formal.json"
LIVING_ROOM_USD = ROOT / "scene_asset/living_room/home_lab.usda"
DEFAULT_LIVING_ROOM_OUTPUT = ROOT / "outputs/living_room_vln"
DEFAULT_LIVING_ROOM_LINGBOT_MAP = DEFAULT_LIVING_ROOM_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_LIVING_ROOM_FORMAL_PLACES = DEFAULT_LIVING_ROOM_OUTPUT / "places_formal.json"
DEFAULT_LIVING_ROOM_FORMAL_OBJECTS = DEFAULT_LIVING_ROOM_OUTPUT / "objects_formal.json"
CGS_OFFICE_USD = ROOT / "scene_asset/cgs_office/cgs_office.usda"
DEFAULT_CGS_OFFICE_OUTPUT = ROOT / "outputs/cgs_office_vln"
DEFAULT_CGS_OFFICE_MAP = ROOT / "outputs/CGS/8181_new/forma_8181_pipeline/maps/map.yaml"
DEFAULT_CGS_OFFICE_PLACES = DEFAULT_CGS_OFFICE_OUTPUT / "places_formal.json"
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
CAMERA_HEIGHT_ABOVE_FLOOR_M = 1.72
CAMERA_FORWARD_OFFSET_M = 0.25
CAMERA_YAW_OFFSET_RAD = math.radians(-15.0)
# With the declared +X-forward, +Z-up camera frame, a positive rotation about
# local +Y pitches the optical axis toward -Z (the floor).
CAMERA_DOWNWARD_PITCH_RAD = math.radians(61.0)
CAMERA_FOCAL_LENGTH_MM = 50.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
CAMERA_VERTICAL_APERTURE_MM = 15.71625
CAMERA_NEAR_CLIP_M = 0.1
CAMERA_FAR_CLIP_M = 1_000_000.0
OVERVIEW_EYE = (3.25, -2.60, 1.75)
OVERVIEW_TARGET = (-1.25, 0.25, -0.20)
HOME_OVERVIEW_EYE = (6.25, -6.00, 6.20)
HOME_OVERVIEW_TARGET = (0.0, 0.70, -0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-profile",
        choices=("simple-room", "family-home", "living-room", "cgs-office"),
        default="simple-room",
        help="Use the original single room or the multi-zone family-home layout",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--interactive-port",
        type=int,
        help=(
            "Serve a local command page and execute submitted SimpleRoom navigation "
            "commands in this same Isaac SimulationApp."
        ),
    )
    parser.add_argument(
        "--interactive-host",
        default="127.0.0.1",
        help="Bind host for --interactive-port (default: 127.0.0.1).",
    )
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
        "--mission-json",
        type=Path,
        default=None,
        help="pre-compiled mission JSON file (bypasses --command compilation)",
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
        "--grasp-calibration",
        action="store_true",
        help=(
            "run only the fixed-pose Family Home physical cup-grasp "
            "calibration (no VLN, RGB search, LLM, or OpenVLA inference)"
        ),
    )
    parser.add_argument(
        "--collect-grasp-demos",
        type=int,
        default=0,
        metavar="N",
        help=(
            "collect N expert grasp demonstrations using the multi-stage IK "
            "pipeline; saves (RGB, 7-D delta action, instruction) tuples in "
            "OpenVLA-compatible format under --output-dir/grasp_demos/"
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
        "--execute-openvla-actions",
        action="store_true",
        help=(
            "apply the OpenVLA-OFT 8x7 world-frame Cartesian action chunk "
            "to the bounded G1-D right-arm IK controller in Isaac Sim only"
        ),
    )
    parser.add_argument(
        "--expert-pick",
        action="store_true",
        help=(
            "after OpenVLA inference, execute the DLS-IK expert pick-lift-drop "
            "pipeline (MaChuanhao controller); mutually exclusive with "
            "--execute-sim-pick"
        ),
    )
    parser.add_argument(
        "--expert-config",
        type=Path,
        default=None,
        help="override the default cup expert config JSON path",
    )
    parser.add_argument(
        "--record-expert-demo",
        action="store_true",
        help=(
            "record the verified in-process Expert control steps as aligned "
            "RGB + 7-D delta-action training pairs; failed episodes are "
            "quarantined outside the training episode glob"
        ),
    )
    parser.add_argument(
        "--openvla-model",
        type=Path,
        default=DEFAULT_OPENVLA_MODEL,
    )
    parser.add_argument(
        "--openvla-adapter",
        type=Path,
        default=None,
        help="optional PEFT/LoRA adapter loaded on top of --openvla-model",
    )
    parser.add_argument(
        "--openvla-action-head",
        type=Path,
        default=None,
        help="optional OpenVLA-OFT continuous L1 action-head checkpoint",
    )
    parser.add_argument(
        "--openvla-dataset-statistics",
        type=Path,
        default=None,
        help="normalization statistics from the OpenVLA-OFT training run",
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
    parser.add_argument(
        "--assisted-motion-scale",
        type=float,
        default=1.0,
        help=(
            "Scale assisted-demo base motion per rendered frame; does not "
            "change wheel-physics-only mode or task safety tolerances"
        ),
    )
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
for openvla_path_name in (
    "openvla_model",
    "openvla_python",
    "openvla_adapter",
    "openvla_action_head",
    "openvla_dataset_statistics",
):
    openvla_path = getattr(args, openvla_path_name)
    if openvla_path is not None and not openvla_path.is_absolute():
        setattr(args, openvla_path_name, ROOT / openvla_path)
if args.expert_config is not None and not args.expert_config.is_absolute():
    args.expert_config = ROOT / args.expert_config
if args.expert_config is not None and not args.expert_config.is_file():
    raise SystemExit(f"--expert-config does not exist: {args.expert_config}")
if args.scene_profile == "family-home":
    if args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_HOME_OUTPUT
    if args.map == DEFAULT_LINGBOT_MAP:
        args.map = DEFAULT_HOME_LINGBOT_MAP
    if args.places == DEFAULT_FORMAL_PLACES:
        args.places = DEFAULT_HOME_FORMAL_PLACES
    if args.objects == DEFAULT_HOME_FORMAL_OBJECTS:
        args.objects = DEFAULT_HOME_FORMAL_OBJECTS
elif args.scene_profile == "living-room":
    if not LIVING_ROOM_USD.is_file():
        raise FileNotFoundError(LIVING_ROOM_USD)
    if args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_LIVING_ROOM_OUTPUT
    if args.map == DEFAULT_LINGBOT_MAP:
        args.map = DEFAULT_LIVING_ROOM_LINGBOT_MAP
    if args.places == DEFAULT_FORMAL_PLACES:
        args.places = DEFAULT_LIVING_ROOM_FORMAL_PLACES
    if args.objects == DEFAULT_HOME_FORMAL_OBJECTS:
        args.objects = DEFAULT_LIVING_ROOM_FORMAL_OBJECTS
elif args.scene_profile == "cgs-office":
    if not CGS_OFFICE_USD.is_file():
        raise FileNotFoundError(CGS_OFFICE_USD)
    if args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_CGS_OFFICE_OUTPUT
    if args.map == DEFAULT_LINGBOT_MAP:
        args.map = DEFAULT_CGS_OFFICE_MAP
    if args.places == DEFAULT_FORMAL_PLACES:
        args.places = DEFAULT_CGS_OFFICE_PLACES
    if args.objects == DEFAULT_HOME_FORMAL_OBJECTS:
        args.objects = DEFAULT_CGS_OFFICE_OUTPUT / "objects_formal.json"
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
if args.execute_openvla_actions and not args.openvla:
    raise SystemExit("--execute-openvla-actions requires --openvla")
if args.expert_pick and not args.openvla:
    raise SystemExit("--expert-pick requires --openvla")
if sum(
    bool(value)
    for value in (
        args.expert_pick,
        args.execute_sim_pick,
        args.execute_openvla_actions,
    )
) > 1:
    raise SystemExit(
        "--expert-pick, --execute-sim-pick and --execute-openvla-actions "
        "are mutually exclusive"
    )
if args.family_task and not args.dual_agent:
    raise SystemExit("--family-task requires --dual-agent")
if args.right_arm_probe and args.dual_agent:
    raise SystemExit("--right-arm-probe cannot be combined with --dual-agent")
if args.grasp_calibration and (
    args.dual_agent
    or args.family_task
    or args.interactive_port is not None
    or args.collect_grasp_demos
):
    raise SystemExit(
        "--grasp-calibration is a standalone physical-grasp mode and cannot "
        "be combined with navigation, dashboard, or data collection flags"
    )
if args.grasp_calibration and args.scene_profile != "family-home":
    raise SystemExit("--grasp-calibration requires --scene-profile family-home")
if args.collect_grasp_demos > 0:
    if args.scene_profile != "family-home":
        raise SystemExit("--collect-grasp-demos requires --scene-profile family-home")
    if args.dual_agent or args.survey or args.right_arm_probe:
        raise SystemExit("--collect-grasp-demos is a standalone mode")
if args.openvla_timeout_sec <= 0.0:
    raise SystemExit("--openvla-timeout-sec must be positive")
if not 0.5 <= args.assisted_motion_scale <= 3.0:
    raise SystemExit("--assisted-motion-scale must be between 0.5 and 3.0")
if args.openvla and not args.openvla_python.is_file():
    sidecar_python = ROOT / ".conda/envs/vln/bin/python"
    if sidecar_python.is_file():
        # The legacy OpenVLA venv's launcher can point at a host-only Conda
        # installation. Use the project-local isolated inference runtime when
        # it is available, never Isaac Kit's in-process Python.
        args.openvla_python = sidecar_python

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
if args.interactive_port is not None and not 1 <= args.interactive_port <= 65535:
    raise SystemExit("--interactive-port must be between 1 and 65535")
if args.interactive_port is not None and (args.survey or args.right_arm_probe):
    raise SystemExit(
        "--interactive-port cannot be combined with --survey or --right-arm-probe"
    )
if (
    args.interactive_port is not None
    and args.scene_profile == "family-home"
    and args.allow_bootstrap
):
    raise SystemExit(
        "family-home interactive navigation requires the reviewed formal map; "
        "do not pass --allow-bootstrap"
    )
if args.dual_agent and args.scene_profile != "family-home":
    raise SystemExit("--dual-agent currently requires --scene-profile family-home")

if args.scene_profile == "living-room":
    required_assets = (ROBOT_USD,)
elif args.scene_profile == "cgs-office":
    required_assets = (CGS_OFFICE_USD, ROBOT_USD)
else:
    required_assets = (ROOM_USD, ROBOT_USD, SOFA_USD)
for required in required_assets:
    if not required.is_file():
        raise FileNotFoundError(required)
if args.scene_profile == "family-home":
    from family_home_vln.household_objects import require_prepared_assets

    require_prepared_assets()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

try:
    RENDER_GPU_COUNT = int(os.environ.get("G1D_RENDER_GPU_COUNT", "1"))
except ValueError as exc:
    raise SystemExit("G1D_RENDER_GPU_COUNT must be an integer") from exc
if not 1 <= RENDER_GPU_COUNT <= 8:
    raise SystemExit("G1D_RENDER_GPU_COUNT must be between 1 and 8")
try:
    ACTIVE_RENDER_GPU = int(os.environ.get("G1D_ACTIVE_GPU", "0"))
except ValueError as exc:
    raise SystemExit("G1D_ACTIVE_GPU must be an integer") from exc
if not 0 <= ACTIVE_RENDER_GPU <= 7:
    raise SystemExit("G1D_ACTIVE_GPU must be between 0 and 7")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1920,
        "height": 1080,
        # Do not use CUDA_VISIBLE_DEVICES with Isaac Sim: Omniverse and CUDA
        # enumerate devices differently, which prevents GPU Foundation (and
        # therefore RGB render products) from initializing.  This native Kit
        # setting uses the physical GPU index reported by nvidia-smi.
        "active_gpu": ACTIVE_RENDER_GPU,
        "multi_gpu": RENDER_GPU_COUNT > 1,
        "max_gpu_count": RENDER_GPU_COUNT,
    }
)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

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
from family_home_vln.live import LivePublisher, publish_failure
from simple_room_vln.artifacts import (
    ROBOT_RADIUS_M,
    SOFA_SET_TRANSLATION,
    build_bootstrap_artifacts,
    load_lingbot_artifacts,
)
from simple_room_vln.core import PathFollower, Pose2D, path_length, resolve_place

# Center of the largest footprint-safe component at 0.3 m (8181_new scan,
# 15x19 m multi-room office; auto-picked by build_cgs_office_places.py).
CGS_OFFICE_START = Pose2D(9.53, 14.11, 0.0)


def command_to_wheel_velocities(linear_speed: float, angular_speed: float) -> np.ndarray:
    left = (linear_speed - angular_speed * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
    right = -(linear_speed + angular_speed * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
    return np.array([left, right], dtype=np.float32)


def configure_joint_drives(robot: WheeledRobot) -> None:
    names = robot.dof_names
    stiffness = np.zeros(len(names), dtype=np.float32)
    damping = np.zeros(len(names), dtype=np.float32)
    # G1-D torso chain (AGV_link → LZ_mt → LZ_it → Yaw → torso).
    # No Pitching_Joint exists in this variant — the upper body is rigid.
    TORSO_JOINTS = {
        "LZ_mt_Joint",
        "LZ_it_Joint",
        "Yaw_Joint",
        "torso_Joint",
    }
    for index, name in enumerate(names):
        if name in (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT):
            damping[index] = 20.0
        elif name in TORSO_JOINTS:
            stiffness[index] = 2000.0
            damping[index] = 150.0
        elif "hand_" in name:
            stiffness[index] = 40.0
            damping[index] = 3.0
        else:
            stiffness[index] = 80.0
            damping[index] = 8.0
    robot.set_dof_gains(stiffnesses=stiffness, dampings=damping)
    wheel_indices = robot.get_dof_indices([LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]).numpy().tolist()
    robot.set_dof_max_efforts([40.0, 40.0], dof_indices=wheel_indices)
    hand_names = [name for name in names if "hand_" in name]
    if hand_names:
        hand_indices = robot.get_dof_indices(hand_names).numpy().tolist()
        robot.set_dof_max_efforts(
            [12.0] * len(hand_indices), dof_indices=hand_indices
        )
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
        positions=np.array(
            [pose.x, pose.y, 0.1055 if args.scene_profile in ("living-room", "cgs-office") else ROBOT_ROOT_ON_FLOOR_Z_M],
            dtype=np.float32,
        ),
        orientations=orientation,
    )
    robot.set_velocities(
        linear_velocities=[linear * math.cos(pose.yaw), linear * math.sin(pose.yaw), 0.0],
        angular_velocities=[0.0, 0.0, angular],
    )
    hold_left_arm_vertical(robot)


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
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
# In this G1-D asset, a zero elbow points the forearm horizontally forward.
# The upper arm already points down at zero shoulder angles; +pi/2 around the
# elbow Y axis rotates the forearm from +X to world-down, so both arm segments
# are vertical instead of the unused hand projecting toward the table.
LEFT_ARM_VERTICAL_RAD = np.asarray(
    [0.0, 0.0, 0.0, math.pi / 2.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
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
# Right-arm pregrasp: upper arm raised FORWARD (shoulder pitch ≈-1.2 rad
# ≈69° forward from vertical) with elbow bent ~30°.  Negative shoulder
# pitch rotates the arm anterior (in front of the body), positive rotates
# posterior (behind).  The expert DLS-IK controller's "raise" phase fine-
# tunes the hand to the correct safe-Z above the table from this seed.
RIGHT_ARM_PREGRASP_SEED_RAD = np.asarray(
    [-1.2, -0.20, 0.0, 0.55, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
# Exact high-reach initialization used by every accepted physical-v14 demo.
# Direct OFT rollout must begin from this state; the older dashboard staging
# seed above is a different observation/control distribution.
RIGHT_ARM_V14_HIGH_REACH_RAD = np.asarray(
    [-0.80, -0.32, -0.30, 1.80, 0.0, -0.50, 0.0],
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
RIGHT_FINGERTIP_LINKS = (
    "right_hand_thumb_2_link",
    "right_hand_middle_1_link",
    "right_hand_index_1_link",
)
RIGHT_HAND_OPEN_RAD = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
RIGHT_HAND_CLOSED_RAD = np.asarray(
    [0.65, 0.25, -1.32, 1.34, 1.35, 0.75, 0.75],
    dtype=np.float64,
)
# URDF palm-to-fingertip envelope: 0.0777 + 0.0458 + 0.0263 m.
RIGHT_HAND_FINGERTIP_REACH_M = 0.16


def _write_left_arm_vertical_pose(robot: WheeledRobot, positions: np.ndarray) -> None:
    """Write the reviewed unused-left-arm pose into a full DOF vector."""

    indices = robot.get_dof_indices(list(LEFT_ARM_JOINTS)).numpy().tolist()
    positions[indices] = LEFT_ARM_VERTICAL_RAD


def hold_left_arm_vertical(robot: WheeledRobot) -> None:
    """Keep the unused left upper arm and forearm vertical during motion."""

    indices = robot.get_dof_indices(list(LEFT_ARM_JOINTS)).numpy().tolist()
    robot.set_dof_position_targets(LEFT_ARM_VERTICAL_RAD, dof_indices=indices)


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


def link_world_orientation_xyzw(
    robot: WheeledRobot, link_name: str
) -> np.ndarray:
    link_index = int(robot.get_link_indices(link_name).numpy()[0])
    transforms = articulation_link_transforms(robot)
    return np.asarray(transforms[0, link_index, 3:7], dtype=np.float64)


def _quaternion_rotation_error_xyzw(
    target_xyzw: np.ndarray, current_xyzw: np.ndarray
) -> np.ndarray:
    """Return the shortest world-frame quaternion error as a rotation vector."""

    tx, ty, tz, tw = target_xyzw
    cx, cy, cz, cw = current_xyzw
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


def _left_facing_palm_grasp_orientation_xyzw(
    planar_direction: np.ndarray,
) -> np.ndarray:
    """Orient palm toward the cup, +Y facing left for a side grasp."""

    x_axis = np.asarray(
        [planar_direction[0], planar_direction[1], 0.0], dtype=np.float64
    )
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)  # LEFT
    matrix = np.column_stack([x_axis, y_axis, z_axis])
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


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
    target_orientation_xyzw: Sequence[float] | None = None,
    orientation_tolerance_rad: float = 0.18,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Move the right palm with bounded position-only DLS IK."""

    target = np.asarray(target_world_m, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("right-palm target must be a finite xyz vector")
    arm_indices = robot.get_dof_indices(list(RIGHT_ARM_JOINTS)).numpy().tolist()
    orientation_target = (
        np.asarray(target_orientation_xyzw, dtype=np.float64)
        if target_orientation_xyzw is not None
        else None
    )
    if orientation_target is not None:
        if orientation_target.shape != (4,):
            raise ValueError("right-palm orientation must be xyzw")
        orientation_target /= np.linalg.norm(orientation_target)
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
        orientation_error = (
            _quaternion_rotation_error_xyzw(
                orientation_target,
                link_world_orientation_xyzw(robot, RIGHT_PALM_LINK),
            )
            if orientation_target is not None
            else np.zeros(3, dtype=np.float64)
        )
        orientation_error_norm = float(np.linalg.norm(orientation_error))
        if (
            error_norm <= tolerance_m
            and (
                orientation_target is None
                or orientation_error_norm <= orientation_tolerance_rad
            )
        ):
            break
        jacobian = robot.get_jacobian_matrices().numpy()[0]
        row = _jacobian_link_row(robot, jacobian, RIGHT_PALM_LINK)
        columns = _jacobian_dof_columns(robot, jacobian, arm_indices)
        position_jacobian = np.asarray(
            jacobian[row, :3, :][:, columns],
            dtype=np.float64,
        )
        if orientation_target is not None:
            orientation_weight = 0.35
            angular_jacobian = np.asarray(
                jacobian[row, 3:6, :][:, columns], dtype=np.float64
            )
            controlled_jacobian = np.vstack(
                [position_jacobian, orientation_weight * angular_jacobian]
            )
            controlled_error = np.concatenate(
                [error, orientation_weight * orientation_error]
            )
        else:
            controlled_jacobian = position_jacobian
            controlled_error = error
        damping = 0.055
        dimension = controlled_jacobian.shape[0]
        delta = controlled_jacobian.T @ np.linalg.solve(
            controlled_jacobian @ controlled_jacobian.T
            + (damping**2) * np.eye(dimension),
            controlled_error,
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
    final_orientation_error = (
        float(
            np.linalg.norm(
                _quaternion_rotation_error_xyzw(
                    orientation_target,
                    link_world_orientation_xyzw(robot, RIGHT_PALM_LINK),
                )
            )
        )
        if orientation_target is not None
        else None
    )
    return {
        "success": bool(
            final_error <= tolerance_m
            and (
                final_orientation_error is None
                or final_orientation_error <= orientation_tolerance_rad
            )
        ),
        "controller": "bounded_damped_least_squares_position_ik",
        "target_world_m": target.tolist(),
        "start_world_m": start.tolist(),
        "final_world_m": final.tolist(),
        "requested_travel_m": requested_travel,
        "final_error_m": final_error,
        "orientation_controlled": orientation_target is not None,
        "final_orientation_error_rad": final_orientation_error,
        "iterations": len(errors),
        "minimum_iteration_error_m": min(errors) if errors else None,
        "maximum_joint_step_rad": maximum_step,
        "joint_limits_checked": True,
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


def _select_physical_cup_grasp_targets(
    robot: WheeledRobot, object_prim
) -> tuple[np.ndarray, dict]:
    """Select an opposing-finger grasp pose without moving the staged cup."""

    indices = robot.get_dof_indices(list(RIGHT_HAND_JOINTS)).numpy().tolist()
    rigid_body = UsdPhysics.RigidBodyAPI(object_prim)
    rigid_body.GetKinematicEnabledAttr().Set(True)
    thumb_options = (
        (-0.70, -0.45, -1.20),
        (-0.70, 0.45, -1.20),
        (0.70, -0.45, -1.20),
        (0.70, 0.45, -1.20),
        (0.65, 0.45, -1.25),
        (0.45, 0.35, -1.10),
    )
    finger_options = (
        (0.35, 0.55),
        (0.65, 0.85),
        (0.95, 1.10),
        (1.20, 1.35),
    )
    candidates = []
    for thumb in thumb_options:
        for proximal, distal in finger_options:
            targets = np.asarray(
                [*thumb, proximal, distal, proximal, distal],
                dtype=np.float64,
            )
            robot.set_dof_position_targets(targets, dof_indices=indices)
            app_utils.update_app(steps=12)
            object_position = _prim_world_position(object_prim)
            distances = {
                name: float(
                    np.linalg.norm(
                        link_world_position(robot, name) - object_position
                    )
                )
                for name in RIGHT_FINGERTIP_LINKS
            }
            opposing_finger = min(
                distances["right_hand_middle_1_link"],
                distances["right_hand_index_1_link"],
            )
            score = distances["right_hand_thumb_2_link"] + opposing_finger
            candidates.append(
                {
                    "score_m": score,
                    "targets_rad": targets.tolist(),
                    "actual_rad": robot.get_dof_positions().numpy()[
                        0, indices
                    ].astype(np.float64).tolist(),
                    "fingertip_distances_m": distances,
                }
            )
    candidates.sort(key=lambda item: item["score_m"])
    selected = candidates[0]
    robot.set_dof_position_targets(RIGHT_HAND_OPEN_RAD, dof_indices=indices)
    app_utils.update_app(steps=20)
    return np.asarray(selected["targets_rad"], dtype=np.float64), {
        "selection_metric": "thumb_tip_plus_nearest_opposing_finger_distance",
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "object_was_kinematic_during_pose_search": True,
    }


def _set_right_arm_pregrasp_seed(
    robot: WheeledRobot,
    *,
    target_rad: Sequence[float] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    indices = robot.get_dof_indices(list(RIGHT_ARM_JOINTS)).numpy().tolist()
    start = robot.get_dof_positions().numpy()[0, indices].astype(np.float64)
    target = np.asarray(
        RIGHT_ARM_PREGRASP_SEED_RAD if target_rad is None else target_rad,
        dtype=np.float64,
    ).copy()
    if target.shape != (len(RIGHT_ARM_JOINTS),) or not np.all(
        np.isfinite(target)
    ):
        raise ValueError("right-arm pregrasp target must be finite 7-D")
    ROLL = 1  # right_shoulder_roll_joint
    total = 60
    roll_steps = 18  # outward first to clear the torso
    for step in range(total):
        if step < roll_steps:
            # Phase 1 ── roll the shoulder outward while keeping other
            # joints close to their start values so the arm does not
            # swing through the body.
            r = (step + 1) / roll_steps
            current = start.copy()
            current[ROLL] = start[ROLL] + r * (target[ROLL] - start[ROLL])
        else:
            # Phase 2 ── move the rest of the arm now that the elbow
            # already clears the torso.
            r = (step + 1 - roll_steps) / (total - roll_steps)
            phase2_start = start.copy()
            phase2_start[ROLL] = target[ROLL]
            current = phase2_start + r * (target - phase2_start)
        robot.set_dof_position_targets(current, dof_indices=indices)
        simulation_app.update()
        if progress_callback is not None:
            progress_callback(step + 1, total)
    actual = robot.get_dof_positions().numpy()[0, indices]
    return {
        "joint_order": list(RIGHT_ARM_JOINTS),
        "target_rad": target.tolist(),
        "actual_rad": np.asarray(actual, dtype=np.float64).tolist(),
        "maximum_error_rad": float(
            np.max(np.abs(np.asarray(actual, dtype=np.float64) - target))
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
                    float(np.linalg.norm(position[:2] - target_world_m[:2])),
                    prim,
                    position,
                )
            )
    if palm is None:
        raise RuntimeError("cannot find the G1-D right-palm rigid body")
    if not candidates:
        raise RuntimeError("no dynamic family object is available for picking")
    candidates.sort(key=lambda item: item[0])
    planar_anchor_error, object_prim, object_position = candidates[0]
    if planar_anchor_error > maximum_object_anchor_error_m:
        raise RuntimeError(
            "nearest dynamic object's horizontal physics pose disagrees with "
            f"the scan-derived anchor by {planar_anchor_error:.3f} m"
        )
    spatial_anchor_error = float(np.linalg.norm(object_position - target_world_m))
    return palm, object_prim, {
        "selection": "nearest_dynamic_body_to_scan_anchor_for_safety_verification",
        "palm_prim_path": str(palm.GetPath()),
        "object_prim_path": str(object_prim.GetPath()),
        "scan_anchor_world_m": target_world_m.tolist(),
        "physics_object_world_m": object_position.tolist(),
        "anchor_error_m": spatial_anchor_error,
        "planar_anchor_error_m": planar_anchor_error,
        "vertical_anchor_error_m": float(
            abs(object_position[2] - target_world_m[2])
        ),
        "maximum_anchor_error_m": maximum_object_anchor_error_m,
        "simulator_semantic_label_read": False,
    }


def _author_physical_fingertip_pads(stage) -> list[str]:
    """Author reviewed distal pads before PhysX parses the articulation."""

    # The imported distal links contain instance-proxy meshes whose collision
    # surface does not coincide with the visible fingertip centre used by the
    # Expert TCP.  Place explicit spherical pads on the two *inner link
    # surfaces*, slightly protruding into the pinch gap.  Putting a sphere at
    # the link centre leaves it buried inside the imported collision mesh and
    # merely recreates the link-centre approximation we are replacing.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link_prims = {}
    for link_name in (
        "right_hand_thumb_2_link",
        "right_hand_middle_1_link",
    ):
        candidates = [
            prim
            for prim in stage.Traverse()
            if prim.GetName() == link_name
            and not prim.IsInstance()
            and not prim.IsInstanceProxy()
        ]
        if not candidates:
            continue
        link_prims[link_name] = min(
            candidates, key=lambda prim: str(prim.GetPath()).count("/")
        )
    if not {
        "right_hand_thumb_2_link",
        "right_hand_middle_1_link",
    }.issubset(link_prims):
        return []

    thumb_link = link_prims["right_hand_thumb_2_link"]
    middle_link = link_prims["right_hand_middle_1_link"]
    boxes = {
        name: bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        for name, prim in link_prims.items()
    }
    centers = {
        name: (
            np.asarray(box.GetMin(), dtype=np.float64)
            + np.asarray(box.GetMax(), dtype=np.float64)
        )
        * 0.5
        for name, box in boxes.items()
    }
    pinch_axis = centers["right_hand_middle_1_link"] - centers[
        "right_hand_thumb_2_link"
    ]
    pinch_axis /= max(float(np.linalg.norm(pinch_axis)), 1.0e-9)
    pad_radius = 0.006
    pad_center_offset = 0.012
    fingertip_pad_paths: list[str] = []
    for link_name, link_prim in link_prims.items():
        pad_path = f"{link_prim.GetPath()}/ExpertPhysicalFingerPad"
        if stage.GetPrimAtPath(pad_path).IsValid():
            fingertip_pad_paths.append(pad_path)
            continue
        world_center = centers[link_name]
        direction_into_gap = (
            pinch_axis
            if link_name == "right_hand_thumb_2_link"
            else -pinch_axis
        )
        # A world-axis-aligned bbox projection returns a corner rather than a
        # material surface and moves vertically as the wrist rotates.  Use a
        # fixed reviewed offset from the distal-link frame instead; the pad's
        # own PhysX sphere is then the authoritative contact surface.
        pad_world_center = world_center + direction_into_gap * pad_center_offset
        local_center = xform_cache.GetLocalToWorldTransform(
            link_prim
        ).GetInverse().Transform(Gf.Vec3d(*pad_world_center))
        pad = UsdGeom.Sphere.Define(stage, pad_path)
        pad.CreateRadiusAttr(pad_radius)
        UsdGeom.Xformable(pad).AddTranslateOp().Set(local_center)
        UsdGeom.Imageable(pad.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        fingertip_pad_paths.append(str(pad.GetPath()))
    return fingertip_pad_paths


def _configure_physical_grasp_friction(object_prim) -> dict:
    """Bind a high-friction physics material to the cup and right fingers."""

    # Item05 is the exact dining cup used by the accepted v14 OpenVLA-OFT
    # demonstrations.  Those trajectories were collected with no authored
    # physics material (PhysX defaults) and no deployment-only fingertip pads.
    # Besides changing the contact distribution, recursively binding a
    # material to the referenced GLB hierarchy can invalidate Fabric data when
    # simulation starts.  Keep deployment identical to the training scene.
    if str(object_prim.GetPath()) == "/World/FamilyHomeObjects/Item05":
        return {
            "material_path": None,
            "static_friction": "physx_default",
            "dynamic_friction": "physx_default",
            "bound_collision_prim_count": 0,
            "bound_collision_prim_paths": [],
            "material_inherits_into_instance_colliders": False,
            "palm_candidate_paths": [],
            "physical_fingertip_pad_paths": [],
            "physical_fingertip_pad_shape": None,
            "training_physics_profile": "expert_demos_head_physical_v14_oft_b",
        }

    stage = stage_utils.get_current_stage()
    fingertip_pad_paths = _author_physical_fingertip_pads(stage)
    palm_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName() == RIGHT_PALM_LINK
    ]
    palm_path = (
        # The shallow prim is the editable rigid-body link; the deeper
        # same-named prim is the referenced visual/collision instance.
        min(set(palm_paths), key=lambda value: value.count("/"))
        if palm_paths
        else ""
    )
    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/G1DPhysicalGrasp"
    )
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(4.0)
    physics_material.CreateDynamicFrictionAttr(3.5)
    physics_material.CreateRestitutionAttr(0.0)
    bound_paths = []
    object_prefix = str(object_prim.GetPath()) + "/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        is_object_collider = (
            path.startswith(object_prefix)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        # The imported distal-finger collision meshes live inside instance
        # prototypes and are not returned by normal Stage.Traverse().  Bind
        # the material to the editable link Xforms as well; USD material
        # inheritance carries it into those instance collision meshes.
        is_right_hand_collision_scope = (
            bool(palm_path)
            and path.startswith(palm_path)
            and not prim.IsInstance()
            and not prim.IsInstanceProxy()
            and (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or prim.GetTypeName() == "Xform"
            )
        )
        if not (is_object_collider or is_right_hand_collision_scope):
            continue
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(material, materialPurpose="physics")
        bound_paths.append(path)
    return {
        "material_path": str(material.GetPath()),
        "static_friction": 4.0,
        "dynamic_friction": 3.5,
        "bound_collision_prim_count": len(bound_paths),
        "bound_collision_prim_paths": bound_paths,
        "material_inherits_into_instance_colliders": True,
        "palm_candidate_paths": palm_paths,
        "physical_fingertip_pad_paths": fingertip_pad_paths,
        "physical_fingertip_pad_shape": "sphere",
        "physical_fingertip_pad_radius_m": 0.006,
        "physical_fingertip_pad_center_offset_m": 0.012,
    }


def _resolve_held_sim_object(object_id: str) -> tuple[object, object, str]:
    """Resolve a physically carried object through verified hand proximity."""

    stage = stage_utils.get_current_stage()
    palms = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == RIGHT_PALM_LINK
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(palms) != 1:
        raise RuntimeError("无法唯一定位 G1-D 右手")
    palm_prim = palms[0]
    palm_position = _prim_world_position(palm_prim)
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            path.startswith("/World/FamilyHomeObjects/")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            distance = float(
                np.linalg.norm(_prim_world_position(prim) - palm_position)
            )
            if distance <= RIGHT_HAND_FINGERTIP_REACH_M + 0.03:
                candidates.append((distance, prim))
    if len(candidates) != 1:
        raise RuntimeError(
            f"右手附近动态物体数量为 {len(candidates)}，无法确认杯子被抓住"
        )
    object_prim = candidates[0][1]
    constraint_path = ""
    if (
        not palm_prim.IsValid()
        or palm_prim.GetName() != RIGHT_PALM_LINK
        or not palm_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        raise RuntimeError("抓取约束未连接到 G1-D 右手")
    if (
        not object_prim.IsValid()
        or not str(object_prim.GetPath()).startswith(
            "/World/FamilyHomeObjects/"
        )
        or not object_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        raise RuntimeError("抓取约束未连接到受控家庭物体")
    return palm_prim, object_prim, constraint_path


def camera_world_pose(
    pose: Pose2D,
    downward_pitch_rad: float = CAMERA_DOWNWARD_PITCH_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(
        [
            pose.x + CAMERA_FORWARD_OFFSET_M * math.cos(pose.yaw),
            pose.y + CAMERA_FORWARD_OFFSET_M * math.sin(pose.yaw),
            (0.0 if args.scene_profile in ("living-room", "cgs-office") else ROOM_FLOOR_Z_M)
            + CAMERA_HEIGHT_ABOVE_FLOOR_M,
        ],
        dtype=np.float32,
    )
    camera_yaw = pose.yaw + CAMERA_YAW_OFFSET_RAD
    cy = math.cos(camera_yaw / 2.0)
    sy = math.sin(camera_yaw / 2.0)
    cp = math.cos(downward_pitch_rad / 2.0)
    sp = math.sin(downward_pitch_rad / 2.0)
    # world_Z(yaw) * local_Y(pitch); positive Y pitch points +X down.
    orientation = np.array(
        [cy * cp, -sy * sp, cy * sp, sy * cp],
        dtype=np.float32,
    )
    return position, orientation


HEAD_LINK_NAME = "head_link"


def _base_pitch_rad(robot: WheeledRobot) -> float:
    """Extract the base-body pitch angle (forward tilt around world +Y).

    The G1-D wheelbase is rear-biased; gravity tips the entire chassis
    forward.  This function reads the root-body quaternion (wxyz format)
    and returns the pitch in radians — positive = forward lean.
    """
    _, orientations = robot.get_world_poses()
    quat = orientations.numpy()[0]  # wxyz
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    # sin(pitch) = 2 * (w*y - z*x)  for wxyz quaternions
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    return float(math.asin(sin_pitch))


def head_camera_pose(
    robot: WheeledRobot,
    downward_pitch_rad: float = CAMERA_DOWNWARD_PITCH_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera rigidly mounted on the robot's head link (egocentric view).

    Returns ``(position_xyz, orientation_wxyz)`` suitable for
    ``camera.set_world_pose()`` so the captured frames reflect the
    robot's actual torso/head posture instead of a fixed base-relative
    offset.

    The base-body forward lean (rear-biased wheelbase) is compensated
    so the camera always points at the intended world-frame downward
    angle regardless of chassis tilt.
    """
    # ── compensate for base-body forward pitch ─────────────────────────
    base_pitch = _base_pitch_rad(robot)
    effective_pitch_rad = downward_pitch_rad - base_pitch
    effective_pitch_rad = max(0.0, min(math.radians(70.0), effective_pitch_rad))

    position = link_world_position(robot, HEAD_LINK_NAME)
    # articulation_link_transforms stores xyzw quaternions.
    ori_xyzw = link_world_orientation_xyzw(robot, HEAD_LINK_NAME)
    hx, hy, hz, hw = ori_xyzw  # xyzw

    # The head-link origin is inside the robot's dark head shell on this USD.
    # Mount the optical center in front of that shell so live RGB observes the
    # scene instead of the robot's own collision/visual geometry.
    head_forward_world = np.asarray(
        [
            1.0 - 2.0 * (hy * hy + hz * hz),
            2.0 * (hx * hy + hw * hz),
            2.0 * (hx * hz - hw * hy),
        ],
        dtype=np.float64,
    )
    position = (
        position.astype(np.float64)
        + 0.20 * head_forward_world
        + np.asarray([0.0, 0.0, 0.025], dtype=np.float64)
    )

    # Additional downward pitch about the head's local +Y axis.
    # positive Y pitch → +X down (same convention as camera_world_pose).
    cp = math.cos(effective_pitch_rad / 2.0)
    sp = math.sin(effective_pitch_rad / 2.0)

    # q_camera_wxyz = q_head_wxyz * q_pitch_wxyz
    #   q_head_wxyz  = [hw, hx, hy, hz]
    #   q_pitch_wxyz = [cp, 0, sp, 0]
    orientation_wxyz = np.array(
        [
            hw * cp - hy * sp,  # w
            hx * cp - hz * sp,  # x
            hw * sp + hy * cp,  # y
            hx * sp + hz * cp,  # z
        ],
        dtype=np.float32,
    )
    return position.astype(np.float32), orientation_wxyz


def update_camera_pose(
    robot: WheeledRobot,
    camera,
    downward_pitch_rad: float = CAMERA_DOWNWARD_PITCH_RAD,
) -> None:
    """Update the RGB camera using the audited G1-D optical mount.

    The USD ``head_link`` is a kinematic link whose origin lies inside the
    dark head-shell collision mesh.  Treating that origin as the optical
    centre makes the shell fill the lower half of captured images, which in
    turn prevents RGB search and creates unusable training examples.  Until
    the asset provides an explicit camera optical-frame, use the calibrated
    base-relative mount shared by map collection and the live-search gate.
    """
    pose = robot_pose(robot)
    pos, ori = camera_world_pose(pose, downward_pitch_rad)
    camera.set_world_pose(pos, ori, camera_axes="world")


def aim_head_camera_at_world_point(
    robot: WheeledRobot,
    camera,
    target_world: Sequence[float],
) -> None:
    """Keep the simulated head RGB optical centre aimed at a world target.

    The normal navigation camera has a fixed downward pitch.  At the final
    manipulation standoff that pitch can put a tabletop object below the
    image even though it was visible at the more distant VLA handoff pose.
    Expert demonstrations therefore use the same head-mounted optical centre
    with a target-tracking gaze, matching the view that the learned
    manipulation policy will receive at inference time.
    """

    # The Expert controller's support-joint calibration can place the USD
    # ``head_link`` origin below the tabletop even though the intended RGB
    # sensor mount is at G1-D head height.  Use the same audited base-relative
    # optical centre as camera_world_pose(), while still following the live
    # mobile-base x/y/yaw pose.
    base_pose = robot_pose(robot)
    position = np.asarray(
        [
            base_pose.x
            + CAMERA_FORWARD_OFFSET_M * math.cos(base_pose.yaw),
            base_pose.y
            + CAMERA_FORWARD_OFFSET_M * math.sin(base_pose.yaw),
            ROOM_FLOOR_Z_M + CAMERA_HEIGHT_ABOVE_FLOOR_M,
        ],
        dtype=np.float32,
    )
    target = np.asarray(target_world, dtype=np.float64)
    delta = target - position
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    if float(np.linalg.norm(delta)) < 1e-6 or horizontal_distance < 1e-6:
        raise ValueError("head-camera gaze target is at the optical centre")
    # Use the exact yaw/local-Y-pitch convention already validated by
    # camera_world_pose().  The generic rotation-matrix conversion in
    # look_at_camera_pose() has a different camera-axis convention in its
    # negative-trace branch (the dining-table view is around 100 degrees),
    # which made the sensor look out the window instead of at the cup.
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    downward_pitch = math.atan2(float(-delta[2]), horizontal_distance)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = (
        math.cos(downward_pitch / 2.0),
        math.sin(downward_pitch / 2.0),
    )
    orientation = np.asarray(
        [cy * cp, -sy * sp, cy * sp, sy * cp],
        dtype=np.float32,
    )
    camera.set_world_pose(position, orientation, camera_axes="world")


def aim_wrist_camera_at_world_point(
    robot: WheeledRobot, camera, target_world: Sequence[float],
) -> None:
    """Update the right-palm RGB sensor while retaining a wrist-mounted pose.

    The camera optical centre follows the physical right palm each control
    sample; its gimbal is aimed at the reviewed target so the cup remains in
    view through wrist rotations.  The recorded metadata marks this explicitly
    as a wrist-mounted, target-tracking simulation sensor.
    """
    position = link_world_position(robot, RIGHT_PALM_LINK).astype(np.float64)
    # Optical centre is mounted on a short wrist bracket above the palm;
    # without this 9 cm standoff the camera begins inside the palm/table
    # collision shell and sees only the tabletop underside.
    position += np.asarray([0.035, 0.0, 0.090], dtype=np.float64)
    camera.set_world_pose(
        *look_at_camera_pose(position, target_world), camera_axes="world"
    )


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


def rgb_black_frame_metrics(image: np.ndarray) -> dict[str, float | bool]:
    """Reject large near-black occluders while allowing small dark objects."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        return {
            "black_fraction": 1.0,
            "bottom_black_fraction": 1.0,
            "large_black_frame": True,
        }
    luminance = rgb[..., :3].astype(np.float32).mean(axis=2)
    black = luminance < 12.0
    bottom = black[int(black.shape[0] * 0.55) :]
    black_fraction = float(black.mean())
    bottom_black_fraction = float(bottom.mean()) if bottom.size else 1.0
    return {
        "black_fraction": black_fraction,
        "bottom_black_fraction": bottom_black_fraction,
        "large_black_frame": bool(
            black_fraction > 0.18 or bottom_black_fraction > 0.45
        ),
    }


def add_composed_scene(
    path: Sequence[tuple[float, float]], target: Pose2D | None
) -> None:
    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    if args.scene_profile == "living-room":
        stage_utils.add_reference_to_stage(
            str(LIVING_ROOM_USD).replace("\\", "/"), "/World/LivingRoom"
        )
    elif args.scene_profile == "cgs-office":
        stage_utils.add_reference_to_stage(
            str(CGS_OFFICE_USD).replace("\\", "/"), "/World/CgsOffice"
        )
    else:
        stage_utils.add_reference_to_stage(str(ROOM_USD).replace("\\", "/"), "/World/Room")
        stage_utils.add_reference_to_stage(str(SOFA_USD).replace("\\", "/"), "/World/SofaSet")
    stage = stage_utils.get_current_stage()
    light = UsdLux.DomeLight.Define(stage, "/World/VLN/DomeLight")
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    if args.scene_profile not in ("living-room", "cgs-office"):
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
            # The back-lit dining-table fascia rendered almost pure black and
            # occupied the lower third of the training RGB.  A complete PBR
            # material (with a small ambient/emissive term) keeps every face
            # at the reviewed fixture colour regardless of view direction.
            fixture_material = UsdShade.Material.Define(
                stage, f"/World/Looks/Fixture_{fixture.fixture_id}"
            )
            fixture_shader = UsdShade.Shader.Define(
                stage,
                f"/World/Looks/Fixture_{fixture.fixture_id}/PreviewSurface",
            )
            fixture_shader.CreateIdAttr("UsdPreviewSurface")
            fixture_shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*fixture.color_rgb))
            fixture_shader.CreateInput(
                "emissiveColor", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*(float(v) * 0.12 for v in fixture.color_rgb)))
            fixture_shader.CreateInput(
                "roughness", Sdf.ValueTypeNames.Float
            ).Set(0.72)
            fixture_material.CreateSurfaceOutput().ConnectToSource(
                fixture_shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                fixture_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
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
            if fixture.fixture_id == "dining_table":
                # Keep this reviewed full-volume prim as the support/collision
                # contract, but do not render it as a solid cabinet-like box.
                # That box was the large black trapezoid in the lower 30% of
                # every head-camera frame.  Render a conventional thin top
                # and four legs as non-colliding children instead.
                UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
                table_sx, table_sy, table_sz = fixture.size_xyz
                top_thickness = 0.075
                leg_width = 0.075
                leg_height = table_sz - top_thickness
                physical_top = UsdGeom.Cube.Define(
                    stage, "/World/FamilyHome/DiningTablePhysicalTop"
                )
                physical_top.CreateSizeAttr(1.0)
                physical_top_xform = UsdGeom.Xformable(physical_top)
                physical_top_xform.AddTranslateOp().Set(
                    Gf.Vec3d(
                        fixture.center_xy[0],
                        fixture.center_xy[1],
                        ROOM_FLOOR_Z_M
                        + table_sz
                        - top_thickness / 2.0,
                    )
                )
                physical_top_xform.AddScaleOp().Set(
                    Gf.Vec3f(table_sx, table_sy, top_thickness)
                )
                UsdGeom.Imageable(physical_top.GetPrim()).MakeInvisible()
                UsdPhysics.CollisionAPI.Apply(physical_top.GetPrim())
                visual_parts: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
                    (
                        "DiningTableTop",
                        (
                            fixture.center_xy[0],
                            fixture.center_xy[1],
                            ROOM_FLOOR_Z_M + table_sz - top_thickness / 2.0,
                        ),
                        (table_sx, table_sy, top_thickness),
                    )
                ]
                for x_sign in (-1.0, 1.0):
                    for y_sign in (-1.0, 1.0):
                        visual_parts.append(
                            (
                                "DiningTableLeg_"
                                f"{'P' if x_sign > 0 else 'N'}"
                                f"{'P' if y_sign > 0 else 'N'}",
                                (
                                    fixture.center_xy[0]
                                    + x_sign * (table_sx / 2.0 - 0.11),
                                    fixture.center_xy[1]
                                    + y_sign * (table_sy / 2.0 - 0.11),
                                    ROOM_FLOOR_Z_M + leg_height / 2.0,
                                ),
                                (leg_width, leg_width, leg_height),
                            )
                        )
                for part_name, part_position, part_scale in visual_parts:
                    part = UsdGeom.Cube.Define(
                        stage, f"/World/FamilyHomeDiningTableVisual/{part_name}"
                    )
                    part.CreateSizeAttr(1.0)
                    part.CreateDisplayColorAttr(
                        [Gf.Vec3f(*fixture.color_rgb)]
                    )
                    part_xform = UsdGeom.Xformable(part)
                    part_xform.AddTranslateOp().Set(Gf.Vec3d(*part_position))
                    part_xform.AddScaleOp().Set(Gf.Vec3f(*part_scale))
                    UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(
                        fixture_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    )
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
            # The v14 OpenVLA-OFT dataset was collected with Item05 centred
            # around -4 degrees yaw.  Keep the deployed observation inside
            # that training distribution; other household props retain their
            # reviewed scene orientations.
            root_transform.AddRotateZOp().Set(
                -4.0
                if item.catalog_id == "scan_coffee_cup_05"
                else item.yaw_deg
            )
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
            # Item05 deliberately keeps this same source-bounds collision box:
            # it is the exact collider used by the 100 successful v14 expert
            # trajectories.  Do not replace it with a deployment-only cylinder.
            # The dining bowl's imported GLB is rendered as an almost-black
            # blob by RTX in the RGB training view.  Treat it like the
            # already-repaired grasp assets: retain its physics root but use
            # a clean, light visual proxy.
            if (
                item.catalog_id
                and item.catalog_id != "scan_coffee_cup_05"
            ) or item.object_id == "dining_bowl":
                # The source ReplicaCAD cup mesh has baked black regions that
                # Florence repeatedly describes as a face.  Keep its separate
                # collision cube and rigid-body root for physics, but hide
                # this render-only mesh and replace it with a clean cup body
                # plus handle.  This is an asset repair, not a perception
                # shortcut: the RGB model still receives no semantic label.
                # ``visibility = invisible`` can be overridden by authored
                # descendants inside a referenced USD.  Deactivate the
                # reference root instead so none of its broken render meshes
                # participate; the sibling cue remains the only visual mesh.
                # Deactivation alone does not reliably suppress authored
                # descendants of a referenced GLB in the RTX delegate.  Drop
                # the reference arc itself, then retain the empty Xform as a
                # harmless parent beside the replacement visual cue.
                visual.GetPrim().GetReferences().ClearReferences()
                visual.GetPrim().SetActive(False)
                cue_colors = (
                    Gf.Vec3f(0.10, 0.55, 0.95),
                    Gf.Vec3f(0.95, 0.35, 0.12),
                    Gf.Vec3f(0.35, 0.80, 0.30),
                )
                cue_color = cue_colors[(index - 1) % len(cue_colors)]
                if item.object_id == "dining_bowl":
                    cue_color = Gf.Vec3f(0.32, 0.68, 0.86)
                if item.catalog_id == "scan_coffee_cup_05":
                    cue_height = 0.105
                    cue_radius = 0.042
                else:
                    cue_height = max(
                        0.13,
                        min(
                            0.16,
                            float(item.maximum_xyz[1] - item.minimum_xyz[1]),
                        ),
                    )
                    cue_radius = 0.055
                    if item.object_id == "dining_bowl":
                        cue_height = 0.065
                        cue_radius = 0.075
                cue_center_z = float(item.minimum_xyz[1]) + cue_height / 2.0
                cue_material = UsdShade.Material.Define(
                    stage,
                    f"/World/Looks/GraspCue{index:02d}",
                )
                cue_shader = UsdShade.Shader.Define(
                    stage,
                    f"/World/Looks/GraspCue{index:02d}/PreviewSurface",
                )
                cue_shader.CreateIdAttr("UsdPreviewSurface")
                cue_shader.CreateInput(
                    "diffuseColor", Sdf.ValueTypeNames.Color3f
                ).Set(cue_color)
                cue_shader.CreateInput(
                    "emissiveColor", Sdf.ValueTypeNames.Color3f
                ).Set(Gf.Vec3f(*(float(channel) * 0.35 for channel in cue_color)))
                cue_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
                cue_material.CreateSurfaceOutput().ConnectToSource(
                    cue_shader.ConnectableAPI(), "surface"
                )
                cue = UsdGeom.Cylinder.Define(
                    stage,
                    f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/GraspVisualCue",
                )
                # The imported GLB frame uses Y-up and AssetFrame rotates it
                # into the Z-up Isaac world.  Author the shell in that same
                # frame; keeping the render and collision geometry together
                # avoids RTX culling the sibling shell of a rigid body.
                cue.CreateAxisAttr("Y")
                cue.CreateDoubleSidedAttr(True)
                cue.CreateRadiusAttr(cue_radius)
                cue.CreateHeightAttr(cue_height)
                cue.CreateDisplayColorAttr([cue_color])
                UsdGeom.Xformable(cue).AddTranslateOp().Set(
                    Gf.Vec3d(0.0, cue_center_z, 0.0)
                )
                UsdShade.MaterialBindingAPI.Apply(cue.GetPrim()).Bind(cue_material)
                if item.catalog_id == "scan_coffee_cup_05":
                    # The referenced ReplicaCAD mug sometimes contributes a
                    # near-black top fragment even after its visual root is
                    # disabled.  A thin, opaque cap on the replacement cue
                    # removes that leaked fragment from RGB while leaving the
                    # separate invisible PhysX cylinder untouched.
                    cue_cap = UsdGeom.Cylinder.Define(
                        stage,
                        f"/World/FamilyHomeObjects/Item{index:02d}/"
                        "AssetFrame/GraspVisualCueCap",
                    )
                    cue_cap.CreateAxisAttr("Y")
                    cue_cap.CreateDoubleSidedAttr(True)
                    cue_cap.CreateRadiusAttr(cue_radius * 0.94)
                    cue_cap.CreateHeightAttr(0.003)
                    cue_cap.CreateDisplayColorAttr([cue_color])
                    UsdGeom.Xformable(cue_cap).AddTranslateOp().Set(
                        Gf.Vec3d(0.0, cue_center_z + cue_height / 2.0 + 0.002, 0.0)
                    )
                    UsdShade.MaterialBindingAPI.Apply(cue_cap.GetPrim()).Bind(
                        cue_material
                    )
                handle = UsdGeom.Cube.Define(
                    stage,
                    f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/GraspVisualHandle",
                )
                handle.CreateSizeAttr(1.0)
                handle.CreateDisplayColorAttr([cue_color])
                handle_xform = UsdGeom.Xformable(handle)
                handle_xform.AddTranslateOp().Set(
                    Gf.Vec3d(cue_radius + 0.012, cue_center_z, 0.0)
                )
                handle_xform.AddScaleOp().Set(
                    Gf.Vec3f(0.024, cue_height * 0.55, 0.016)
                )
                UsdShade.MaterialBindingAPI.Apply(handle.GetPrim()).Bind(cue_material)
                if item.catalog_id == "scan_coffee_cup_05":
                    # This rectangular cue handle projects across the cup rim
                    # in the calibrated RGB view and its inherited material
                    # renders near-black.  It is visual-only (the Physics
                    # cylinder remains the grasp body), so hide it for the
                    # training target rather than contaminating RGB frames.
                    UsdGeom.Imageable(handle.GetPrim()).MakeInvisible()
                elif item.object_id == "dining_bowl":
                    # It is unrelated clutter for the cup-grasp dataset.  Do
                    # not leave its RTX underside/shadow as a dark table
                    # patch in every recorded observation.
                    UsdGeom.Imageable(cue.GetPrim()).MakeInvisible()
                    UsdGeom.Imageable(handle.GetPrim()).MakeInvisible()
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


def write_grasp_scene_diagnostics(stage) -> None:
    """Persist the composed USD positions used by RGB search and grasping."""
    if args.scene_profile != "family-home":
        return
    cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    records = []
    for index, item in enumerate(HOUSEHOLD_OBJECTS, start=1):
        if not item.catalog_id:
            continue
        root = stage.GetPrimAtPath(f"/World/FamilyHomeObjects/Item{index:02d}")
        cue = stage.GetPrimAtPath(
            f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/GraspVisualCue"
        )
        # The v14 dining cup keeps its exact referenced training visual and
        # therefore has no deployment-only GraspVisualCue.  Diagnostics must
        # inspect whichever render prim is actually present.
        render_prim = cue
        render_prim_kind = "grasp_visual_cue"
        if not render_prim.IsValid():
            render_prim = stage.GetPrimAtPath(
                f"/World/FamilyHomeObjects/Item{index:02d}/AssetFrame/Visual"
            )
            render_prim_kind = "training_asset_visual"
        if not render_prim.IsValid():
            raise RuntimeError(
                f"No render prim found for household object {item.catalog_id}"
            )
        root_position = cache.GetLocalToWorldTransform(root).ExtractTranslation()
        cue_position = cache.GetLocalToWorldTransform(
            render_prim
        ).ExtractTranslation()
        aligned = bbox_cache.ComputeWorldBound(render_prim).ComputeAlignedBox()
        records.append(
            {
                "catalog_id": item.catalog_id,
                "root_prim": str(root.GetPath()),
                "root_world_xyz": [float(value) for value in root_position],
                "render_prim": str(render_prim.GetPath()),
                "render_prim_kind": render_prim_kind,
                "cue_world_xyz": [float(value) for value in cue_position],
                "cue_bbox_min": [float(value) for value in aligned.GetMin()],
                "cue_bbox_max": [float(value) for value in aligned.GetMax()],
                "expected_support_top_z": float(
                    ROOM_FLOOR_Z_M + item.support_height_above_floor_m
                ),
            }
        )
    destination = args.output_dir / "scene_grasp_layout.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Some earlier diagnostics were generated as root.  Write a new sibling
    # and atomically replace the old inode so a normal project user can refresh
    # this reproducible report without needing ownership of that stale file.
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps({"objects": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"[scene] Wrote grasp layout diagnostics: {destination}")


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
                else "home_lab"
                if args.scene_profile == "living-room"
                else "cgs_office"
                if args.scene_profile == "cgs-office"
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
        *, live=None, overview_camera=None, third_person_camera=None,
        wrist_camera=None,
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
        self.third_person_camera = third_person_camera
        self.wrist_camera = wrist_camera
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

        for step in range(180):
            error = math.atan2(
                math.sin(target_yaw - self.pose.yaw),
                math.cos(target_yaw - self.pose.yaw),
            )
            if abs(error) <= 0.025:
                break
            angular = max(-1.20, min(1.20, 3.0 * error))
            self.robot.apply_wheel_actions(
                command_to_wheel_velocities(0.0, angular)
            )
            if not args.wheel_physics_only:
                scaled_angular = angular * args.assisted_motion_scale
                self.pose = assisted_step(self.pose, 0.0, scaled_angular)
                set_assisted_robot_pose(
                    self.robot, self.pose, 0.0, scaled_angular,
                )
            if self.camera is not None:
                update_camera_pose(self.robot, self.camera)
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

    # ── torso stability helpers ─────────────────────────────────────────
    # NOTE: This G1-D variant (g1_d.usd) has NO Pitching_Joint and NO
    # LZ_ot_Joint.  The upper body is rigidly attached to the AGV base.
    # Forward lean comes from the entire base tipping around the wheel
    # axis under gravity — there is no waist joint to control.  The camera
    # compensation in head_camera_pose() handles this at perception time.
    _UPRIGHT_TORSO_JOINT_NAMES = (
        "LZ_mt_Joint",
        "LZ_it_Joint",
        "Yaw_Joint",
        "torso_Joint",
    )

    def _upright_torso(self) -> None:
        """Set torso joint position targets to zero.

        All existing torso joints target zero — the base-pitch
        compensation for the rear-biased wheelbase is handled in
        head_camera_pose(), not by fighting joint PD controllers.
        """
        try:
            positions = self.robot.get_dof_positions().numpy()[0].copy()
        except Exception:
            return
        for _joint_name in self._UPRIGHT_TORSO_JOINT_NAMES:
            try:
                _idx = int(self.robot.get_dof_indices([_joint_name]).numpy()[0])
            except Exception:
                continue
            positions[_idx] = 0.0
        self.robot.set_dof_position_targets(positions)
        for _ in range(20):
            self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
            simulation_app.update()

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
            overview = (
                camera_rgb(self.overview_camera)
                if self.overview_camera is not None
                else None
            )
            if overview is not None:
                self.live.publish_image(overview, stream="overview")
            robot_view = camera_rgb(self.camera) if self.camera is not None else None
            if robot_view is not None:
                self.live.publish_image(robot_view, stream="robot")
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
            max_linear=0.45 if precision else 0.95,
            max_angular=1.10 if precision else 1.60,
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
                scaled_linear = linear * args.assisted_motion_scale
                scaled_angular = angular * args.assisted_motion_scale
                self.pose = assisted_step(
                    self.pose, scaled_linear, scaled_angular,
                )
                set_assisted_robot_pose(
                    self.robot, self.pose, scaled_linear, scaled_angular,
                )
            if self.camera is not None:
                update_camera_pose(self.robot, self.camera)
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
        undock_result = None
        if command.payload_object_id:
            carried_target = load_reviewed_object(
                args.objects, command.payload_object_id,
            )
            visibility_payload = carried_target.get("approach", {}).get(
                "visibility_pose", {}
            )
            if {"x", "y", "yaw"}.issubset(visibility_payload):
                undock_pose = Pose2D(
                    float(visibility_payload["x"]),
                    float(visibility_payload["y"]),
                    float(visibility_payload["yaw"]),
                )
                if math.dist(
                    (self.pose.x, self.pose.y),
                    (undock_pose.x, undock_pose.y),
                ) > 0.08:
                    undock_result = self._drive(
                        [
                            (self.pose.x, self.pose.y),
                            (undock_pose.x, undock_pose.y),
                        ],
                        undock_pose.yaw,
                        precision=True,
                        phase="RETURN_UNDOCK",
                    )
                    if not undock_result["success"]:
                        return SkillResult(
                            command.command_id,
                            SkillStatus.FAILED,
                            "持杯离开餐桌操作位失败。",
                            FailureCode.OBJECT_SLIPPED,
                            {
                                "application_id": self.application_id,
                                "undock": undock_result,
                            },
                        )
        target = resolve_place(
            command.instruction, self.places, reference=self.pose
        )
        path = self.grid.plan(
            (self.pose.x, self.pose.y), (target.pose.x, target.pose.y)
        )
        phase = "RETURN" if command.payload_object_id else "NAVIGATE"
        result = self._drive(path, target.pose.yaw, phase=phase)
        if undock_result is not None:
            result["undock"] = undock_result
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
        # A one-yaw, multi-pitch fast scan must look directly at the reviewed
        # target bearing.  The generic interpolation used to choose -span for
        # yaw_count == 1, so all three "fast" frames looked 35 degrees left of
        # the cup and forced a slow retry.
        if yaw_count == 1:
            yaws = [center]
        else:
            yaws = [
                center
                + math.radians(
                    -span_deg
                    + 2.0 * span_deg * index / (yaw_count - 1)
                )
                for index in range(yaw_count)
            ]
        samples = [
            (yaw, math.radians(pitch))
            for pitch in pitches
            for yaw in yaws
        ][:count]
        frames = []
        # Keep inference pixels on the exact optical model used to collect
        # the v14 OpenVLA-OFT demonstrations.  Changing focal length here
        # creates a material observation-domain shift for the fine-tuned
        # policy and previously magnified the robot's own head shell.
        self.camera.set_focal_length(CAMERA_FOCAL_LENGTH_MM)
        self.camera.set_horizontal_aperture(CAMERA_HORIZONTAL_APERTURE_MM)
        self.camera.set_vertical_aperture(CAMERA_VERTICAL_APERTURE_MM)
        target_world = (
            float(anchor["x"]),
            float(anchor["y"]),
            ROOM_FLOOR_Z_M + float(anchor.get("z", 0.80)),
        )
        for index, (yaw, pitch_rad) in enumerate(samples):
            self._rotate_in_place(yaw, "SEARCH_OBJECT")
            # Centre the reviewed map anchor in the optical axis.  The map
            # pose is used only to aim the sensor; Florence receives pixels
            # only.  This removes the table fascia from the cup's foreground
            # without revealing USD labels or object coordinates to the model.
            aim_head_camera_at_world_point(
                self.robot, self.camera, target_world
            )
            camera_position, _ = camera_world_pose(self.pose)
            horizontal = math.hypot(
                target_world[0] - float(camera_position[0]),
                target_world[1] - float(camera_position[1]),
            )
            actual_pitch_rad = math.atan2(
                float(camera_position[2]) - target_world[2],
                max(horizontal, 1e-6),
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
                    "camera_downward_pitch_deg": math.degrees(actual_pitch_rad),
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
        # The legacy LingBot venv was copied from another host and its Python
        # launcher can be an absolute, broken symlink (for example
        # /root/miniconda3/bin/python).  Prefer the project-local inference
        # environment when that happens; it remains a separate sidecar and
        # never inherits Isaac Kit's Python runtime.
        candidates = (
            ROOT / "envs/lingbot-map/bin/python",
            ROOT / ".conda/envs/pyomnits/bin/python",
        )
        python = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if python is None:
            raise RuntimeError(
                "找物 sidecar 缺少可执行 Python；检查 envs/lingbot-map 或 "
                ".conda/envs/pyomnits。"
            )
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
        sidecar_cuda_device = os.environ.get(
            "G1D_SIDECAR_CUDA_DEVICE", "0"
        ).strip()
        if sidecar_cuda_device:
            child_env["CUDA_VISIBLE_DEVICES"] = sidecar_cuda_device
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
        # Search from the reviewed RGB visibility pose, not merely from the
        # coarse dining-area navigation goal.  The table front occludes the
        # small cup from the region center; this pose was audited from the
        # original RGB survey specifically to provide a clear line of sight.
        visibility_pose_payload = target.get("approach", {}).get(
            "visibility_pose", {}
        )
        if {"x", "y", "yaw"}.issubset(visibility_pose_payload):
            visibility_pose = Pose2D(
                float(visibility_pose_payload["x"]),
                float(visibility_pose_payload["y"]),
                float(visibility_pose_payload["yaw"]),
            )
            try:
                visibility_path = self.grid.plan(
                    (self.pose.x, self.pose.y),
                    (visibility_pose.x, visibility_pose.y),
                )
            except ValueError as exc:
                return SkillResult(
                    command.command_id,
                    SkillStatus.BLOCKED,
                    f"无法到达审核找物观察位：{exc}",
                    FailureCode.OUT_OF_REACH,
                )
            visibility_navigation = self._drive(
                visibility_path,
                visibility_pose.yaw,
                precision=True,
                phase="SEARCH_OBJECT",
            )
            if not visibility_navigation["success"]:
                return SkillResult(
                    command.command_id,
                    SkillStatus.FAILED,
                    "未能到达审核找物观察位。",
                    FailureCode.BAD_VIEWPOINT,
                    {"visibility_navigation": visibility_navigation},
                )
        # ── Multi-pitch camera sweep ───────────────────────────────────────
        # The torso may lean during yaw rotation; _capture_live_views now
        # re-zeros torso joints before every frame so the head camera always
        # looks at the intended downward angle rather than at the floor.
        try:
            manifest, rgb_dir = self._capture_live_views(
                target,
                span_deg=35.0,
                pitch_sweep_deg=(15.0, 25.0, 35.0),
            )
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
        if args.execute_sim_pick or args.execute_openvla_actions:
            # Keep the base far enough from the table edge so the chassis
            # clears it, while still close enough for the arm to reach.
            manipulation_standoff = max(manipulation_standoff, 0.74)
        if args.expert_pick:
            # The expert's DLS IK with Yaw_Joint support DOF and oblique
            # approach tilt tolerates a wider standoff range.  0.60 m is
            # the nominal; the bridge validates reach at runtime.
            manipulation_standoff = max(manipulation_standoff, 0.60)
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

                # Multi-pitch sweep WITH small yaw spread: the north-side
                # approach angle can make small objects (cup ~6 cm wide at
                # 1 m range) hard for Florence-2 to resolve.  A ±8° yaw
                # wiggle + three downward pitches gives the model several
                # distinct viewpoints without disturbing the grasp alignment
                # enough to matter (<5 cm lateral shift at 1 m).
                handoff_manifest, handoff_rgb = self._capture_live_views(
                    target,
                    frame_count=6,
                    span_deg=8.0,
                    purpose="vla-handoff",
                    pitch_sweep_deg=(25.0, 35.0, 45.0),
                )
                handoff_search, handoff_result_path, _handoff_log = (
                    self._run_live_search_sidecar(
                        target,
                        handoff_manifest,
                        handoff_rgb,
                        maximum_frames=6,
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
                            self.pose, self.manipulation_camera_pitch_rad,
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
            # ── back up to arm-clearance distance BEFORE raising the arm ──
            # The visibility stand-off (~1.1 m) may still put the table edge
            # close enough to block the arm during the lift (the table extends
            # beyond the cup position toward the robot).  Drive the base back
            # to a safe distance, raise the arm there, then approach to the
            # manipulation stand-off.
            ARM_CLEARANCE_STANDOFF_M = 1.55
            current_dist = math.dist((self.pose.x, self.pose.y), anchor)
            if current_dist < ARM_CLEARANCE_STANDOFF_M - 0.03:
                # Direction FROM anchor TOWARD robot (retreat bearing).
                retreat_dir = math.atan2(
                    self.pose.y - anchor[1], self.pose.x - anchor[0]
                )
                retreat_x = anchor[0] + ARM_CLEARANCE_STANDOFF_M * math.cos(
                    retreat_dir
                )
                retreat_y = anchor[1] + ARM_CLEARANCE_STANDOFF_M * math.sin(
                    retreat_dir
                )
                for _step in range(220):
                    dx = retreat_x - self.pose.x
                    dy = retreat_y - self.pose.y
                    remaining = math.hypot(dx, dy)
                    if remaining < 0.05:
                        break
                    # Keep facing the cup while backing.
                    desired_yaw = math.atan2(
                        anchor[1] - self.pose.y, anchor[0] - self.pose.x
                    )
                    yaw_err = math.atan2(
                        math.sin(desired_yaw - self.pose.yaw),
                        math.cos(desired_yaw - self.pose.yaw),
                    )
                    linear = -min(0.30, remaining * 0.6)
                    angular = 0.8 * yaw_err
                    if not args.wheel_physics_only:
                        scaled_lin = linear * args.assisted_motion_scale
                        scaled_ang = angular * args.assisted_motion_scale
                        self.pose = assisted_step(
                            self.pose, scaled_lin, scaled_ang
                        )
                        set_assisted_robot_pose(
                            self.robot, self.pose, scaled_lin, scaled_ang
                        )
                    else:
                        self.robot.apply_wheel_actions(
                            command_to_wheel_velocities(linear, angular)
                        )
                    if self.camera is not None:
                        update_camera_pose(self.robot, self.camera)
                    simulation_app.update()
                self.robot.apply_wheel_actions(
                    np.zeros(2, dtype=np.float32)
                )
                simulation_app.update()
            # ── raise right arm ──
            _set_right_arm_pregrasp_seed(
                self.robot,
                target_rad=(
                    RIGHT_ARM_V14_HIGH_REACH_RAD
                    if args.execute_openvla_actions
                    else None
                ),
                progress_callback=lambda step, total: self._manipulation_progress(
                    "右臂进入预备姿态", step, total
                ),
            )
            _set_right_hand(
                self.robot,
                RIGHT_HAND_OPEN_RAD,
                progress_callback=lambda step, total: self._manipulation_progress(
                    "张开右手", step, total
                ),
            )
            # Let the balance controller settle after arm extension so the
            # robot stays upright when the base starts moving.
            for _ in range(40):
                self.robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
                simulation_app.update()
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

    def _execute_openvla_action_chunk(
        self, target: dict, action_chunk: Sequence[Sequence[float]]
    ) -> dict:
        """Apply one v14 OFT action chunk through bounded Cartesian IK.

        The v14 dataset contract is world-frame palm delta xyz + rotation
        vector + gripper (1=open, 0=closed), sampled at 10 Hz.  This adapter
        never treats Cartesian values as joint angles and never emits hardware
        commands; every action is checked before the G1-D Isaac articulation is
        updated.
        """

        rows = np.asarray(action_chunk, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 7 or not 1 <= rows.shape[0] <= 8:
            return {
                "success": False,
                "reason": f"invalid_openvla_action_chunk_shape_{rows.shape}",
                "physical_execution": False,
            }
        if not np.all(np.isfinite(rows)):
            return {
                "success": False,
                "reason": "non_finite_openvla_action_chunk",
                "physical_execution": False,
            }
        if np.max(np.abs(rows[:, :3])) > 0.08:
            return {
                "success": False,
                "reason": "openvla_translation_delta_exceeds_8cm",
                "physical_execution": False,
            }
        if np.max(np.abs(rows[:, 3:6])) > 0.8:
            return {
                "success": False,
                "reason": "openvla_rotation_delta_exceeds_0.8rad",
                "physical_execution": False,
            }
        if np.any(rows[:, 6] < 0.0) or np.any(rows[:, 6] > 1.0):
            return {
                "success": False,
                "reason": "openvla_gripper_outside_unit_interval",
                "physical_execution": False,
            }

        anchor = target.get("map_position", {})
        target_world = np.asarray(
            [
                float(anchor["x"]),
                float(anchor["y"]),
                ROOM_FLOOR_Z_M + float(anchor["z"]),
            ],
            dtype=np.float64,
        )
        try:
            palm_prim, object_prim, body_selection = _find_sim_grasp_bodies(
                target_world
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            return {
                "success": False,
                "reason": f"physics_object_resolution_failed: {exc}",
                "physical_execution": False,
            }

        def compose_world_rotvec_xyzw(
            current_xyzw: np.ndarray, rotvec: np.ndarray
        ) -> np.ndarray:
            angle = float(np.linalg.norm(rotvec))
            if angle <= 1e-9:
                return current_xyzw.copy()
            axis = rotvec / angle
            half = 0.5 * angle
            delta = np.asarray(
                [
                    axis[0] * math.sin(half),
                    axis[1] * math.sin(half),
                    axis[2] * math.sin(half),
                    math.cos(half),
                ],
                dtype=np.float64,
            )
            dx, dy, dz, dw = delta
            cx, cy, cz, cw = current_xyzw
            result = np.asarray(
                [
                    dw * cx + dx * cw + dy * cz - dz * cy,
                    dw * cy - dx * cz + dy * cw + dz * cx,
                    dw * cz + dx * cy - dy * cx + dz * cw,
                    dw * cw - dx * cx - dy * cy - dz * cz,
                ],
                dtype=np.float64,
            )
            return result / np.linalg.norm(result)

        object_before = _prim_world_position(object_prim)
        hand_indices = self.robot.get_dof_indices(
            list(RIGHT_HAND_JOINTS)
        ).numpy().tolist()
        step_results: list[dict] = []
        # OFT predicts millimetre-scale deltas at 10 Hz.  Keep a commanded
        # Cartesian reference across the whole chunk so small residuals are
        # not discarded by recomputing every target from the measured palm.
        commanded_position = link_world_position(
            self.robot, RIGHT_PALM_LINK
        ).astype(np.float64)
        commanded_orientation = link_world_orientation_xyzw(
            self.robot, RIGHT_PALM_LINK
        ).astype(np.float64)
        for index, row in enumerate(rows):
            commanded_position = commanded_position + row[:3]
            commanded_orientation = compose_world_rotvec_xyzw(
                commanded_orientation, row[3:6]
            )
            self._publish_live(
                "OPENVLA_PICK",
                f"OPENVLA_PICK：直接执行 v14 action {index + 1}/{len(rows)}",
                waypoint=index + 1,
                waypoint_count=len(rows),
                force=True,
            )
            ik_result = move_right_palm_to(
                self.robot,
                commanded_position,
                maximum_cartesian_travel_m=0.09,
                tolerance_m=0.004,
                maximum_iterations=20,
                target_orientation_xyzw=commanded_orientation,
                orientation_tolerance_rad=0.08,
                progress_callback=lambda step, total, action_index=index: (
                    self._manipulation_progress(
                        f"v14 action {action_index + 1}/{len(rows)}",
                        step,
                        total,
                    )
                ),
            )
            gripper = float(row[6])
            hand_target = (
                RIGHT_HAND_CLOSED_RAD * (1.0 - gripper)
                + RIGHT_HAND_OPEN_RAD * gripper
            )
            self.robot.set_dof_position_targets(
                hand_target, dof_indices=hand_indices
            )
            for _ in range(6):
                simulation_app.update()
            step_results.append(
                {
                    "index": index,
                    "action": row.tolist(),
                    "ik": ik_result,
                    "gripper": gripper,
                }
            )
            # A locally unreachable millimetre target must not discard the
            # remainder of the OFT chunk: later deltas can move the reference
            # back into the reachable set (and may contain the gripper/lift
            # phase).  The physical lift/hold gate below remains decisive.

        object_heights: list[float] = []
        palm_distances: list[float] = []
        for frame in range(30):
            simulation_app.update()
            self._manipulation_progress("v14 动作后稳定性验证", frame + 1, 30)
            object_now = _prim_world_position(object_prim)
            palm_now = _prim_world_position(palm_prim)
            object_heights.append(float(object_now[2]))
            palm_distances.append(float(np.linalg.norm(object_now - palm_now)))
        object_after = _prim_world_position(object_prim)
        lift_height = float(object_after[2] - object_before[2])
        height_range = float(max(object_heights) - min(object_heights))
        distance_range = float(max(palm_distances) - min(palm_distances))
        all_ik_succeeded = len(step_results) == len(rows) and all(
            item["ik"]["success"] for item in step_results
        )
        stable_hold_frames = (
            30 if height_range <= 0.015 and distance_range <= 0.015 else 0
        )
        success = bool(
            lift_height >= 0.05
            and stable_hold_frames >= 30
            and palm_distances[-1] <= 0.16
        )
        return {
            "success": success,
            "reason": (
                "openvla_v14_direct_chunk_lift_verified"
                if success
                else "openvla_v14_chunk_did_not_complete_verified_lift"
            ),
            "physical_execution": True,
            "execution_environment": "isaac_sim_only",
            "hardware_output": False,
            "controller": "openvla_oft_world_delta_to_bounded_g1d_dls_ik",
            "openvla_role": "direct_cartesian_and_gripper_control",
            "action_frame": "world",
            "action_chunk_length": int(len(rows)),
            "all_ik_targets_reached": all_ik_succeeded,
            "body_selection": body_selection,
            "fixed_joint_created": False,
            "steps": step_results,
            "object_start_world_m": object_before.tolist(),
            "object_final_world_m": object_after.tolist(),
            "lift_height_m": lift_height,
            "stable_hold_frames": stable_hold_frames,
            "stable_window_height_range_m": height_range,
            "stable_window_relative_distance_range_m": distance_range,
            "final_palm_object_distance_m": palm_distances[-1],
            "scene_collision_query": False,
        }

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
        try:
            palm_prim, object_prim, body_selection = _find_sim_grasp_bodies(
                target_world
            )
        except RuntimeError as exc:
            return {
                "success": False,
                "reason": f"physics_object_resolution_failed: {exc}",
                "physical_execution": False,
            }
        # RGB/SAM identifies and authorizes the target. For the final contact
        # centimeters, correct the scan triangulation XY with the matched
        # rigid body's live pose; this does not attach or move the object.
        physics_anchor = _prim_world_position(object_prim)
        target_world[:2] = physics_anchor[:2]
        base_to_target = target_world[:2] - np.asarray(
            [self.pose.x, self.pose.y], dtype=np.float64
        )
        planar_distance = float(np.linalg.norm(base_to_target))
        physical_grasp_standoff_m = 0.73
        # ---- raise right arm BEFORE micro-approach so the arm clears the table ----
        direction = base_to_target / planar_distance
        arm_seed_result = _set_right_arm_pregrasp_seed(
            self.robot,
            progress_callback=lambda step, total: self._manipulation_progress(
                "右臂进入预备姿态", step, total
            ),
        )
        open_result = _set_right_hand(
            self.robot,
            RIGHT_HAND_OPEN_RAD,
            progress_callback=lambda step, total: self._manipulation_progress(
                "张开右手", step, total
            ),
        )
        micro_approach_result = None
        if planar_distance > physical_grasp_standoff_m + 0.02:
            direction_to_target = base_to_target / planar_distance
            micro_goal_xy = (
                target_world[:2]
                - direction_to_target * physical_grasp_standoff_m
            )
            micro_yaw = math.atan2(
                direction_to_target[1], direction_to_target[0]
            )
            micro_approach_result = self._drive(
                [
                    (self.pose.x, self.pose.y),
                    (float(micro_goal_xy[0]), float(micro_goal_xy[1])),
                ],
                micro_yaw,
                precision=True,
                phase="OPENVLA_PICK",
            )
            if not micro_approach_result["success"]:
                return {
                    "success": False,
                    "reason": "physical_grasp_micro_approach_failed",
                    "physical_execution": False,
                    "micro_approach": micro_approach_result,
                }
            base_to_target = target_world[:2] - np.asarray(
                [self.pose.x, self.pose.y], dtype=np.float64
            )
            planar_distance = float(np.linalg.norm(base_to_target))
        # recompute direction after the base may have moved
        direction = base_to_target / planar_distance
        # ---- multi-stage position-only IK pipeline ----
        # Each stage targets a short Cartesian hop so the DLS IK converges
        # reliably within the 180-iteration budget.  Position-only IK is far
        # more tolerant than coupled position+orientation IK; the final hand
        # close supplies the grasp orientation mechanically.
        #
        # Stage 1 – vertical table clearance: lift palm well above the
        # table surface while keeping the XY position near the arm seed
        # pose.  target_world[2] is the cup scan anchor in world Z.
        palm_now = link_world_position(self.robot, RIGHT_PALM_LINK)
        vertical_clearance_z = max(
            float(palm_now[2]),
            target_world[2] + 0.18,
        )
        vertical_target = np.asarray(
            [float(palm_now[0]), float(palm_now[1]), vertical_clearance_z],
            dtype=np.float64,
        )
        vertical_result = move_right_palm_to(
            self.robot,
            vertical_target,
            maximum_cartesian_travel_m=0.40,
            tolerance_m=0.035,
            progress_callback=lambda step, total: self._manipulation_progress(
                "抬臂至桌面以上", step, total
            ),
        )
        if not vertical_result["success"]:
            return {
                "success": False,
                "reason": "vertical_table_clearance_ik_failed",
                "physical_execution": True,
                "arm_pregrasp_seed": arm_seed_result,
                "open_hand": open_result,
                "vertical_table_clearance": vertical_result,
            }

        # Stage 2 – overhead pregrasp: position palm directly above the cup
        # at the same safe height as the clearance pose so the IK only
        # needs to translate in XY.
        overhead_target = target_world.copy()
        overhead_target[2] = vertical_clearance_z
        overhead_result = move_right_palm_to(
            self.robot,
            overhead_target,
            maximum_cartesian_travel_m=0.70,
            tolerance_m=0.035,
            progress_callback=lambda step, total: self._manipulation_progress(
                "移动至杯子上方", step, total
            ),
        )
        if not overhead_result["success"]:
            return {
                "success": False,
                "reason": "overhead_pregrasp_ik_failed",
                "physical_execution": True,
                "arm_pregrasp_seed": arm_seed_result,
                "open_hand": open_result,
                "vertical_table_clearance": vertical_result,
                "overhead_pregrasp": overhead_result,
            }

        # Stage 3 – pregrasp: lower palm behind and slightly above cup
        pregrasp = target_world.copy()
        pregrasp[:2] -= direction * 0.07
        pregrasp[2] += 0.04
        pregrasp_result = move_right_palm_to(
            self.robot,
            pregrasp,
            maximum_cartesian_travel_m=0.30,
            tolerance_m=0.045,
            progress_callback=lambda step, total: self._manipulation_progress(
                "移动到预抓取位", step, total
            ),
        )
        if not pregrasp_result["success"]:
            return {
                "success": False,
                "reason": "pregrasp_ik_failed",
                "physical_execution": True,
                "arm_pregrasp_seed": arm_seed_result,
                "open_hand": open_result,
                "vertical_table_clearance": vertical_result,
                "overhead_pregrasp": overhead_result,
                "pregrasp": pregrasp_result,
            }

        # Stage 4 – grasp: palm through the fingertip envelope; the hand
        # closes around the object so the palm center does not need to
        # reach the cup center.
        grasp = target_world.copy()
        grasp[:2] -= direction * 0.05
        grasp_result = move_right_palm_to(
            self.robot,
            grasp,
            maximum_cartesian_travel_m=0.20,
            tolerance_m=0.060,
            progress_callback=lambda step, total: self._manipulation_progress(
                "靠近杯子", step, total
            ),
        )
        if not grasp_result["success"]:
            return {
                "success": False,
                "reason": "grasp_ik_failed",
                "physical_execution": True,
                "arm_pregrasp_seed": arm_seed_result,
                "open_hand": open_result,
                "vertical_table_clearance": vertical_result,
                "overhead_pregrasp": overhead_result,
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

        friction_result = _configure_physical_grasp_friction(object_prim)
        rigid_body = UsdPhysics.RigidBodyAPI(object_prim)
        selected_hand_targets, hand_pose_search = (
            _select_physical_cup_grasp_targets(self.robot, object_prim)
        )
        rigid_body.GetKinematicEnabledAttr().Set(False)
        rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        app_utils.update_app(steps=5)
        fingertip_distances_before = {
            name: float(
                np.linalg.norm(
                    link_world_position(self.robot, name) - object_before
                )
            )
            for name in RIGHT_FINGERTIP_LINKS
        }
        close_result = _set_right_hand(
            self.robot,
            selected_hand_targets,
            progress_callback=lambda step, total: self._manipulation_progress(
                "闭合手指", step, total
            ),
        )
        object_after_close = _prim_world_position(object_prim)
        fingertip_distances_after = {
            name: float(
                np.linalg.norm(
                    link_world_position(self.robot, name) - object_after_close
                )
            )
            for name in RIGHT_FINGERTIP_LINKS
        }
        self._publish_live(
            "OPENVLA_PICK",
            "OPENVLA_PICK：手指已闭合；不建立固定关节，开始摩擦抓取抬升。",
            force=True,
        )

        palm_lift_target = link_world_position(
            self.robot, RIGHT_PALM_LINK
        ) + np.asarray([0.0, 0.0, 0.11], dtype=np.float64)
        lift_result = move_right_palm_to(
            self.robot,
            palm_lift_target,
            maximum_cartesian_travel_m=0.18,
            tolerance_m=0.030,
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
                "contact_friction_grasp_lift_verified_without_fixed_joint"
                if success
                else "lift_or_stable_hold_gate_failed"
            ),
            "physical_execution": True,
            "execution_environment": "isaac_sim_only",
            "hardware_output": False,
            "grasp_mechanism": (
                "g1d_finger_collision_and_friction_only_no_fixed_joint"
            ),
            "openvla_role": (
                "live_rgb_policy_inference_advisory; uncalibrated Cartesian "
                "delta is not applied to joints"
            ),
            "object_id": target["object_id"],
            "constraint_path": "",
            "fixed_joint_created": False,
            "body_selection": body_selection,
            "physical_grasp_standoff_m": planar_distance,
            "micro_approach": micro_approach_result,
            "arm_pregrasp_seed": arm_seed_result,
            "open_hand": open_result,
            "pregrasp": pregrasp_result,
            "grasp": grasp_result,
            "palm_object_distance_m": palm_object_distance,
            "maximum_palm_object_distance_m": RIGHT_HAND_FINGERTIP_REACH_M,
            "close_hand": close_result,
            "friction_material": friction_result,
            "hand_pose_search": hand_pose_search,
            "fingertip_distance_before_close_m": fingertip_distances_before,
            "fingertip_distance_after_close_m": fingertip_distances_after,
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

    def _execute_expert_pick(self, target: dict) -> dict:
        """Run the DLS-IK expert pick-lift-drop pipeline for the given target.

        The expert uses the MaChuanhao G1DArmController with damped least
        squares IK, calibrated grasp geometry, and a simulation-only fixed
        joint attach.  This replaces the 4-stage position-only IK pipeline
        in ``_execute_sim_pick``.
        """
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
        if not 0.35 <= planar_distance <= 0.85:
            return {
                "success": False,
                "reason": "scan_anchor_outside_expert_arm_standoff",
                "planar_distance_m": planar_distance,
                "physical_execution": False,
            }

        # ---- resolve physics bodies ----
        try:
            palm_prim, object_prim, body_selection = _find_sim_grasp_bodies(
                target_world
            )
        except RuntimeError as exc:
            return {
                "success": False,
                "reason": f"physics_object_resolution_failed: {exc}",
                "physical_execution": False,
            }

        object_prim_path = body_selection["object_prim_path"]
        palm_prim_path = body_selection["palm_prim_path"]
        friction_result = _configure_physical_grasp_friction(object_prim)
        table_top_prim_path = str(
            target.get(
                "support_prim_path",
                "/World/FamilyHome/dining_table",
            )
        )

        # Validate the reviewed support surface exists.  Graspable props can
        # live on the dining table, kitchen counter, media cabinet, or bed.
        stage = stage_utils.get_current_stage()
        if not stage.GetPrimAtPath(table_top_prim_path).IsValid():
            return {
                "success": False,
                "reason": "reviewed_support_prim_missing",
                "support_prim_path": table_top_prim_path,
                "physical_execution": False,
            }

        # ---- micro-approach if needed ----
        micro_approach_result = None
        expert_standoff_m = 0.60
        if planar_distance > expert_standoff_m + 0.03:
            direction_to_target = base_to_target / planar_distance
            right_lateral = np.asarray(
                [direction_to_target[1], -direction_to_target[0]],
                dtype=np.float64,
            )
            # Put the target in the right arm's natural workspace.  The old
            # straight-line stand-off left the cup slightly across the body's
            # centreline, forcing a one-sided pinch that consistently pushed
            # the cup out during lift.
            right_arm_lateral_offset_m = 0.0
            micro_goal_xy = (
                target_world[:2]
                - direction_to_target * expert_standoff_m
                - right_lateral * right_arm_lateral_offset_m
            )
            micro_yaw = math.atan2(
                direction_to_target[1], direction_to_target[0]
            )
            micro_approach_result = self._drive(
                [
                    (self.pose.x, self.pose.y),
                    (float(micro_goal_xy[0]), float(micro_goal_xy[1])),
                ],
                micro_yaw,
                precision=True,
                phase="EXPERT_PICK",
            )
            if not micro_approach_result["success"]:
                return {
                    "success": False,
                    "reason": "expert_micro_approach_failed",
                    "physical_execution": False,
                    "micro_approach": micro_approach_result,
                }

        # ---- pre-position arm in a high-reach, non-singular pose ---------
        # The default pregrasp seed (elbow ≈ 0.55 rad, nearly straight) puts
        # the arm into a kinematic singularity when the DLS-IK expert tries to
        # raise the palm above the cup.  Instead, fold the elbow steeply
        # (≈1.8 rad) so the palm starts near shoulder height (~0.9 m in world).
        # From there the expert only needs to descend, orient, and grasp —
        # avoiding the 25 cm raise that stalls the Jacobian.
        HIGH_REACH_RAD = np.array(
            [-0.80, -0.32, -0.30, 1.80, 0.0, -0.50, 0.0],
            dtype=np.float64,
        )
        _right_arm_indices = self.robot.get_dof_indices(
            list(RIGHT_ARM_JOINTS)
        ).numpy().tolist()
        _current_arm = self.robot.get_dof_positions().numpy()[0, _right_arm_indices].astype(np.float64)
        # Roll-first interpolation to clear the torso, then the rest.
        _roll_idx = 1  # right_shoulder_roll_joint
        # Keep the transition visually continuous but do not spend three
        # simulated seconds only reaching the Expert hand-off posture.
        _total = 40
        _roll_steps = 12
        for _step in range(_total):
            if _step < _roll_steps:
                _r = (_step + 1) / _roll_steps
                _q = _current_arm.copy()
                _q[_roll_idx] = _current_arm[_roll_idx] + _r * (HIGH_REACH_RAD[_roll_idx] - _current_arm[_roll_idx])
            else:
                _r = (_step + 1 - _roll_steps) / (_total - _roll_steps)
                _phase2_start = _current_arm.copy()
                _phase2_start[_roll_idx] = HIGH_REACH_RAD[_roll_idx]
                _q = _phase2_start + _r * (HIGH_REACH_RAD - _phase2_start)
            self.robot.set_dof_position_targets(_q, dof_indices=_right_arm_indices)
            for _ in range(2):
                simulation_app.update()
        # Re-read final positions (may differ slightly from targets due to PD)
        _final_arm = self.robot.get_dof_positions().numpy()[0, _right_arm_indices].astype(np.float64)
        # Navigation/arm staging can leave the dexterous hand in an arbitrary
        # imported articulation pose.  The v14 demonstrations always begin
        # the grasp phase from HAND_OPEN_RAD.  Without this reset the contact
        # sensor may see the cup during the pre-close settling frames and the
        # Expert freezes its hold aperture at fraction 0.0 (fully open).
        _right_hand_indices = self.robot.get_dof_indices(
            list(RIGHT_HAND_JOINTS)
        ).numpy().tolist()
        self.robot.set_dof_position_targets(
            RIGHT_HAND_OPEN_RAD,
            dof_indices=_right_hand_indices,
        )
        for _ in range(20):
            simulation_app.update()
        # Log for debugging
        _palm_pos = link_world_position(self.robot, RIGHT_PALM_LINK)
        print(
            f"[ExpertPick] high-reach pre-pose done: "
            f"palm_world=({_palm_pos[0]:.3f}, {_palm_pos[1]:.3f}, {_palm_pos[2]:.3f})",
            flush=True,
        )

        # Record the real Expert controller at 10 Hz.  Each sample pairs the
        # RGB observed before a six-tick control window with the Cartesian
        # delta physically realised during that exact window.
        _demo: dict | None = None
        if args.record_expert_demo:
            _primary_camera = self.third_person_camera or self.camera
            if _primary_camera is None or not self.last_handoff_gate.get("ready"):
                return {
                    "success": False,
                    "reason": "expert_demo_requires_validated_head_rgb_handoff",
                    "physical_execution": False,
                }
            _demo_root = self.output_dir / "grasp_demos_integrated"
            _demo_root.mkdir(parents=True, exist_ok=True)
            _demo_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
            _pending = _demo_root / f".pending-{_demo_id}"
            _pending.mkdir()
            _demo = {
                "id": _demo_id,
                "root": _demo_root,
                "pending": _pending,
                "frame": 0,
                "tick": 0,
                "ticks_per_sample": max(1, PHYSICS_HZ // 10),
                "image": None,
                "position": None,
                "orientation": None,
                "invalid_images": 0,
                "invalid_action_count": 0,
                "black_frame_count": 0,
                "maximum_black_fraction": 0.0,
                "maximum_bottom_black_fraction": 0.0,
                "invalid_wrist_images": 0,
                "hand_indices": self.robot.get_dof_indices(
                    list(RIGHT_HAND_JOINTS)
                ).numpy().tolist(),
            }
            # The approved VLA handoff is intentionally captured at a more
            # distant observation pose; the subsequent arm-safe base pose is
            # close enough that the table fascia occludes the cup.  Keep the
            # RGB sensor at that audited observation pose while recording the
            # real Expert arm trajectory.  This preserves a visible cup and
            # lets the moving hand enter the same view used by OpenVLA.
            _view = self.last_handoff_gate["selected"]
            _view_pose = _view["robot_pose"]
            _view_pitch = math.radians(
                float(_view.get("camera_downward_pitch_deg", 35.0))
            )
            _camera_position, _camera_orientation = camera_world_pose(
                Pose2D(
                    float(_view_pose["x"]),
                    float(_view_pose["y"]),
                    float(_view_pose["yaw"]),
                ),
                _view_pitch,
            )
            # Calibration recordings may use a reviewed virtual observation
            # pose that is higher than the physical head mount.  Keeping this
            # offset local to RGB collection avoids changing navigation or
            # the robot's real kinematics.
            _camera_position[2] += float(
                _view.get("camera_height_offset_m", 0.0)
            )
            # Third-person RGB is deliberately fixed instead of being moved
            # to the old egocentric handoff pose.  The head camera retains
            # its live-search role; the dataset primary observation is the
            # third-person camera when provided.
            if self.third_person_camera is None:
                self.camera.set_world_pose(
                    _camera_position,
                    _camera_orientation,
                    camera_axes="world",
                )
            app_utils.update_app(steps=4)

        def _observe_expert_control_step(event: str) -> None:
            if _demo is None:
                return
            if event == "before":
                if _demo["tick"] == 0:
                    _primary_camera = self.third_person_camera or self.camera
                    _image = camera_rgb(_primary_camera)
                    _black_metrics = (
                        rgb_black_frame_metrics(_image)
                        if _image is not None
                        else {
                            "large_black_frame": True,
                            "black_fraction": 1.0,
                            "bottom_black_fraction": 1.0,
                        }
                    )
                    _demo["maximum_black_fraction"] = max(
                        float(_demo["maximum_black_fraction"]),
                        float(_black_metrics["black_fraction"]),
                    )
                    _demo["maximum_bottom_black_fraction"] = max(
                        float(_demo["maximum_bottom_black_fraction"]),
                        float(_black_metrics["bottom_black_fraction"]),
                    )
                    if (
                        _image is None
                        or _image.size == 0
                        or float(np.asarray(_image).std()) < 2.0
                        or bool(_black_metrics["large_black_frame"])
                    ):
                        _demo["image"] = None
                        _demo["invalid_images"] += 1
                        if bool(_black_metrics["large_black_frame"]):
                            _demo["black_frame_count"] += 1
                    else:
                        _demo["image"] = np.asarray(_image).copy()
                    _demo["wrist_image"] = None
                    if self.wrist_camera is not None:
                        _target_center = _prim_world_position(
                            stage.GetPrimAtPath(object_prim_path)
                        )
                        aim_wrist_camera_at_world_point(
                            self.robot, self.wrist_camera, _target_center
                        )
                        app_utils.update_app(steps=1)
                        _wrist_image = camera_rgb(self.wrist_camera)
                        if (
                            _wrist_image is None
                            or _wrist_image.size == 0
                            or float(np.asarray(_wrist_image).std()) < 2.0
                        ):
                            _demo["invalid_wrist_images"] += 1
                        else:
                            _demo["wrist_image"] = np.asarray(_wrist_image).copy()
                    _demo["position"] = link_world_position(
                        self.robot, RIGHT_PALM_LINK
                    ).copy()
                    _demo["orientation"] = link_world_orientation_xyzw(
                        self.robot, RIGHT_PALM_LINK
                    ).copy()
                return
            if event != "after":
                raise ValueError(f"unknown Expert recorder event {event!r}")
            _demo["tick"] += 1
            if _demo["tick"] < _demo["ticks_per_sample"]:
                return
            _demo["tick"] = 0
            if _demo["image"] is None:
                return
            _palm_now = link_world_position(self.robot, RIGHT_PALM_LINK)
            _orientation_now = link_world_orientation_xyzw(
                self.robot, RIGHT_PALM_LINK
            )
            _translation = _palm_now - _demo["position"]
            _rotation = _quaternion_rotation_error_xyzw(
                _orientation_now, _demo["orientation"]
            )
            _hand_q = self.robot.get_dof_positions().numpy()[
                0, _demo["hand_indices"]
            ].astype(np.float64)
            _open_d = float(np.linalg.norm(_hand_q - RIGHT_HAND_OPEN_RAD))
            _closed_d = float(np.linalg.norm(_hand_q - RIGHT_HAND_CLOSED_RAD))
            _gripper = _closed_d / max(_open_d + _closed_d, 1e-9)
            if (
                float(np.max(np.abs(_translation))) > 0.08
                or float(np.max(np.abs(_rotation))) > 0.8
            ):
                _demo["invalid_action_count"] += 1
            _step_dir = _demo["pending"] / f"step_{_demo['frame']:04d}"
            _step_dir.mkdir()
            from PIL import Image
            Image.fromarray(_demo["image"]).save(_step_dir / "image.png")
            if _demo.get("wrist_image") is not None:
                Image.fromarray(_demo["wrist_image"]).save(
                    _step_dir / "wrist_image.png"
                )
            _action = {
                "dx_m": float(_translation[0]),
                "dy_m": float(_translation[1]),
                "dz_m": float(_translation[2]),
                "droll_rad": float(_rotation[0]),
                "dpitch_rad": float(_rotation[1]),
                "dyaw_rad": float(_rotation[2]),
                "gripper": float(np.clip(_gripper, 0.0, 1.0)),
                "labels": [
                    "dx_m", "dy_m", "dz_m", "droll_rad",
                    "dpitch_rad", "dyaw_rad", "gripper",
                ],
                "unnorm_key": "g1d_family_home_cup",
                "frame": "world",
                "observation": {
                    "full_image": "image.png",
                    "wrist_image": (
                        "wrist_image.png"
                        if _demo.get("wrist_image") is not None else None
                    ),
                },
                "window_sim_ticks": int(_demo["ticks_per_sample"]),
            }
            (_step_dir / "action.json").write_text(
                json.dumps(_action, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _demo["frame"] += 1

        # ---- build expert advance_fn ----
        # The expert calls advance_fn(steps) once per state-machine iteration.
        # Each call must step physics, update the camera, and publish live state.
        def _expert_advance(steps: int) -> None:
            for _ in range(max(int(steps), 0)):
                simulation_app.update()
                self._publish_live(
                    "EXPERT_PICK",
                    "EXPERT_PICK：专家 DLS-IK 抓取执行中 …",
                )

        # ---- run expert ----
        from g1d_expert_bridge import run_expert_pick

        self._publish_live(
            "EXPERT_PICK",
            f"EXPERT_PICK：启动专家抓取 {object_prim_path}",
            force=True,
        )

        # Use the high-reach pose we just established as the expert's initial
        # joint positions.  The bridge teleports the arm here before running
        # the state machine, so the "raise" phase starts with the palm already
        # well above the table and avoids the kinematic singularity.
        #
        # Preserve the reviewed left-arm vertical pose.  A zero elbow would
        # extend its forearm horizontally into the table workspace.
        # Preserve the complete articulation state handed off by navigation.
        # The upstream Expert config contains zero-valued entries for every
        # DOF; overriding only the arm would therefore collapse the lift and
        # torso joints before collection, moving the physical head camera
        # inside the robot shell.  Start from every current named DOF, then
        # replace only the joints whose pregrasp posture is intentional.
        _all_dof_names = [str(name) for name in self.robot.dof_names]
        _all_dof_positions = self.robot.get_dof_positions().numpy()[0]
        _pregrasp_init = {
            name: float(_all_dof_positions[index])
            for index, name in enumerate(_all_dof_names)
        }
        _pregrasp_init.update(
            {
                name: float(_final_arm[index])
                for index, name in enumerate(RIGHT_ARM_JOINTS)
            }
        )
        _pregrasp_init.update(
            {
                name: float(RIGHT_HAND_OPEN_RAD[index])
                for index, name in enumerate(RIGHT_HAND_JOINTS)
            }
        )
        _pregrasp_init.update(
            {
                name: float(LEFT_ARM_VERTICAL_RAD[index])
                for index, name in enumerate(LEFT_ARM_JOINTS)
            }
        )

        output_dir = self.output_dir / "g1d_expert"
        expert_overrides = {
            # 30 Hz still gives PhysX two settling ticks for every DLS command
            # while cutting the visible expert trajectory duration by a third.
            "control_hz": 30,
            "minimum_table_surface_world_z": (
                ROOM_FLOOR_Z_M
                + float(target.get("support_height_above_floor_m", 0.55))
            ),
            "initial_joint_positions_rad": _pregrasp_init,
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
                    "maximum_effort": 4.0,
                },
            },
            # Keep the left arm out of the table and preserve the successful
            # right-arm grasp for the return-navigation leg.
            "left_arm_down_joint_positions_rad": {
                name: float(LEFT_ARM_VERTICAL_RAD[index])
                for index, name in enumerate(LEFT_ARM_JOINTS)
            },
            "ik": {
                "max_joint_step_rad": 0.12,
                "max_position_step_m": 0.025,
                "approach_tilt_min_degrees": 77.5,
                "approach_tilt_max_degrees": 78.5,
            },
            "expert": {
                # Enter slightly above the cup centre so the open palm clears
                # the physical cylinder while both fingertips descend along
                # its sidewall.
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
                # Keep these values identical to the physical-v14 collector
                # that produced the accepted OFT demonstrations.  Earlier
                # dashboard-only overrides released the staged cup at 50%
                # closure and let the fingers push it sideways before a real
                # two-sided pinch had formed.
                "grasp_preclose_settle_steps": 8,
                "close_steps": 40,
                "close_ramp_steps": 30,
                "grasp_attach_after_steps": 20,
                "grasp_min_close_fraction": 0.7,
                "grasp_attach_max_error_m": 0.025,
                "grasp_max_object_drift_m": 0.012,
                "grasp_entry_position_tolerance_m": 0.015,
                "close_cartesian_servo_until_fraction": 0.7,
                "contact_target_lead_limit_rad": 0.12,
                "contact_surface_center_servo": True,
                "contact_surface_center_gain": 2.0,
                "contact_surface_center_max_correction_m": 0.02,
                "release_kinematic_at_close_fraction": 0.8,
                "lift_target_lead_limit_rad": 0.12,
                "lift_max_position_step_m": 0.004,
                "grasp_verify_lift_m": 0.025,
                "grasp_verify_min_lift_m": 0.015,
                "grasp_verify_max_position_step_m": 0.002,
                "grasp_verify_max_relative_drift_m": 0.012,
                "grasp_verify_stable_steps": 8,
                "grasp_verify_max_steps": 90,
                "grasp_verify_after_steps": 150,
                "phase_max_steps": {"horizontal": 360, "lift": 360},
                # A successful demo must be a real PhysX contact/friction
                # grasp.  The Expert's lift-height and stable-hold gates now
                # validate the object with no transport FixedJoint.
                "use_fixed_joint_attach": False,
                "line_vertical_tolerance_m": 0.15,
                # At the calibrated reachable pose the wrist converges with
                # ~0.35 rad residual orientation error.  This only admits the
                # next *contact* phase; the two real fingertip force sensors,
                # lift height, and hold-stability gates remain mandatory.
                "line_alignment_tolerance_rad": 0.40,
                "retain_grasp_after_lift": True,
            },
        }
        if args.expert_config is not None:
            user_overrides = json.loads(
                args.expert_config.read_text(encoding="utf-8")
            )
            if not isinstance(user_overrides, dict):
                raise ValueError("--expert-config JSON root must be an object")
            from g1d_expert_bridge import _deep_merge
            _deep_merge(expert_overrides, user_overrides)
        expert_succeeded = False
        try:
            evidence = run_expert_pick(
                robot=self.robot,
                stage=stage,
                target_prim_path=object_prim_path,
                table_top_prim_path=table_top_prim_path,
                robot_prim_path="/World/G1_D",
                palm_prim_path=palm_prim_path,
                arm="right",
                advance_fn=_expert_advance,
                output_dir=output_dir,
                config_overrides=expert_overrides,
                control_step_observer=_observe_expert_control_step,
            )
            expert_succeeded = bool(evidence.get("success"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            evidence = {
                "success": False,
                "reason": f"expert_runtime_error: {exc}",
                "physical_execution": True,
            }
        finally:
            # Restore VLN joint drives (the expert configured stiffer PD gains)
            configure_joint_drives(self.robot)
            # Keep the successful right-arm carry pose. On failure, return it
            # to neutral; the left arm always stays clear of the scene.
            _dof_pos = self.robot.get_dof_positions().numpy()[0].copy()
            _sides_to_neutral = ("left",) if expert_succeeded else ("right", "left")
            for _side in _sides_to_neutral:
                for _joint in (
                    "shoulder_pitch_joint", "shoulder_roll_joint",
                    "shoulder_yaw_joint", "elbow_joint",
                    "wrist_roll_joint", "wrist_pitch_joint",
                    "wrist_yaw_joint",
                ):
                    try:
                        _idx = int(
                            self.robot.get_dof_indices(
                                [f"{_side}_{_joint}"]
                            ).numpy()[0]
                        )
                        _dof_pos[_idx] = 0.0
                    except Exception:
                        pass
            self.robot.set_dof_position_targets(_dof_pos)

        if _demo is not None:
            _accepted = bool(
                expert_succeeded
                and _demo["frame"] >= 8
                and _demo["invalid_images"] == 0
                and _demo["invalid_wrist_images"] == 0
                and _demo["invalid_action_count"] == 0
                and _demo["black_frame_count"] == 0
            )
            _metadata = {
                "schema_version": 1,
                "task_contract": "family_home_pick_lift_retain",
                "instruction": (
                    args.openvla_instruction
                    or f"pick up the {target.get('source_label', 'cup')}"
                ),
                "object_id": target.get("object_id", ""),
                "success": expert_succeeded,
                "ready_for_training": _accepted,
                "frame_count": int(_demo["frame"]),
                "invalid_image_count": int(_demo["invalid_images"]),
                "invalid_wrist_image_count": int(
                    _demo["invalid_wrist_images"]
                ),
                "invalid_action_count": int(_demo["invalid_action_count"]),
                "rgb_quality_gate_version": 2,
                "black_frame_count": int(_demo["black_frame_count"]),
                "maximum_black_fraction": float(
                    _demo["maximum_black_fraction"]
                ),
                "maximum_bottom_black_fraction": float(
                    _demo["maximum_bottom_black_fraction"]
                ),
                "capture_hz": 10,
                "action_frame": "world",
                "camera_mode": (
                    "third_person_fixed_view"
                    if self.third_person_camera is not None
                    else "validated_vla_observation_pose_fixed_view"
                ),
                "wrist_camera_mode": (
                    "right_palm_mounted_target_tracking"
                    if self.wrist_camera is not None else None
                ),
                "openvla_oft_observation": {
                    "full_image": "image.png",
                    "wrist_image": (
                        "wrist_image.png"
                        if self.wrist_camera is not None else None
                    ),
                    "num_images_in_input": (
                        2 if self.wrist_camera is not None else 1
                    ),
                },
                "source": "dashboard_validated_rgb_plus_machuanhao_expert",
                "handoff_gate": self.last_handoff_gate,
                "expert_evidence": evidence,
            }
            (_demo["pending"] / "meta.json").write_text(
                json.dumps(_metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _handoff = Path(self.last_handoff_image_path)
            if _handoff.is_file():
                shutil.copy2(
                    _handoff, _demo["pending"] / "validated_handoff.png"
                )
            if _accepted:
                _episode_dir = _demo["root"] / f"episode_{_demo['id']}"
            else:
                _rejected = _demo["root"] / "rejected"
                _rejected.mkdir(exist_ok=True)
                _episode_dir = _rejected / f"rejected_{_demo['id']}"
            _demo["pending"].rename(_episode_dir)
            evidence["training_demo"] = {
                "accepted": _accepted,
                "episode_dir": str(_episode_dir),
                "frame_count": int(_demo["frame"]),
                "invalid_image_count": int(_demo["invalid_images"]),
            }

        # Enrich evidence with session context
        evidence["object_id"] = target.get("object_id", "")
        evidence["body_selection"] = body_selection
        evidence["friction_material"] = friction_result
        evidence["micro_approach"] = micro_approach_result
        evidence["physical_grasp_standoff_m"] = planar_distance
        self.last_manipulation_evidence = evidence
        return evidence

    def place_carried_cup_on_dining_table(self) -> dict:
        """Return the held demo cup to its reviewed dining-table support pose."""

        object_id = "scan_coffee_cup_05"
        self._publish_live(
            "PLACE_OBJECT",
            "PLACE_OBJECT：检查右手中的杯子并规划返回餐桌。",
            force=True,
        )
        try:
            palm_prim, object_prim, constraint_path = (
                _resolve_held_sim_object(object_id)
            )
        except RuntimeError as exc:
            return {
                "success": False,
                "reason": f"held_object_validation_failed: {exc}",
                "object_id": object_id,
                "physical_execution": False,
            }

        target = load_reviewed_object(args.objects, object_id)
        approach = target["approach"]["pose"]
        approach_xy = (float(approach["x"]), float(approach["y"]))
        try:
            path = self.grid.plan(
                (self.pose.x, self.pose.y), approach_xy
            )
        except ValueError as exc:
            return {
                "success": False,
                "reason": f"dining_table_path_failed: {exc}",
                "object_id": object_id,
                "physical_execution": False,
            }
        navigation = self._drive(
            path,
            float(approach["yaw"]),
            precision=False,
            phase="PLACE_OBJECT",
        )
        if not navigation["success"]:
            return {
                "success": False,
                "reason": "dining_table_approach_failed",
                "object_id": object_id,
                "physical_execution": False,
                "navigation": navigation,
            }

        # The audited scan id maps to the one dynamic cup fixture. Its support
        # height and converted source-space lower bound reproduce the original
        # stable tabletop pose without asking a language model for coordinates.
        cup_fixture = next(
            item
            for item in HOUSEHOLD_OBJECTS
            if item.object_id == "dining_cup" and item.dynamic
        )
        desired_object = np.asarray(
            [
                cup_fixture.position_xy[0],
                cup_fixture.position_xy[1],
                ROOM_FLOOR_Z_M
                + cup_fixture.support_height_above_floor_m
                - cup_fixture.minimum_xyz[1],
            ],
            dtype=np.float64,
        )
        object_start = _prim_world_position(object_prim)
        palm_start = _prim_world_position(palm_prim)
        palm_object_distance = float(
            np.linalg.norm(palm_start - object_start)
        )
        if palm_object_distance > RIGHT_HAND_FINGERTIP_REACH_M + 0.03:
            return {
                "success": False,
                "reason": "held_object_outside_right_hand_envelope",
                "object_id": object_id,
                "physical_execution": False,
                "palm_object_distance_m": palm_object_distance,
            }

        # This demo uses position-only IK. With the verified grasp transform,
        # the palm sits about 0.13 m below the cup origin, so commanding the
        # cup origin directly onto the tabletop would put the palm below the
        # support surface. Move over the table, then release from the lowest
        # reachable clearance and let PhysX settle the final short descent.
        release_object = desired_object + np.asarray(
            [0.0, 0.0, 0.20], dtype=np.float64
        )
        above_object = desired_object + np.asarray(
            [0.0, 0.0, 0.30], dtype=np.float64
        )
        above_palm = palm_start + (above_object - object_start)
        above_result = move_right_palm_to(
            self.robot,
            above_palm,
            maximum_cartesian_travel_m=0.80,
            tolerance_m=0.035,
            progress_callback=lambda step, total: self._publish_live(
                "PLACE_OBJECT",
                f"PLACE_OBJECT：移动到餐桌上方 {step}/{total}",
                waypoint=step,
                waypoint_count=total,
            ),
        )
        if not above_result["success"]:
            return {
                "success": False,
                "reason": "place_above_table_ik_failed",
                "object_id": object_id,
                "physical_execution": True,
                "navigation": navigation,
                "above_table": above_result,
            }

        lower_results = []
        for correction in range(3):
            object_now = _prim_world_position(object_prim)
            palm_now = _prim_world_position(palm_prim)
            correction_target = palm_now + (release_object - object_now)
            lower_result = move_right_palm_to(
                self.robot,
                correction_target,
                maximum_cartesian_travel_m=0.18,
                tolerance_m=0.018,
                maximum_iterations=100,
                progress_callback=lambda step, total, index=correction: (
                    self._publish_live(
                        "PLACE_OBJECT",
                        f"PLACE_OBJECT：缓慢放低杯子 {index + 1}/3",
                        waypoint=step,
                        waypoint_count=total,
                    )
                ),
            )
            lower_results.append(lower_result)
            if np.linalg.norm(
                _prim_world_position(object_prim) - release_object
            ) <= 0.035:
                break
        pre_release_object = _prim_world_position(object_prim)
        placement_error = float(
            np.linalg.norm(pre_release_object - release_object)
        )
        table_fixture = next(
            fixture
            for fixture in HOME_FIXTURES
            if fixture.fixture_id == "dining_table"
        )
        table_half_size = np.asarray(
            table_fixture.size_xyz[:2], dtype=np.float64
        ) / 2.0
        table_center = np.asarray(
            table_fixture.center_xy, dtype=np.float64
        )
        tabletop_inset_m = 0.12
        inside_tabletop = bool(
            np.all(
                np.abs(pre_release_object[:2] - table_center)
                <= table_half_size - tabletop_inset_m
            )
        )
        release_height_above_support = float(
            pre_release_object[2] - desired_object[2]
        )
        safe_release_pose = bool(
            inside_tabletop
            and 0.10 <= release_height_above_support <= 0.35
        )
        if not safe_release_pose:
            return {
                "success": False,
                "reason": "cup_outside_safe_tabletop_release_volume",
                "object_id": object_id,
                "physical_execution": True,
                "navigation": navigation,
                "above_table": above_result,
                "lowering": lower_results,
                "object_pre_release_world_m": pre_release_object.tolist(),
                "pre_release_error_m": placement_error,
                "inside_tabletop": inside_tabletop,
                "release_height_above_support_m": (
                    release_height_above_support
                ),
            }

        open_result = _set_right_hand(
            self.robot,
            RIGHT_HAND_OPEN_RAD,
            progress_callback=lambda step, total: self._publish_live(
                "PLACE_OBJECT",
                f"PLACE_OBJECT：张开右手 {step}/{total}",
                waypoint=step,
                waypoint_count=total,
            ),
        )
        stage = stage_utils.get_current_stage()
        if constraint_path:
            stage.RemovePrim(constraint_path)
        object_rigid_body = UsdPhysics.RigidBodyAPI(object_prim)
        object_rigid_body.GetKinematicEnabledAttr().Set(False)
        object_rigid_body.CreateVelocityAttr().Set(
            Gf.Vec3f(0.0, 0.0, -0.15)
        )
        app_utils.update_app(steps=8)
        if constraint_path and stage.GetPrimAtPath(constraint_path).IsValid():
            return {
                "success": False,
                "reason": "grasp_constraint_removal_failed",
                "object_id": object_id,
                "physical_execution": True,
            }

        # Pull the open hand back and upward before checking that the cup stays
        # on the table independently of the gripper.
        object_after_release = _prim_world_position(object_prim)
        palm_after_release = _prim_world_position(palm_prim)
        planar = object_after_release[:2] - np.asarray(
            [self.pose.x, self.pose.y], dtype=np.float64
        )
        planar_norm = float(np.linalg.norm(planar))
        direction = (
            planar / planar_norm
            if planar_norm > 1e-6
            else np.asarray([0.0, 1.0], dtype=np.float64)
        )
        retract_target = palm_after_release.copy()
        retract_target[:2] -= direction * 0.12
        retract_target[2] += 0.08
        retract_result = move_right_palm_to(
            self.robot,
            retract_target,
            maximum_cartesian_travel_m=0.18,
            tolerance_m=0.030,
            maximum_iterations=100,
            progress_callback=lambda step, total: self._publish_live(
                "PLACE_OBJECT",
                f"PLACE_OBJECT：右手撤离杯子 {step}/{total}",
                waypoint=step,
                waypoint_count=total,
            ),
        )

        positions = []
        for frame in range(45):
            simulation_app.update()
            positions.append(_prim_world_position(object_prim))
            self._publish_live(
                "VERIFY_PLACE",
                f"VERIFY_PLACE：确认杯子稳定留在餐桌 {frame + 1}/45",
                waypoint=frame + 1,
                waypoint_count=45,
            )
        assisted_settle = False
        # Some Kit/PhysX builds leave a rigid body asleep after deleting a
        # fixed joint. If it did not descend, place it just above the audited
        # support pose while kinematic, then restore dynamics and wake it. This
        # mirrors the demo's explicitly declared assisted base-motion mode.
        if (
            positions
            and float(positions[-1][2] - desired_object[2]) > 0.08
        ):
            assisted_settle = True
            object_rigid_body.GetKinematicEnabledAttr().Set(True)
            UsdGeom.XformCommonAPI(object_prim).SetTranslate(
                Gf.Vec3d(
                    float(desired_object[0]),
                    float(desired_object[1]),
                    float(desired_object[2] + 0.006),
                )
            )
            app_utils.update_app(steps=5)
            object_rigid_body.GetKinematicEnabledAttr().Set(False)
            object_rigid_body.CreateVelocityAttr().Set(
                Gf.Vec3f(0.0, 0.0, -0.03)
            )
            positions = []
            for frame in range(60):
                simulation_app.update()
                positions.append(_prim_world_position(object_prim))
                self._publish_live(
                    "VERIFY_PLACE",
                    f"VERIFY_PLACE：唤醒刚体并确认桌面支撑 {frame + 1}/60",
                    waypoint=frame + 1,
                    waypoint_count=60,
                )
        stable = np.asarray(positions[-30:], dtype=np.float64)
        final_object = stable[-1]
        stable_range = np.ptp(stable, axis=0)
        final_xy_error = float(
            np.linalg.norm(final_object[:2] - desired_object[:2])
        )
        final_z_error = abs(float(final_object[2] - desired_object[2]))
        final_inside_tabletop = bool(
            np.all(
                np.abs(final_object[:2] - table_center)
                <= table_half_size - tabletop_inset_m
            )
        )
        success = bool(
            retract_result["success"]
            and final_inside_tabletop
            and final_z_error <= 0.06
            and float(np.max(stable_range)) <= 0.015
        )
        payload = {
            "success": success,
            "reason": (
                "cup_released_and_stable_on_dining_table"
                if success
                else "post_release_table_stability_gate_failed"
            ),
            "object_id": object_id,
            "physical_execution": True,
            "execution_environment": "isaac_sim_only",
            "navigation": navigation,
            "constraint_path": constraint_path,
            "constraint_removed": bool(constraint_path),
            "constraint_used_for_grasp": bool(constraint_path),
            "desired_object_world_m": desired_object.tolist(),
            "release_object_world_m": release_object.tolist(),
            "object_start_world_m": object_start.tolist(),
            "object_pre_release_world_m": pre_release_object.tolist(),
            "object_final_world_m": final_object.tolist(),
            "pre_release_error_m": placement_error,
            "inside_tabletop_at_release": inside_tabletop,
            "release_height_above_support_m": release_height_above_support,
            "final_inside_tabletop": final_inside_tabletop,
            "final_xy_error_m": final_xy_error,
            "final_z_error_m": final_z_error,
            "stable_window_axis_range_m": stable_range.tolist(),
            "stable_frames": 30 if float(np.max(stable_range)) <= 0.015 else 0,
            "physx_wake_velocity_mps": -0.15,
            "assisted_settle_after_sleeping_body": assisted_settle,
            "open_hand": open_result,
            "above_table": above_result,
            "lowering": lower_results,
            "retract": retract_result,
            "hardware_output": False,
        }
        evidence_dir = self.output_dir / "place"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / (
            f"{time.strftime('%Y%m%dT%H%M%SZ')}-place-cup.json"
        )
        evidence_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        payload["evidence_path"] = str(evidence_path)
        self._publish_live(
            "VERIFY_PLACE",
            (
                "VERIFY_PLACE：杯子已稳定放到餐桌上。"
                if success
                else "VERIFY_PLACE：杯子放置后的稳定性验证失败。"
            ),
            force=True,
            result=payload,
        )
        return payload

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
        if args.execute_openvla_actions:
            # v14 observations are captured after the high-reach arm is in
            # place; using the earlier visibility-gate frame removes the hand
            # that the policy relies on for closed-loop spatial feedback.
            if not save_camera_rgb(self.camera, image_path):
                return SkillResult(
                    command.command_id,
                    SkillStatus.FAILED,
                    "OpenVLA post-staging RGB camera produced no image.",
                    FailureCode.BAD_VIEWPOINT,
                    {"application_id": self.application_id},
                )
        else:
            shutil.copy2(staged_observation, image_path)

        instruction = args.openvla_instruction.strip() or (
            "pick the cup from the dining table"
        )
        process_args = [
            "/usr/bin/env",
            (
                "CUDA_VISIBLE_DEVICES="
                + os.environ.get("G1D_SIDECAR_CUDA_DEVICE", "0").strip()
            ),
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
        if args.openvla_adapter is not None:
            process_args.extend(["--adapter", str(args.openvla_adapter)])
        if args.openvla_action_head is not None:
            process_args.extend(
                ["--action-head", str(args.openvla_action_head)]
            )
        if args.openvla_dataset_statistics is not None:
            process_args.extend(
                [
                    "--dataset-statistics",
                    str(args.openvla_dataset_statistics),
                ]
            )
        child_env = os.environ.copy()
        sidecar_cuda_device = os.environ.get(
            "G1D_SIDECAR_CUDA_DEVICE", "0"
        ).strip()
        if sidecar_cuda_device:
            child_env["CUDA_VISIBLE_DEVICES"] = sidecar_cuda_device
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
        using_oft_sidecar = "openvla-oft" in str(args.openvla_python)
        legacy_openvla_site = ROOT / "envs/openvla/lib/python3.10/site-packages"
        legacy_mobilemanibench_site = (
            ROOT / "envs/mobilemanibench/lib/python3.10/site-packages"
        )
        child_env["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT),
                *(
                    [str(ROOT / "third_party/openvla-oft")]
                    if using_oft_sidecar
                    else []
                ),
                *(
                    [str(legacy_openvla_site)]
                    if legacy_openvla_site.is_dir() and not using_oft_sidecar
                    else []
                ),
                *(
                    [str(legacy_mobilemanibench_site)]
                    if legacy_mobilemanibench_site.is_dir() and not using_oft_sidecar
                    else []
                ),
            ]
        )
        child_env["PATH"] = os.pathsep.join(
            [str(args.openvla_python.parent), "/usr/local/bin", "/usr/bin", "/bin"]
        )
        conda_lib = str(args.openvla_python.parent.parent / "lib")
        legacy_openvla_lib = ROOT / "envs/openvla/lib"
        child_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                *(
                    [str(legacy_openvla_lib)]
                    if legacy_openvla_lib.is_dir() and not using_oft_sidecar
                    else []
                ),
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
        action_chunk = inference.get("action_chunk", [list(action.values)])
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
        if args.execute_openvla_actions:
            handoff["execution_permitted"] = bool(target["manipulation_ready"])
            handoff["execution_mode"] = (
                "isaac_sim_only_bounded_world_delta_ik"
            )
            handoff["joint_command"] = (
                "computed_online_by_bounded_dls_ik_per_action"
            )
            handoff["blocked_reasons"] = (
                []
                if target["manipulation_ready"]
                else [
                    "reviewed_object_is_search_and_docking_only_not_manipulation_ready"
                ]
            )
        handoff_path.write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            args.execute_sim_pick
            or args.expert_pick
            or args.execute_openvla_actions
        ):
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
            if args.expert_pick:
                evidence = self._execute_expert_pick(target)
            elif args.execute_openvla_actions:
                evidence = self._execute_openvla_action_chunk(
                    target, action_chunk
                )
                rollout_evidence = [dict(evidence)]
                rollout_inferences = [str(result_path)]
                rollout_start_z = float(evidence.get(
                    "object_start_world_m", [0.0, 0.0, 0.0]
                )[2])
                # v14 episodes contain roughly 52-61 control samples while
                # one OFT prediction contains eight.  Re-observe and re-plan
                # after every executed chunk instead of declaring failure at
                # 0.8 seconds.  Eight chunks cover the accepted trajectory
                # length while keeping a hard safety/time bound.
                for rollout_index in range(1, 8):
                    final_z = float(evidence.get(
                        "object_final_world_m", [0.0, 0.0, 0.0]
                    )[2])
                    total_lift = final_z - rollout_start_z
                    verified = bool(
                        total_lift >= 0.05
                        and int(evidence.get("stable_hold_frames", 0)) >= 30
                        and float(evidence.get(
                            "final_palm_object_distance_m", float("inf")
                        )) <= 0.16
                    )
                    if evidence.get("success") or verified:
                        evidence["success"] = True
                        evidence["reason"] = (
                            "openvla_v14_receding_horizon_lift_verified"
                        )
                        break

                    rollout_root = root / f"rollout_{rollout_index:02d}"
                    rollout_root.mkdir(parents=True, exist_ok=True)
                    rollout_image = rollout_root / "head_rgb.png"
                    rollout_result = rollout_root / "inference.json"
                    rollout_log = rollout_root / "sidecar.log"
                    if not save_camera_rgb(self.camera, rollout_image):
                        evidence["reason"] = (
                            "openvla_rollout_camera_produced_no_image"
                        )
                        break
                    self._publish_live(
                        "OPENVLA_PICK",
                        (
                            "OPENVLA_PICK：滚动闭环推理 "
                            f"{rollout_index + 1}/8"
                        ),
                        waypoint=rollout_index + 1,
                        waypoint_count=8,
                        force=True,
                    )
                    rollout_args = list(process_args)
                    rollout_args[rollout_args.index("--image") + 1] = str(
                        rollout_image
                    )
                    rollout_args[rollout_args.index("--output") + 1] = str(
                        rollout_result
                    )
                    rollout_timed_out = False
                    with rollout_log.open("w", encoding="utf-8") as log:
                        rollout_process = subprocess.Popen(
                            rollout_args,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=child_env,
                        )
                        rollout_started = time.monotonic()
                        while (
                            rollout_process.poll() is None
                            and simulation_app.is_running()
                        ):
                            self.robot.apply_wheel_actions(
                                np.zeros(2, dtype=np.float32)
                            )
                            simulation_app.update()
                            if (
                                time.monotonic() - rollout_started
                                > args.openvla_timeout_sec
                            ):
                                rollout_process.terminate()
                                try:
                                    rollout_process.wait(timeout=15)
                                except subprocess.TimeoutExpired:
                                    rollout_process.kill()
                                    rollout_process.wait(timeout=15)
                                rollout_timed_out = True
                                break
                    if (
                        rollout_timed_out
                        or rollout_process.returncode != 0
                        or not rollout_result.is_file()
                    ):
                        evidence["reason"] = (
                            "openvla_receding_horizon_sidecar_failed"
                        )
                        evidence["failed_rollout_log"] = str(rollout_log)
                        break
                    rollout_payload = json.loads(
                        rollout_result.read_text(encoding="utf-8")
                    )
                    rollout_action = OpenVlaAction.from_values(
                        rollout_payload.get("action", []),
                        unnorm_key=args.openvla_unnorm_key,
                    )
                    rollout_chunk = rollout_payload.get(
                        "action_chunk", [list(rollout_action.values)]
                    )
                    evidence = self._execute_openvla_action_chunk(
                        target, rollout_chunk
                    )
                    evidence["openvla_action"] = rollout_action.to_dict()
                    evidence["openvla_inference"] = str(rollout_result)
                    rollout_evidence.append(dict(evidence))
                    rollout_inferences.append(str(rollout_result))

                final_z = float(evidence.get(
                    "object_final_world_m", [0.0, 0.0, rollout_start_z]
                )[2])
                total_lift = final_z - rollout_start_z
                if (
                    total_lift >= 0.05
                    and int(evidence.get("stable_hold_frames", 0)) >= 30
                    and float(evidence.get(
                        "final_palm_object_distance_m", float("inf")
                    )) <= 0.16
                ):
                    evidence["success"] = True
                    evidence["reason"] = (
                        "openvla_v14_receding_horizon_lift_verified"
                    )
                evidence["rollout_count"] = len(rollout_evidence)
                evidence["rollout_inferences"] = rollout_inferences
                evidence["rollout_evidence"] = rollout_evidence
                evidence["total_lift_height_m"] = total_lift
            else:
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
        from g1d_dual_brain_agent.planner import (
            compile_family_home_command,
            compile_family_home_selection,
        )

        places_catalog = json.loads(args.places.read_text(encoding="utf-8"))
        objects_catalog = json.loads(args.objects.read_text(encoding="utf-8"))
        if args.mission_json is not None and args.mission_json.is_file():
            # Load pre-compiled mission from dashboard or saved artifact
            from g1d_dual_brain_agent.models import Mission
            mission = Mission.from_dict(
                json.loads(args.mission_json.read_text(encoding="utf-8"))
            )
        else:
            # Prefer LLM-powered go-pick-return parsing; degrade to rule grammar
            try:
                _lingbot_src = ROOT / "lingbot_semantic_nav/src"
                if str(_lingbot_src) not in sys.path:
                    sys.path.insert(0, str(_lingbot_src))
                from lingbot_nav.config import load_dotenv
                load_dotenv(ROOT / ".env")
                from family_home_vln.task_intent import FamilyTaskIntentResolver
                resolver = FamilyTaskIntentResolver(
                    places_catalog, objects_catalog,
                )
                resolution = resolver.resolve(args.command)
                if resolution.task_type == "go_pick_return":
                    mission = compile_family_home_selection(
                        args.command,
                        outbound_place_id=resolution.outbound_place_id,
                        object_id=resolution.object_id,
                        return_place_id=resolution.return_place_id,
                        places_catalog=places_catalog,
                        objects_catalog=objects_catalog,
                        mission_id=f"family-home-{int(time.time())}",
                    )
                elif resolution.task_type == "vln_navigation":
                    # LLM interpreted it as a single-place nav → wrap as mission
                    from g1d_dual_brain_agent.models import (
                        GoalKind,
                        Mission,
                        TaskGoal,
                    )
                    mission = Mission(
                        mission_id=f"family-home-{int(time.time())}",
                        instruction=args.command.strip(),
                        goals=(
                            TaskGoal(
                                goal_id="navigate-1",
                                kind=GoalKind.NAVIGATE,
                                instruction=resolution.destination_place_id,
                                success_condition=(
                                    f"机器人到达审核地点 "
                                    f"{resolution.destination_place_id}"
                                ),
                            ),
                        ),
                        maximum_transitions=12,
                        maximum_attempts_per_skill=3,
                    )
                else:
                    raise ValueError(
                        f"LLM 返回了不支持的任务类型：{resolution.task_type}"
                    )
            except Exception:
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


class InteractiveCommandServer:
    """Thread-safe HTTP command ingress for a live SimpleRoom SimulationApp.

    The HTTP worker never accesses USD, PhysX, or Kit.  The main Isaac thread
    drains ``commands`` and owns every simulation mutation.
    """

    def __init__(self, host: str, port: int) -> None:
        self.commands: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._state: dict[str, object] = {
            "state": "idle",
            "message": "Isaac 已就绪；请输入导航指令。",
            "command": "",
            "target": "",
            "updated_at": time.time(),
        }
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/api/state":
                    self._send_json(owner.state())
                    return
                if path != "/":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = _INTERACTIVE_COMMAND_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] != "/api/command":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(size).decode("utf-8"))
                    action = str(payload.get("action", "")).strip()
                    command = str(payload.get("command", "")).strip()
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                    self._send_json({"error": "命令必须是 JSON。"}, HTTPStatus.BAD_REQUEST)
                    return
                if action == "reset":
                    owner.submit_reset()
                    self._send_json(owner.state(), HTTPStatus.ACCEPTED)
                    return
                if not command:
                    self._send_json({"error": "请输入导航指令。"}, HTTPStatus.BAD_REQUEST)
                    return
                owner.submit(command)
                self._send_json(owner.state(), HTTPStatus.ACCEPTED)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="simple-room-command-http",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def update(self, **values: object) -> None:
        with self._lock:
            self._state.update(values)
            self._state["updated_at"] = time.time()

    def submit(self, command: str) -> None:
        self.commands.put(command)
        self.update(
            state="queued",
            message="指令已排队，等待 Isaac 仿真线程执行。",
            command=command,
            target="",
        )

    def submit_reset(self) -> None:
        self.commands.put("__RESET_FAMILY_HOME_SCENE__")
        self.update(
            state="queued",
            message="恢复请求已排队，等待 Isaac 仿真线程执行。",
            command="恢复初始状态",
            target="initial-state",
        )


_INTERACTIVE_COMMAND_PAGE = """<!doctype html>
<html lang=\"zh-CN\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>G1-D 家庭任务控制台</title>
<style>body{max-width:720px;margin:48px auto;padding:0 18px;font:16px system-ui;background:#10151b;color:#e9eef4}input,button{font:inherit;padding:12px;border-radius:8px}input{width:min(500px,70%)}button{margin-left:8px;background:#70d6a6;border:0;color:#092016;font-weight:700}.reset{background:#f6c85f;color:#3b2800}#state{margin-top:24px;padding:16px;background:#19232e;border-radius:8px;white-space:pre-wrap}.hint{color:#aab9c9}</style>
<h1>G1-D 家庭任务控制台</h1><p class=\"hint\">可输入导航，或“去—拿—返回”任务；执行过程始终在同一 Isaac Sim 中，直接在 noVNC 桌面观察机器人。</p>
<form id=\"form\"><input id=\"command\" autofocus value=\"请带我去餐厅，拿杯子，然后回到客厅沙发旁\" aria-label=\"任务指令\"><button>执行</button><button type=\"button\" id=\"reset\" class=\"reset\">恢复初始状态</button></form><pre id=\"state\">正在连接…</pre>
<script>const s=document.querySelector('#state'),i=document.querySelector('#command');async function post(p){let r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});let x=await r.json();if(x.error)alert(x.error);poll()}async function poll(){try{let r=await fetch('/api/state'),x=await r.json();s.textContent=`状态：${x.state}\\n${x.message}\\n指令：${x.command||'—'}\\n目标：${x.target||'—'}`}catch(e){s.textContent='控制服务暂不可用：'+e}}document.querySelector('#form').onsubmit=e=>{e.preventDefault();post({command:i.value})};document.querySelector('#reset').onclick=()=>post({action:'reset'});poll();setInterval(poll,500);</script></html>"""


def update_interactive_goal_visual(path: list[tuple[float, float]], target: Pose2D) -> None:
    """Update the existing target and route without reloading the USD scene."""

    stage = stage_utils.get_current_stage()
    goal = stage.GetPrimAtPath("/World/VLN/Goal")
    if goal.IsValid():
        ops = UsdGeom.Xformable(goal).GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(target.x, target.y, ROOM_FLOOR_Z_M + 0.025))
    route = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/VLN/PlannedPath"))
    route.CreateCurveVertexCountsAttr([len(path)])
    route.CreatePointsAttr(
        [Gf.Vec3f(x, y, ROOM_FLOOR_Z_M + 0.045) for x, y in path]
    )


def restore_family_home_initial_state(
    robot: WheeledRobot,
    camera,
    initial_dof_positions: np.ndarray,
) -> Pose2D:
    """Restore the robot and all household props without restarting Kit."""

    stage = stage_utils.get_current_stage()
    # There have been two grasp backends.  The active Expert backend creates
    # a FixedJoint directly at G1DExpertGraspConstraint; leaving it in the
    # stage makes PhysX immediately pull a just-reset cup back into the hand.
    # Remove both known forms before changing any rigid-body transforms.
    for grasp_path in (
        "/World/G1DSimGrasp",
        "/World/G1DExpertGraspConstraint",
    ):
        if stage.GetPrimAtPath(grasp_path).IsValid():
            stage.RemovePrim(grasp_path)
    # Allow the stage/PhysX relationship removal to take effect before props
    # are made kinematic and moved back onto their reviewed support surfaces.
    app_utils.update_app(steps=2)
    for index, item in enumerate(HOUSEHOLD_OBJECTS, start=1):
        root = stage.GetPrimAtPath(f"/World/FamilyHomeObjects/Item{index:02d}")
        if not root.IsValid():
            continue
        if item.dynamic:
            rigid_body = UsdPhysics.RigidBodyAPI(root)
            rigid_body.GetKinematicEnabledAttr().Set(True)
            rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        # The scene authoring creates translate + rotateZ ops.  Set those
        # exact operations instead of XformCommonAPI: the latter can add an
        # extra rotateXYZ op, leaving the old rotateZ in the transform stack.
        target_translate = Gf.Vec3d(
            item.position_xy[0],
            item.position_xy[1],
            ROOM_FLOOR_Z_M
            + item.support_height_above_floor_m
            - item.minimum_xyz[1],
        )
        translate_attr = root.GetAttribute("xformOp:translate")
        rotate_z_attr = root.GetAttribute("xformOp:rotateZ")
        if not translate_attr.IsValid():
            raise RuntimeError(
                f"household prop has unexpected transform ops: {root.GetPath()}"
            )
        translate_attr.Set(target_translate)
        # Dynamic assets such as Item05 are intentionally authored with a
        # translate-only stack.  Do not require a rotateZ op that was never
        # present; reset it when available and clear the Expert-added orient
        # op below when it is not.
        if rotate_z_attr.IsValid():
            rotate_z_attr.Set(float(item.yaw_deg))
        # Clean up any transform ops created by an older reset implementation.
        rotate_xyz_attr = root.GetAttribute("xformOp:rotateXYZ")
        if rotate_xyz_attr.IsValid():
            rotate_xyz_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        orient_attr = root.GetAttribute("xformOp:orient")
        if orient_attr.IsValid():
            orient_attr.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    reset_pose = (
        CGS_OFFICE_START
        if args.scene_profile == "cgs-office"
        else FAMILY_HOME_START
    )
    # Teleport both articulation state and targets.  Targets alone do not
    # instantly undo an arm posture left by the Expert controller.
    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    try:
        robot.set_dof_positions(initial_dof_positions)
    except (AttributeError, TypeError):
        # Isaac versions exposing only targets still get the verified root
        # reset below; keep compatibility with the project image.
        pass
    robot.set_dof_position_targets(initial_dof_positions)
    set_assisted_robot_pose(robot, reset_pose, 0.0, 0.0)
    if camera is not None:
        camera.set_world_pose(
            *camera_world_pose(reset_pose), camera_axes="world"
        )
    app_utils.update_app(steps=4)
    # PhysX may update once while consuming the transform changes.  Reassert
    # the exact root state afterwards so the user sees the real initial pose,
    # rather than an intermediate settling pose.
    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    robot.set_dof_position_targets(initial_dof_positions)
    set_assisted_robot_pose(robot, reset_pose, 0.0, 0.0)

    # Do not report success unless the robot and every household root are
    # actually back at their catalogued locations.
    observed_pose = robot_pose(robot)
    if math.hypot(observed_pose.x - reset_pose.x, observed_pose.y - reset_pose.y) > 0.02:
        raise RuntimeError(
            "robot root reset did not stick: "
            f"observed=({observed_pose.x:.3f}, {observed_pose.y:.3f})"
        )
    for index, item in enumerate(HOUSEHOLD_OBJECTS, start=1):
        root = stage.GetPrimAtPath(f"/World/FamilyHomeObjects/Item{index:02d}")
        if not root.IsValid():
            continue
        observed = _prim_world_position(root)
        expected = np.asarray(
            (
                item.position_xy[0],
                item.position_xy[1],
                ROOM_FLOOR_Z_M + item.support_height_above_floor_m - item.minimum_xyz[1],
            ),
            dtype=np.float64,
        )
        if float(np.linalg.norm(observed - expected)) > 0.02:
            raise RuntimeError(
                f"prop reset did not stick for Item{index:02d}: "
                f"error={float(np.linalg.norm(observed - expected)):.3f} m"
            )
    return reset_pose


def run_family_cup_grasp_calibration(
    robot, camera, grid, places, output_dir: Path, *,
    third_person_camera=None, wrist_camera=None,
    initial_dof_positions: np.ndarray | None = None,
    num_episodes: int = 0,
) -> int:
    """Run one repeatable, fixed-pose physical cup-grasp diagnostic only."""

    # This pose is the reviewed arm hand-off beside the dining table from the
    # normal approach-and-align phase.  It removes every unrelated subsystem
    # from the experiment: no route planning, RGB search, language model, or
    # VLA action is executed here.
    calibration_pose = Pose2D(1.8668, 2.3501, 1.7477)
    # Every standalone collection starts from the freshly loaded USD posture,
    # never from a previous Expert target.  The calibration root placement is
    # applied only after all joints have been restored and held.
    if initial_dof_positions is not None:
        baseline = np.asarray(initial_dof_positions, dtype=np.float32).copy()
        robot.set_dof_positions(baseline)
        robot.set_dof_position_targets(baseline)
    set_assisted_robot_pose(robot, calibration_pose, 0.0, 0.0)
    for _ in range(60):
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        simulation_app.update()

    session = FamilyHomeDualAgentSession(
        robot, camera, grid, places, output_dir,
        third_person_camera=third_person_camera,
        wrist_camera=wrist_camera,
    )
    session.pose = calibration_pose
    # Reassert root and joint targets after PhysX consumes the first teleport.
    if initial_dof_positions is not None:
        robot.set_dof_position_targets(baseline)
    session._upright_torso()
    set_assisted_robot_pose(robot, calibration_pose, 0.0, 0.0)
    for _ in range(30):
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        simulation_app.update()
    # A calibration recording deliberately bypasses VLN and live search, but
    # it must still use the same reviewed, cup-visible RGB observation pose as
    # the full pipeline.  This marks that fixed pose as the audited handoff so
    # --record-expert-demo records fresh camera frames instead of returning
    # early for the absence of a navigation/search handoff.
    if args.record_expert_demo:
        session.last_handoff_gate = {
            "ready": True,
            "selected": {
                "robot_pose": {
                    # Move the RGB viewpoint 0.30 m back from the table and
                    # raise it slightly.  The previous 35-degree view put
                    # the near table fascia across the manipulation area.
                    "x": 2.10,
                    "y": 1.70,
                    "yaw": math.radians(102.0),
                },
                "camera_downward_pitch_deg": 25.0,
                "camera_height_offset_m": 0.18,
                "source": "raised_rear_calibration_observation_pose",
            },
        }
    target = load_reviewed_object(args.objects, "scan_coffee_cup_05")
    result = session._execute_expert_pick(target)
    report = {
        "schema_version": 1,
        "mode": "fixed_pose_physical_cup_grasp_calibration",
        "base_pose": {
            "x": calibration_pose.x,
            "y": calibration_pose.y,
            "yaw": calibration_pose.yaw,
        },
        "no_vln": True,
        "no_rgb_search": True,
        "no_llm": True,
        "no_openvla_inference": True,
        "result": result,
    }
    report_path = output_dir / "grasp_calibration" / "latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "Grasp calibration: "
        f"success={result.get('success')} reason={result.get('reason')} "
        f"report={report_path}",
        flush=True,
    )
    simulation_app.close()
    return 0 if result.get("success") else 1


def run_interactive_navigation_session(
    robot, camera, grid, places, pose: Pose2D,
) -> int:
    """Keep one GUI SimulationApp alive for navigation and family missions."""

    server = InteractiveCommandServer(args.interactive_host, args.interactive_port)
    # Keep the live joint configuration from the freshly loaded USD.  The
    # reset action restores this snapshot rather than assuming a particular
    # G1-D joint ordering or a hard-coded home pose.
    initial_dof_positions = robot.get_dof_positions().numpy()[0].copy()
    # ── Force torso/waist joints to zero (upright posture) ──────────────
    # The G1-D variant (g1_d.usd) has NO Pitching_Joint and NO LZ_ot_Joint.
    # The upper body is rigidly attached to the base; forward lean comes
    # from the entire base tipping around the wheel axis under gravity.
    # Camera compensation is handled by head_camera_pose().
    UPRIGHT_TORSO_JOINTS = [
        "LZ_mt_Joint",
        "LZ_it_Joint",
        "Yaw_Joint",
        "torso_Joint",
    ]
    for _joint_name in UPRIGHT_TORSO_JOINTS:
        try:
            _idx = int(robot.get_dof_indices([_joint_name]).numpy()[0])
        except Exception:
            continue
        initial_dof_positions[_idx] = 0.0
    # Right arm starts neutral.  The unused left arm uses the USD-reviewed
    # vertical pose: its elbow must be +pi/2, not zero.
    for _joint in RIGHT_ARM_JOINTS:
        try:
            _idx = int(robot.get_dof_indices([_joint]).numpy()[0])
        except Exception:
            continue
        initial_dof_positions[_idx] = 0.0
    _write_left_arm_vertical_pose(robot, initial_dof_positions)
    robot.set_dof_position_targets(initial_dof_positions)
    # Let the torso settle toward upright before accepting commands.
    for _ in range(60):
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        simulation_app.update()
    # A freshly composed articulation can retain its authored USD root pose,
    # which is not necessarily the reviewed navigation start.  Complete the
    # exact same verified reset used by the dashboard before exposing the
    # HTTP endpoint; otherwise an early command can plan from an occupied
    # stale pose while the startup settling loop is still running.
    pose = restore_family_home_initial_state(
        robot, camera, initial_dof_positions,
    )
    server.start()
    print(
        "Interactive navigation control page: "
        f"http://{args.interactive_host}:{args.interactive_port}/"
    )
    print("Open the noVNC desktop separately to watch this same SimulationApp.")
    # ── Write initial joint diagnostic for forward-lean debugging ──
    try:
        _pos = robot.get_dof_positions().numpy()[0]
        _tgt = robot.get_dof_position_targets().numpy()[0]
        _diag = {"positions": {}, "targets": {}}
        _torso_joints = [
            "LZ_mt_Joint", "LZ_it_Joint",
            "Yaw_Joint", "torso_Joint",
        ]
        for _name in _torso_joints:
            try:
                _i = int(robot.get_dof_indices([_name]).numpy()[0])
                _diag["positions"][_name] = {
                    "rad": round(float(_pos[_i]), 7),
                    "deg": round(float(np.degrees(_pos[_i])), 4),
                }
                _diag["targets"][_name] = {
                    "rad": round(float(_tgt[_i]), 7),
                    "deg": round(float(np.degrees(_tgt[_i])), 4),
                }
            except Exception:
                pass
        _all_pos, _all_quat = robot.get_world_poses()
        _base_pos = _all_pos.numpy()[0]   # AGV_link root body
        _base_quat = _all_quat.numpy()[0]
        _diag["base_quat_wxyz"] = [round(float(c), 7) for c in _base_quat]
        _w, _x, _y, _z = [float(c) for c in _base_quat]
        _sp = 2.0 * (_w * _y - _z * _x)
        _diag["base_pitch_deg"] = round(float(math.degrees(math.asin(max(-1.0, min(1.0, _sp))))), 3)
        _diag["base_position"] = {
            "x": round(float(_base_pos[0]), 4),
            "y": round(float(_base_pos[1]), 4),
            "z": round(float(_base_pos[2]), 4),
        }
        with open("/tmp/diag_startup_joints.json", "w", encoding="utf-8") as _f:
            json.dump(_diag, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        while simulation_app.is_running():
            try:
                command = server.commands.get_nowait()
            except queue.Empty:
                robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
                simulation_app.update()
                continue
            if command == "__DIAG__":
                try:
                    # Dump ALL joint names first
                    _all_names = list(robot.dof_names)
                    _pos = robot.get_dof_positions().numpy()[0]
                    _tgt = robot.get_dof_position_targets().numpy()[0]
                    _diag = {"all_joint_names": list(_all_names), "positions": {}, "targets": {}}
                    for _name in ["LZ_mt_Joint", "LZ_it_Joint",
                                   "Yaw_Joint", "torso_Joint"]:
                        try:
                            _i = int(robot.get_dof_indices([_name]).numpy()[0])
                            _diag["positions"][_name] = {"rad": round(float(_pos[_i]), 7), "deg": round(float(np.degrees(_pos[_i])), 4)}
                            _diag["targets"][_name] = {"rad": round(float(_tgt[_i]), 7), "deg": round(float(np.degrees(_tgt[_i])), 4)}
                        except Exception:
                            pass
                    _all_pos, _all_quat = robot.get_world_poses()
                    _base_pos = _all_pos.numpy()[0]
                    _base_quat = _all_quat.numpy()[0]
                    _diag["base_quat_wxyz"] = [round(float(c), 7) for c in _base_quat]
                    _w2, _x2, _y2, _z2 = [float(c) for c in _base_quat]
                    _sp2 = 2.0 * (_w2 * _y2 - _z2 * _x2)
                    _diag["base_pitch_deg"] = round(float(math.degrees(math.asin(max(-1.0, min(1.0, _sp2))))), 3)
                    _diag["base_position"] = {"x": round(float(_base_pos[0]), 4), "y": round(float(_base_pos[1]), 4), "z": round(float(_base_pos[2]), 4)}
                    with open("/tmp/diag_joints.json", "w", encoding="utf-8") as _f:
                        json.dump(_diag, _f, ensure_ascii=False, indent=2)
                    server.update(state="succeeded", command=command, message="Joint diagnostic written to /tmp/diag_joints.json")
                except Exception as _exc:
                    server.update(state="failed", command=command, message=f"Diag failed: {_exc}")
                continue
                try:
                    _pos = robot.get_dof_positions().numpy()[0]
                    _tgt = robot.get_dof_position_targets().numpy()[0]
                    _diag = {"positions": {}, "targets": {}}
                    for _name in ["LZ_mt_Joint", "LZ_it_Joint",
                                   "Yaw_Joint", "torso_Joint"]:
                        try:
                            _i = int(robot.get_dof_indices([_name]).numpy()[0])
                            _diag["positions"][_name] = {"rad": round(float(_pos[_i]), 7), "deg": round(float(np.degrees(_pos[_i])), 4)}
                            _diag["targets"][_name] = {"rad": round(float(_tgt[_i]), 7), "deg": round(float(np.degrees(_tgt[_i])), 4)}
                        except Exception:
                            pass
                    _all_pos, _all_quat = robot.get_world_poses()
                    _base_pos = _all_pos.numpy()[0]
                    _base_quat = _all_quat.numpy()[0]
                    _diag["base_quat_wxyz"] = [round(float(c), 7) for c in _base_quat]
                    _w2, _x2, _y2, _z2 = [float(c) for c in _base_quat]
                    _sp2 = 2.0 * (_w2 * _y2 - _z2 * _x2)
                    _diag["base_pitch_deg"] = round(float(math.degrees(math.asin(max(-1.0, min(1.0, _sp2))))), 3)
                    _diag["base_position"] = {"x": round(float(_base_pos[0]), 4), "y": round(float(_base_pos[1]), 4), "z": round(float(_base_pos[2]), 4)}
                    with open("/tmp/diag_joints.json", "w", encoding="utf-8") as _f:
                        json.dump(_diag, _f, ensure_ascii=False, indent=2)
                    server.update(state="succeeded", command=command, message="Joint diagnostic written to /tmp/diag_joints.json")
                except Exception as _exc:
                    server.update(state="failed", command=command, message=f"Diag failed: {_exc}")
                continue
            if command == "__RESET_FAMILY_HOME_SCENE__":
                server.update(
                    state="running",
                    command="恢复初始状态",
                    target="initial-state",
                    message="正在恢复机器人、杯子和家庭场景的初始状态。",
                )
                try:
                    pose = restore_family_home_initial_state(
                        robot, camera, initial_dof_positions,
                    )
                    # A reset is a new session boundary: do not accidentally
                    # execute commands that were entered before the reset.
                    while True:
                        try:
                            server.commands.get_nowait()
                        except queue.Empty:
                            break
                    server.update(
                        state="succeeded",
                        command="恢复初始状态",
                        target="initial-state",
                        message="已恢复初始状态：机器人、杯子和场景已复位。",
                        result={
                            "reset": True,
                            "robot_pose": {
                                "x": pose.x,
                                "y": pose.y,
                                "yaw": pose.yaw,
                            },
                        },
                    )
                except Exception as exc:
                    server.update(
                        state="failed",
                        command="恢复初始状态",
                        target="initial-state",
                        message=(
                            "恢复初始状态失败："
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                continue
            is_place_task = any(
                marker in command
                for marker in ("放到", "放在", "放下", "放回", "摆到", "摆在")
            )
            if is_place_task:
                if not (args.dual_agent and args.family_task):
                    server.update(
                        state="rejected",
                        command=command,
                        message=(
                            "放置任务需要以 --dual-agent --family-task 启动。"
                        ),
                    )
                    continue
                if not any(marker in command for marker in ("杯", "cup", "mug")):
                    server.update(
                        state="rejected",
                        command=command,
                        message="当前放置技能只接受已审核的餐桌杯子。",
                    )
                    continue
                if not any(marker in command for marker in ("桌", "餐厅", "餐区")):
                    server.update(
                        state="rejected",
                        command=command,
                        message="请明确说把杯子放到餐桌上。",
                    )
                    continue
                server.update(
                    state="running",
                    command=command,
                    target="place-cup-on-dining-table",
                    message=(
                        "正在返回餐桌、放低杯子、松开右手并验证稳定性。"
                    ),
                )
                try:
                    session = FamilyHomeDualAgentSession(
                        robot, camera, grid, places, args.output_dir,
                    )
                    payload = session.place_carried_cup_on_dining_table()
                    summary_path = (
                        args.output_dir / "place_cup_interactive_summary.json"
                    )
                    summary_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    server.update(
                        state="succeeded" if payload["success"] else "failed",
                        command=command,
                        target="place-cup-on-dining-table",
                        message=(
                            "杯子已稳定放到餐桌上。"
                            if payload["success"]
                            else f"杯子未能放稳：{payload['reason']}"
                        ),
                        result=payload,
                        summary=str(summary_path),
                    )
                except Exception as exc:
                    server.update(
                        state="failed",
                        command=command,
                        target="place-cup-on-dining-table",
                        message=f"放置任务未执行：{type(exc).__name__}: {exc}",
                    )
                finally:
                    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
                    pose = robot_pose(robot)
                continue
            is_family_task = any(marker in command for marker in ("拿", "取", "抓"))
            if is_family_task:
                if not (args.dual_agent and args.family_task):
                    server.update(
                        state="rejected",
                        command=command,
                        message=(
                            "抓取任务需要以 --dual-agent --family-task 启动；"
                            "当前实例仅允许导航。"
                        ),
                    )
                    continue
                server.update(
                    state="running",
                    command=command,
                    target="go-pick-return",
                    message=(
                        "Dual Brain 正在执行：导航、RGB 搜索、精确对齐、"
                        "OpenVLA、抓取验证与返回。"
                    ),
                )
                previous_command = args.command
                try:
                    # The Executive owns all skill transitions. Temporarily
                    # supplying the queued command reuses the audited family
                    # compiler without allowing the HTTP worker to touch Kit.
                    args.command = command
                    session = FamilyHomeDualAgentSession(
                        robot, camera, grid, places, args.output_dir,
                    )
                    payload = run_dual_agent_session(session)
                    summary_path = args.output_dir / "dual_agent_interactive_summary.json"
                    summary_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    server.update(
                        state=(
                            "succeeded"
                            if payload["status"] == "succeeded"
                            else "failed"
                        ),
                        target="go-pick-return",
                        message=(
                            "家庭任务完成。"
                            if payload["status"] == "succeeded"
                            else f"家庭任务未完成：{payload['message']}"
                        ),
                        result=payload,
                        summary=str(summary_path),
                    )
                except Exception as exc:
                    server.update(
                        state="failed",
                        command=command,
                        target="go-pick-return",
                        message=f"家庭任务未执行：{type(exc).__name__}: {exc}",
                    )
                finally:
                    args.command = previous_command
                    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
                    pose = robot_pose(robot)
                continue
            target = None
            try:
                target = resolve_place(command, places, reference=robot_pose(robot))
            except ValueError:
                # Rule-based alias matching failed – try LLM semantic resolution
                try:
                    _lingbot_src = ROOT / "lingbot_semantic_nav/src"
                    if str(_lingbot_src) not in sys.path:
                        sys.path.insert(0, str(_lingbot_src))
                    from lingbot_nav.config import load_dotenv
                    load_dotenv(ROOT / ".env")
                    from family_home_vln.intent import FamilyIntentResolver
                    resolver = FamilyIntentResolver(args.places)
                    semantic = resolver.resolve(command)
                    # Build lookup from the full JSON so LLM can also select
                    # provisional / rejected places that still exist in the
                    # catalog (e.g. kitchen_counter in formal mode).
                    from simple_room_vln.core import Place as _Place
                    catalog = json.loads(
                        args.places.read_text(encoding="utf-8")
                    )
                    places_all = {
                        item.get("id")
                        or item.get("place_id", ""): _Place(
                            place_id=item.get("id")
                            or item.get("place_id", ""),
                            name=item.get("name", ""),
                            aliases=tuple(item.get("aliases", [])),
                            pose=Pose2D(
                                float(
                                    item.get("entrance_pose", {}).get(
                                        "x", 0.0
                                    )
                                ),
                                float(
                                    item.get("entrance_pose", {}).get(
                                        "y", 0.0
                                    )
                                ),
                                float(
                                    item.get("entrance_pose", {}).get(
                                        "yaw", 0.0
                                    )
                                ),
                            ),
                            status=item.get("status", "approved"),
                        )
                        for item in catalog.get("places", [])
                        if item.get("id") or item.get("place_id")
                    }
                    target = places_all.get(semantic.place_id)
                    if target is None:
                        raise ValueError(
                            f"LLM 返回了地图中不存在的地点："
                            f"{semantic.place_id}"
                        )
                except Exception as llm_exc:
                    server.update(
                        state="rejected",
                        message="指令未执行：规则匹配与 LLM 解析均未成功",
                        command=command,
                    )
            try:
                path = grid.plan((pose.x, pose.y), (target.pose.x, target.pose.y))
            except Exception as exc:
                server.update(
                    state="rejected",
                    message=f"指令未执行：{type(exc).__name__}: {exc}",
                    command=command,
                )
                continue
            update_interactive_goal_visual(path, target.pose)
            follower = PathFollower(
                path, goal_yaw=target.pose.yaw, max_linear=0.45, max_angular=1.10,
            )
            server.update(
                state="running", command=command, target=target.place_id,
                message=f"正在前往 {target.place_id}。",
            )
            frame = 0
            while simulation_app.is_running() and not follower.done:
                observed = robot_pose(robot) if args.wheel_physics_only else pose
                linear, angular, label = follower.command(observed)
                robot.apply_wheel_actions(command_to_wheel_velocities(linear, angular))
                if not args.wheel_physics_only:
                    pose = assisted_step(pose, linear, angular)
                    set_assisted_robot_pose(robot, pose, linear, angular)
                if camera is not None:
                    camera.set_world_pose(
                        *camera_world_pose(
                            observed if args.wheel_physics_only else pose
                        ),
                        camera_axes="world",
                    )
                simulation_app.update()
                frame += 1
                if frame % 10 == 0:
                    server.update(
                        state="running",
                        message=(
                            f"正在导航：{label}，航点 "
                            f"{follower.index}/{len(path) - 1}。"
                        ),
                    )
            robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
            app_utils.update_app(steps=5)
            pose = robot_pose(robot) if args.wheel_physics_only else pose
            error = math.dist((pose.x, pose.y), path[-1])
            server.update(
                state="succeeded" if follower.done else "failed",
                message=(
                    f"已到达 {target.place_id}，位置误差 {error:.3f} m。"
                    if follower.done
                    else "Isaac 在到达前停止了本次导航。"
                ),
            )
    finally:
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        server.stop()
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Grasp-demo data collection (--collect-grasp-demos N)
# ═══════════════════════════════════════════════════════════════════════════

_GRASP_INSTRUCTIONS = [
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
]


def run_grasp_demo_collection(
    robot,
    camera,
    output_dir: Path,
    *,
    num_episodes: int = 50,
    seed: int = 42,
    base_variation_xy_m: float = 0.15,
    base_variation_yaw_deg: float = 10.0,
) -> dict:
    """Collect expert grasp demos using multi-stage IK and save in OpenVLA format."""
    import numpy as np

    rng = np.random.default_rng(seed)
    demo_dir = output_dir / "grasp_demos"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Dining-area fixed reference pose (approach_and_align final pose)
    DINING_BASE_X = 1.90
    DINING_BASE_Y = 2.22
    DINING_BASE_YAW = math.radians(104.0)

    dataset_meta = {
        "schema_version": 1,
        "task": "cup_grasping",
        "robot": "g1_d_wheeled",
        "action_space": {
            "type": "continuous_7d_delta",
            "labels": ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper"],
            "unnorm_key": "bridge_orig",
        },
        "observation_space": {"type": "rgb", "resolution": [640, 480], "camera": "head_camera"},
        "instructions": _GRASP_INSTRUCTIONS,
        "episodes": [],
    }

    # Default cup target: scan_coffee_cup_05 map anchor
    default_anchor = np.array([1.746, 2.968, 0.828], dtype=np.float64)

    successful = 0
    for ep in range(num_episodes):
        ep_dir = demo_dir / f"episode_{ep:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Randomize base pose
        base_x = DINING_BASE_X + rng.uniform(-base_variation_xy_m, base_variation_xy_m)
        base_y = DINING_BASE_Y + rng.uniform(-base_variation_xy_m, base_variation_xy_m)
        base_yaw = DINING_BASE_YAW + math.radians(
            rng.uniform(-base_variation_yaw_deg, base_variation_yaw_deg)
        )

        # Teleport robot to randomized pose
        set_assisted_robot_pose(robot, base_x, base_y, base_yaw)
        _upright_torso(robot)
        robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
        for _ in range(30):
            simulation_app.update()

        # Target world Z = floor + scan anchor Z
        cup_world = np.array([
            default_anchor[0] + rng.uniform(-0.05, 0.05),
            default_anchor[1] + rng.uniform(-0.05, 0.05),
            ROOM_FLOOR_Z_M + default_anchor[2],
        ], dtype=np.float64)

        instruction = rng.choice(_GRASP_INSTRUCTIONS)

        print(
            f"  Demo {ep:04d}/{num_episodes}  "
            f"base=({base_x:.2f},{base_y:.2f},{math.degrees(base_yaw):.0f}°)  "
            f"\"{instruction}\"",
            end=" ",
        )

        # ── Expert trajectory (images + actions saved inline) ──────────
        try:
            frame_count, success = _collect_single_expert_trajectory(
                robot, camera, cup_world, instruction, ep_dir,
            )
        except Exception as exc:
            print(f"✗ {exc}")
            import traceback
            traceback.print_exc()
            continue

        ep_meta = {
            "episode_id": ep,
            "instruction": instruction,
            "base_pose": {"x": base_x, "y": base_y, "yaw": base_yaw},
            "cup_world_m": cup_world.tolist(),
            "success": success,
            "frame_count": frame_count,
        }
        (ep_dir / "meta.json").write_text(
            json.dumps(ep_meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        if success:
            successful += 1
            print("✓")
        else:
            print("✗ IK failed")

        dataset_meta["episodes"].append(ep_meta)
        app_utils.update_app(steps=5)

    dataset_meta["summary"] = {
        "total_episodes": num_episodes,
        "successful_episodes": successful,
        "total_frames": sum(ep["frame_count"] for ep in dataset_meta["episodes"]),
    }
    (demo_dir / "dataset.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\nDone: {successful}/{num_episodes} episodes, "
          f"{dataset_meta['summary']['total_frames']} frames → {demo_dir}")
    return dataset_meta


def _collect_single_expert_trajectory(
    robot, camera, cup_world, instruction: str,
    ep_dir: Path,
    *,
    capture_hz: float = 10.0,
) -> tuple[int, bool]:
    """Capture a continuous expert trajectory, saving (image + action) inline.

    Images are captured at *capture_hz* (default 10 Hz) **during** IK
    execution via the ``progress_callback`` hook — each saved frame
    corresponds to the actual rendered view at that simulation step.

    Returns (frame_count, success).
    """
    import numpy as np

    sim_ticks_per_capture = max(1, int(round(PHYSICS_HZ / capture_hz)))
    prev_palm = link_world_position(robot, RIGHT_PALM_LINK).copy()
    _tick_counter = 0
    _gripper_cmd = 1.0  # open by default
    _frame_idx = 0

    def _capture():
        """Save (image + action) for the current sim step."""
        nonlocal prev_palm, _frame_idx
        step_dir = ep_dir / f"step_{_frame_idx:04d}"
        step_dir.mkdir(exist_ok=True)

        # Image — rendered frame for the sim step that just completed
        if not save_camera_rgb(camera, step_dir / "image.png"):
            return  # skip if camera read fails

        # Action delta since last capture
        palm_now = link_world_position(robot, RIGHT_PALM_LINK)
        delta = palm_now - prev_palm
        import json as _json
        action = {
            "dx_m": float(delta[0]),
            "dy_m": float(delta[1]),
            "dz_m": float(delta[2]),
            "droll_rad": 0.0,
            "dpitch_rad": 0.0,
            "dyaw_rad": 0.0,
            "gripper": _gripper_cmd,
            "labels": ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper"],
            "unnorm_key": "bridge_orig",
        }
        (step_dir / "action.json").write_text(
            _json.dumps(action, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        prev_palm = palm_now.copy()
        _frame_idx += 1

    def _on_sim_step(_iteration: int, _max_iter: int) -> None:
        """Called every ``simulation_app.update()`` during IK loops."""
        nonlocal _tick_counter
        _tick_counter += 1
        if _tick_counter >= sim_ticks_per_capture:
            _capture()
            _tick_counter = 0

    # ── Arm seed ──
    _set_right_arm_pregrasp_seed(robot)
    app_utils.update_app(steps=sim_ticks_per_capture)
    _capture()

    # ── Open hand ──
    _set_right_hand(robot, RIGHT_HAND_OPEN_RAD)
    for _ in range(sim_ticks_per_capture):
        simulation_app.update()
        _on_sim_step(0, 1)

    base_xy = np.array([robot_pose(robot)[0], robot_pose(robot)[1]], dtype=np.float64)
    direction = cup_world[:2] - base_xy
    planar = float(np.linalg.norm(direction))
    direction /= planar

    # ── Stage 1: vertical table clearance ──
    palm_now = link_world_position(robot, RIGHT_PALM_LINK)
    clearance_z = max(float(palm_now[2]), cup_world[2] + 0.18)
    move_right_palm_to(
        robot,
        np.array([float(palm_now[0]), float(palm_now[1]), clearance_z], dtype=np.float64),
        maximum_cartesian_travel_m=0.40,
        tolerance_m=0.035,
        progress_callback=_on_sim_step,
    )

    # ── Stage 2: overhead pregrasp ──
    overhead_target = cup_world.copy()
    overhead_target[2] = clearance_z
    move_right_palm_to(
        robot, overhead_target,
        maximum_cartesian_travel_m=0.70,
        tolerance_m=0.035,
        progress_callback=_on_sim_step,
    )

    # ── Stage 3: pregrasp ──
    pregrasp = cup_world.copy()
    pregrasp[:2] -= direction * 0.07
    pregrasp[2] += 0.04
    res = move_right_palm_to(
        robot, pregrasp,
        maximum_cartesian_travel_m=0.30,
        tolerance_m=0.045,
        progress_callback=_on_sim_step,
    )
    if not res["success"]:
        return _frame_idx, False

    # ── Stage 4: grasp approach ──
    grasp = cup_world.copy()
    grasp[:2] -= direction * 0.05
    res = move_right_palm_to(
        robot, grasp,
        maximum_cartesian_travel_m=0.20,
        tolerance_m=0.060,
        progress_callback=_on_sim_step,
    )
    if not res["success"]:
        return _frame_idx, False

    # ── Close hand ──
    try:
        palm_prim, object_prim, _ = _find_sim_grasp_bodies(cup_world)
    except RuntimeError as exc:
        print(f"  body_resolution: {exc}")
        return _frame_idx, False

    _configure_physical_grasp_friction(object_prim)
    _set_right_hand(robot, RIGHT_HAND_CLOSED_RAD)
    _gripper_cmd = 0.0
    for _ in range(sim_ticks_per_capture):
        simulation_app.update()
        _on_sim_step(0, 1)

    # ── Grasp constraint + settle ──
    constraint_path = _create_sim_grasp_constraint(palm_prim, object_prim, "grasp_collect")
    for _ in range(15):
        simulation_app.update()
        _on_sim_step(0, 1)

    # ── Lift ──
    lift_target = link_world_position(robot, RIGHT_PALM_LINK) + np.array(
        [0.0, 0.0, 0.09], dtype=np.float64,
    )
    move_right_palm_to(
        robot, lift_target,
        maximum_cartesian_travel_m=0.14,
        tolerance_m=0.025,
        progress_callback=_on_sim_step,
    )

    # ── Stable hold ──
    for _ in range(45):
        simulation_app.update()
        _on_sim_step(0, 1)

    # ── Clean up constraint ──
    try:
        stage_utils.get_current_stage().RemovePrim(constraint_path)
    except Exception:
        pass
    app_utils.update_app(steps=8)

    return _frame_idx, True


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.right_arm_probe:
        print("Mode: bounded G1-D right-arm kinematics probe")
    if args.collect_grasp_demos > 0:
        print(f"Mode: collect {args.collect_grasp_demos} expert grasp demonstrations")
    if args.survey or args.allow_bootstrap or args.collect_grasp_demos > 0:
        if args.scene_profile == "cgs-office":
            raise SystemExit(
                "cgs-office has a reviewed formal catalog; --survey/--allow-bootstrap "
                "are not supported for this profile"
            )
        if args.scene_profile == "family-home":
            grid, places = build_family_home_bootstrap_artifacts(args.output_dir)
            map_source = "reviewed_procedural_family_home_bootstrap"
        elif args.scene_profile == "living-room":
            from living_room_vln.layout import build_survey_grid

            grid = build_survey_grid()
            places = []
            map_source = "survey_coverage_only_not_a_navigation_map"
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
        grid, places = load_lingbot_artifacts(
            args.map,
            args.places,
            robot_radius_m=(
                0.3
                if args.scene_profile == "cgs-office"
                else ROBOT_RADIUS_M
            ),
        )
        map_source = (
            "lingbot_rgb_depth+isaac_survey_pose_offline_diagnostic"
            if args.map.parent.name == "lingbot_pose_fused_map"
            else "lingbot_map_rgb_only_pointcloud"
        )
    if args.collect_grasp_demos > 0:
        # Data-collection mode skips navigation task planning entirely.
        # The collector manages its own cup placement and robot base poses.
        # Add two close points to avoid "Incorrect number of vertices" bezier
        # curve warnings that can upset the renderer / PhysX scene loading.
        target = None
        path = [(1.90, 2.22), (1.901, 2.22)]
        final_yaw = 0.0
        task_name = "grasp_demo_collection"
    elif args.survey:
        target = None
        path = (
            build_family_home_survey_path(grid)
            if args.scene_profile == "family-home"
            else (
                __import__("living_room_vln.layout", fromlist=["build_survey_path"])
                .build_survey_path(grid)
                if args.scene_profile == "living-room"
                else build_simple_survey_path(grid)
            )
        )
        final_yaw = 0.0
        task_name = "rgb_survey"
    elif args.family_task and args.interactive_port is None:
        from g1d_dual_brain_agent.planner import (
            compile_family_home_command,
            compile_family_home_selection,
        )

        places_catalog = json.loads(args.places.read_text(encoding="utf-8"))
        objects_catalog = json.loads(args.objects.read_text(encoding="utf-8"))
        try:
            _lingbot_src = ROOT / "lingbot_semantic_nav/src"
            if str(_lingbot_src) not in sys.path:
                sys.path.insert(0, str(_lingbot_src))
            from lingbot_nav.config import load_dotenv
            load_dotenv(ROOT / ".env")
            from family_home_vln.task_intent import FamilyTaskIntentResolver
            resolver = FamilyTaskIntentResolver(
                places_catalog, objects_catalog,
            )
            resolution = resolver.resolve(args.command)
            if resolution.task_type == "go_pick_return":
                preview_mission = compile_family_home_selection(
                    args.command,
                    outbound_place_id=resolution.outbound_place_id,
                    object_id=resolution.object_id,
                    return_place_id=resolution.return_place_id,
                    places_catalog=places_catalog,
                    objects_catalog=objects_catalog,
                    mission_id="family-home-preview",
                )
            else:
                raise ValueError(
                    f"LLM 返回了不支持的任务类型：{resolution.task_type}"
                )
        except Exception:
            preview_mission = compile_family_home_command(
                args.command,
                places_catalog=places_catalog,
                objects_catalog=objects_catalog,
                mission_id="family-home-preview",
            )
        places_by_id = {place.place_id: place for place in places}
        outbound = places_by_id[preview_mission.goals[0].instruction]
        target = places_by_id[preview_mission.goals[2].instruction]
        start = FAMILY_HOME_START
        outbound_path = grid.plan(
            (start.x, start.y), (outbound.pose.x, outbound.pose.y)
        )
        return_path = grid.plan(
            (outbound.pose.x, outbound.pose.y),
            (target.pose.x, target.pose.y),
        )
        path = outbound_path + return_path[1:]
        final_yaw = target.pose.yaw
        task_name = "family_go_pick_return"
    elif args.family_task:
        # The browser supplies the mission after Isaac has loaded.  Do not
        # compile the CLI's placeholder command before the control page is up.
        target = None
        start = FAMILY_HOME_START
        path = [(start.x, start.y), (start.x + 0.001, start.y)]
        final_yaw = start.yaw
        task_name = "interactive_family_go_pick_return"
    elif args.interactive_port is not None:
        # The control page supplies commands after Isaac has loaded; do not
        # compile the CLI's placeholder command before the page is up.
        target = None
        start = (
            CGS_OFFICE_START
            if args.scene_profile == "cgs-office"
            else FAMILY_HOME_START
            if args.scene_profile == "family-home"
            else Pose2D(0.0, 0.0, 0.0)
        )
        path = [(start.x, start.y), (start.x + 0.001, start.y)]
        final_yaw = start.yaw
        task_name = "interactive_navigation"
    else:
        start = (
            FAMILY_HOME_START
            if args.scene_profile == "family-home"
            else (
                CGS_OFFICE_START
                if args.scene_profile == "cgs-office"
                else Pose2D(0.0, 0.0, 0.0)
            )
        )
        target = resolve_place(args.command, places, reference=start)
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
        positions=[
            path[0][0],
            path[0][1],
            (0.0 if args.scene_profile in ("living-room", "cgs-office") else ROOM_FLOOR_Z_M) + 0.12,
        ],
    )
    # The v14 dining-cup policy was trained without deployment-only fingertip
    # pad colliders.  Preserve that contact geometry here; non-v14 objects can
    # still request their own contact profile when their manipulation skill is
    # selected.
    grasp_stage = stage_utils.get_current_stage()
    # Physics materials are parsed into PhysX shapes at simulation startup as
    # well.  Binding them later updates USD metadata but can leave the live
    # cup/finger shapes using their default low-friction material.
    staged_cup = grasp_stage.GetPrimAtPath(
        "/World/FamilyHomeObjects/Item05"
    )
    if staged_cup.IsValid():
        _configure_physical_grasp_friction(staged_cup)

    camera = None
    third_person_camera = None
    wrist_camera = None
    recorder = None

    overview_camera = None
    gif_frames = []
    if args.record_gif is not None or live is not None:
        if args.record_fps <= 0 or args.record_fps > PHYSICS_HZ:
            raise ValueError("--record-fps must be between 1 and 60")

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
    print("[collect] Starting simulation...")
    app_utils.play()
    app_utils.update_app(steps=24)
    write_grasp_scene_diagnostics(stage_utils.get_current_stage())
    print("[collect] Simulation running, configuring joints...")
    configure_joint_drives(robot)
    # Snapshot the authored USD pose before any assisted base placement or
    # Expert target is applied.  Standalone grasp collection restores this
    # exact vector at the beginning of every episode.
    initial_dof_positions = robot.get_dof_positions().numpy()[0].copy()
    _write_left_arm_vertical_pose(robot, initial_dof_positions)
    robot.set_dof_positions(initial_dof_positions)
    robot.set_dof_position_targets(initial_dof_positions)

    pose = Pose2D(path[0][0], path[0][1], 0.0)
    if not args.wheel_physics_only:
        set_assisted_robot_pose(robot, pose, 0.0, 0.0)

    # Camera render products must be created only after Isaac's timeline is
    # playing.  Creating them while the stage is being assembled can leave
    # the RTX Hydra texture without a GPU foundation in headless Isaac 6;
    # both RGB annotators then remain None forever.  This follows the proven
    # lifecycle used by the v14 expert-data collector.
    if not args.no_camera:
        from isaacsim.sensors.camera import Camera

        camera_position, camera_orientation = camera_world_pose(pose)
        camera = Camera(
            # Match the proven v14 collector camera namespace.  In this
            # Isaac build, world-root camera prims can create a render product
            # before RTX binds a backing texture in headless Kit.
            prim_path="/World/Sensors/G1DHeadCamera",
            position=camera_position,
            orientation=camera_orientation,
            frequency=30,
            resolution=(640, 480),
        )
        if args.grasp_calibration and args.record_expert_demo:
            third_eye = np.asarray([1.66, 4.18, 1.32], dtype=np.float32)
            third_target = np.asarray([1.84, 2.42, 0.43], dtype=np.float32)
            third_person_camera = Camera(
                prim_path="/World/G1DThirdPersonGraspCamera",
                position=third_eye,
                orientation=look_at_camera_pose(third_eye, third_target)[1],
                frequency=20,
                resolution=(640, 480),
            )
            wrist_camera = Camera(
                prim_path="/World/G1DRightWristCamera",
                position=np.asarray([1.84, 2.42, 0.43], dtype=np.float32),
                orientation=look_at_camera_pose(
                    [1.84, 2.42, 0.43], [1.84, 3.02, 0.10]
                )[1],
                frequency=20,
                resolution=(640, 480),
            )

    if args.record_gif is not None or live is not None:
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
            home_chase_camera_pose(pose)
            if live is not None and args.scene_profile == "family-home"
            else look_at_camera_pose(overview_eye, overview_target)
        )
        overview_camera = Camera(
            prim_path="/World/Sensors/VLNOverviewCamera",
            position=overview_position,
            orientation=overview_orientation,
            frequency=30,
            resolution=(640, 480),
        )

    if camera is not None:
        camera.initialize()
        camera.set_focal_length(CAMERA_FOCAL_LENGTH_MM)
        camera.set_horizontal_aperture(CAMERA_HORIZONTAL_APERTURE_MM)
        camera.set_vertical_aperture(CAMERA_VERTICAL_APERTURE_MM)
        camera.set_clipping_range(
            near_distance=CAMERA_NEAR_CLIP_M,
            far_distance=CAMERA_FAR_CLIP_M,
        )
        camera.set_world_pose(*camera_world_pose(pose), camera_axes="world")
        app_utils.update_app(steps=10)
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
    if third_person_camera is not None:
        third_person_camera.initialize()
        third_person_camera.set_focal_length(24.0)
        third_person_camera.set_horizontal_aperture(28.0)
        app_utils.update_app(steps=12)
    if wrist_camera is not None:
        wrist_camera.initialize()
        app_utils.update_app(steps=12)

    print(f"[collect] Mode dispatch: interactive={args.interactive_port is not None} probe={args.right_arm_probe} collect={args.collect_grasp_demos}")

    if args.grasp_calibration:
        print("Mode: fixed-pose physical cup-grasp calibration", flush=True)
        return run_family_cup_grasp_calibration(
            robot, camera, grid, places, args.output_dir,
            third_person_camera=third_person_camera,
            wrist_camera=wrist_camera,
            initial_dof_positions=initial_dof_positions,
        )

    if args.interactive_port is not None:
        return run_interactive_navigation_session(
            robot, camera, grid, places, pose,
        )

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

    if args.collect_grasp_demos > 0:
        print("[collect] Entering grasp demo collection mode...")
        payload = run_grasp_demo_collection(
            robot,
            camera,
            args.output_dir,
            num_episodes=args.collect_grasp_demos,
        )
        manifest_path = args.output_dir / "grasp_demos" / "dataset.json"
        outcome = "success" if payload["success"] else "failure"
        print(
            f"Grasp demo collection: {outcome} — "
            f"collected {payload['episodes_collected']}/{payload['episodes_requested']} "
            f"episodes, {payload['total_steps']} action frames, "
            f"{payload['total_grasp_successes']} grasp successes."
        )
        print(f"Dataset manifest: {manifest_path}")
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
            final_overview = (
                camera_rgb(overview_camera)
                if overview_camera is not None
                else None
            )
            if final_overview is not None:
                live.publish_image(final_overview, stream="overview")
            final_robot_view = camera_rgb(camera) if camera is not None else None
            if final_robot_view is not None:
                live.publish_image(final_robot_view, stream="robot")
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
            update_camera_pose(robot, camera)
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
            overview_image = (
                camera_rgb(overview_camera)
                if overview_camera is not None
                else None
            )
            if overview_image is not None:
                live.publish_image(overview_image, stream="overview")
            robot_image = camera_rgb(camera) if camera is not None else None
            if robot_image is not None:
                live.publish_image(robot_image, stream="robot")
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
            else "home_lab"
            if args.scene_profile == "living-room"
            else "cgs_office"
            if args.scene_profile == "cgs-office"
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
