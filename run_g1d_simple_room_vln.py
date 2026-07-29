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
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parent
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
if args.scene_profile == "family-home":
    if args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = DEFAULT_HOME_OUTPUT
    if args.map == DEFAULT_LINGBOT_MAP:
        args.map = DEFAULT_HOME_LINGBOT_MAP
    if args.places == DEFAULT_FORMAL_PLACES:
        args.places = DEFAULT_HOME_FORMAL_PLACES
if args.test:
    args.headless = True
    if args.steps <= 0:
        args.steps = 1800
if args.survey:
    args.no_camera = False
    if args.steps <= 0:
        args.steps = 6000

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

for required in (ROOM_USD, ROBOT_USD, SOFA_USD):
    if not required.is_file():
        raise FileNotFoundError(required)

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
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

from family_home_vln.layout import (
    HOME_FIXTURES,
    SCENE_NAME as FAMILY_HOME_SCENE_NAME,
    START_POSE as FAMILY_HOME_START,
    build_bootstrap_artifacts as build_family_home_bootstrap_artifacts,
    build_survey_path as build_family_home_survey_path,
)
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


def camera_world_pose(pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
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
    cp = math.cos(CAMERA_DOWNWARD_PITCH_RAD / 2.0)
    sp = math.sin(CAMERA_DOWNWARD_PITCH_RAD / 2.0)
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


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.survey or args.test or args.allow_bootstrap:
        if args.scene_profile == "family-home":
            grid, places = build_family_home_bootstrap_artifacts(args.output_dir)
            map_source = "reviewed_procedural_family_home_bootstrap"
        else:
            grid, places = build_bootstrap_artifacts(args.output_dir)
            map_source = "isaac_geometry_bootstrap"
    else:
        missing = [str(path) for path in (args.map, args.places) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "正式导航拒绝退化到 Isaac 几何；缺少 LingBot/SAM3 工件："
                + ", ".join(missing)
                + "。请先运行 build_simple_room_vln.ps1，或仅调试时显式传 --allow-bootstrap。"
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
    if args.record_gif is not None or (live is not None and camera is None):
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
        "success": follower.done,
        "frames": frame,
        "path_length_m": path_length(path),
        "final_pose": {"x": final_pose.x, "y": final_pose.y, "yaw": final_pose.yaw},
        "position_error_m": position_error,
        "yaw_error_rad": yaw_error,
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
            state="succeeded" if follower.done else "failed",
            message=(
                f"已到达 {task_name}，位置误差 {position_error:.3f} m。"
                if follower.done
                else f"任务结束但未到达目标，位置误差 {position_error:.3f} m。"
            ),
            frame=frame,
            action="arrived" if follower.done else "failed",
            pose=final_pose,
            linear=0.0,
            angular=0.0,
            waypoint=follower.index,
            waypoint_count=max(0, len(path) - 1),
            result=result,
        )
    print(
        f"Result: success={follower.done} position_error={position_error:.3f} m "
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
