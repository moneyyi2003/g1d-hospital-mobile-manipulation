r"""Survey and navigate the Isaac Sim Hospital scene with G1-D.

This executable is the simulator boundary of the Hospital semantic-navigation
pipeline.  It records the robot RGB camera for LingBot-Map, records a chase
camera for human inspection, and executes paths from either explicit bootstrap
geometry or formal LingBot/SAM3 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parent
ROOM_USD = ROOT / "Assets/room/IsaacSim/Hospital.usd"
ROBOT_USD = ROOT / "Assets/g1_d_robot/g1_d.usd"
DEFAULT_OUTPUT = ROOT / "outputs/hospital_vln"
DEFAULT_MAP = DEFAULT_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_PLACES = DEFAULT_OUTPUT / "places_formal.json"

ROBOT_PRIM_PATH = "/World/G1_D"
LEFT_WHEEL_JOINT = "Left_Wheel_Joint"
RIGHT_WHEEL_JOINT = "Right_Wheel_Joint"
WHEEL_RADIUS_M = 0.0848
WHEEL_BASE_M = 0.4062
PHYSICS_HZ = 60
FLOOR_Z_M = 0.001
ROBOT_ROOT_Z_M = 0.1065
CAMERA_HEIGHT_M = 1.34
CAMERA_FORWARD_M = 0.18
CAMERA_PITCH_RAD = math.radians(16.0)
CAMERA_FOCAL_LENGTH_MM = 16.0
CAMERA_HORIZONTAL_APERTURE_MM = 28.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run a short headless assertion")
    parser.add_argument("--survey", action="store_true", help="Record a LingBot-ready RGB survey")
    parser.add_argument("--command", default="请带我到医院前台")
    parser.add_argument(
        "--target-id",
        default="",
        help="Use a previously validated catalog place id instead of parsing --command again",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--places", type=Path, default=DEFAULT_PLACES)
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Explicitly use the measured lobby grid before LingBot artifacts exist",
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--record-gif", type=Path)
    parser.add_argument("--record-fps", type=int, default=5)
    parser.add_argument(
        "--live-dir",
        type=Path,
        help="Publish live state JSON and chase-camera JPEGs for the Hospital web dashboard",
    )
    parser.add_argument("--live-fps", type=int, default=10)
    parser.add_argument("--live-resolution", default="960x540")
    parser.add_argument(
        "--viewport-mode",
        choices=("chase", "overview"),
        default="chase",
        help="GUI viewport camera mode (default: chase the robot)",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the GUI open after arrival until the window is closed",
    )
    parser.add_argument(
        "--rgb-fps",
        type=int,
        default=0,
        help="Robot-camera render rate (default: 8 for survey, 2 for navigation)",
    )
    parser.add_argument("--wheel-physics-only", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--resolution", default="480x270", help="Robot RGB WIDTHxHEIGHT")
    parser.add_argument(
        "--survey-distance-step",
        type=float,
        default=0.12,
        help="Minimum translation between saved survey RGB frames in metres",
    )
    parser.add_argument(
        "--survey-angle-step-deg",
        type=float,
        default=8.0,
        help="Minimum rotation between saved survey RGB frames in degrees",
    )
    parser.add_argument("--arrival-hold-seconds", type=float, default=0.0)
    parser.add_argument(
        "--full-hospital",
        action="store_true",
        help="Keep geometry outside the reception-area MVP (slower rendering)",
    )
    return parser.parse_args()


args = parse_args()
if args.record_gif is not None and not args.record_gif.is_absolute():
    args.record_gif = ROOT / args.record_gif
if args.live_dir is not None and not args.live_dir.is_absolute():
    args.live_dir = ROOT / args.live_dir
if args.test:
    args.headless = True
    if args.steps <= 0:
        args.steps = 1200
if args.survey:
    args.no_camera = False
    args.allow_bootstrap = True
    if args.steps <= 0:
        args.steps = 7200
if args.rgb_fps <= 0:
    args.rgb_fps = 8 if args.survey else 2
if not 1 <= args.rgb_fps <= PHYSICS_HZ:
    raise SystemExit("--rgb-fps must be between 1 and 60")

try:
    camera_width, camera_height = (
        int(value) for value in args.resolution.lower().split("x", 1)
    )
except (TypeError, ValueError) as exc:
    raise SystemExit("--resolution must be WIDTHxHEIGHT") from exc
try:
    live_width, live_height = (
        int(value) for value in args.live_resolution.lower().split("x", 1)
    )
except (TypeError, ValueError) as exc:
    raise SystemExit("--live-resolution must be WIDTHxHEIGHT") from exc
if not 1 <= args.live_fps <= 30:
    raise SystemExit("--live-fps must be between 1 and 30")
if live_width <= 0 or live_height <= 0:
    raise SystemExit("--live-resolution dimensions must be positive")

for required in (ROOM_USD, ROBOT_USD):
    if not required.is_file():
        raise FileNotFoundError(required)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1440,
        "height": 900,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 1,
        # The Hospital asset is large; its default viewport otherwise renders
        # at 30 Hz even in headless mode.  Sensor render products remain active.
        "disable_viewport_updates": args.headless,
    }
)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
from pxr import Gf, Usd, UsdGeom, UsdLux

from hospital_vln.artifacts import (
    HOSPITAL_START,
    build_bootstrap_artifacts,
    build_survey_path,
)
from hospital_vln.live import LivePublisher, publish_failure
from simple_room_vln.artifacts import load_lingbot_artifacts
from simple_room_vln.core import PathFollower, Pose2D, path_length, resolve_place


def command_to_wheels(linear: float, angular: float) -> np.ndarray:
    left = (linear - angular * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
    right = -(linear + angular * WHEEL_BASE_M / 2.0) / WHEEL_RADIUS_M
    return np.array([left, right], dtype=np.float32)


def configure_joint_drives(robot: WheeledRobot) -> None:
    names = robot.dof_names
    stiffness = np.zeros(len(names), dtype=np.float32)
    damping = np.zeros(len(names), dtype=np.float32)
    for index, name in enumerate(names):
        if name in (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT):
            damping[index] = 20.0
        elif name in ("LZ_mt_Joint", "LZ_it_Joint"):
            stiffness[index], damping[index] = 2000.0, 150.0
        elif "hand_" in name:
            stiffness[index], damping[index] = 10.0, 1.0
        else:
            stiffness[index], damping[index] = 80.0, 8.0
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


def set_assisted_pose(robot: WheeledRobot, pose: Pose2D, linear: float, angular: float) -> None:
    orientation = np.array(
        [math.cos(pose.yaw / 2.0), 0.0, 0.0, math.sin(pose.yaw / 2.0)],
        dtype=np.float32,
    )
    robot.set_world_poses(
        positions=np.array([pose.x, pose.y, ROBOT_ROOT_Z_M], dtype=np.float32),
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
            pose.x + CAMERA_FORWARD_M * math.cos(pose.yaw),
            pose.y + CAMERA_FORWARD_M * math.sin(pose.yaw),
            FLOOR_Z_M + CAMERA_HEIGHT_M,
        ],
        dtype=np.float32,
    )
    cy, sy = math.cos(pose.yaw / 2.0), math.sin(pose.yaw / 2.0)
    cp, sp = math.cos(CAMERA_PITCH_RAD / 2.0), math.sin(CAMERA_PITCH_RAD / 2.0)
    orientation = np.array([cy * cp, -sy * sp, cy * sp, sy * cp], dtype=np.float32)
    return position, orientation


def look_at_pose(
    eye: Sequence[float], target: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    eye_array = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_array
    forward /= np.linalg.norm(forward)
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
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
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = math.sqrt(1.0 + rotation[index, index] - rotation[j, j] - rotation[k, k]) * 2.0
        quaternion = np.zeros(4, dtype=np.float32)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (rotation[k, j] - rotation[j, k]) / scale
        quaternion[j + 1] = (rotation[j, index] + rotation[index, j]) / scale
        quaternion[k + 1] = (rotation[k, index] + rotation[index, k]) / scale
    return eye_array.astype(np.float32), quaternion


def chase_camera_pose(pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
    eye, target = chase_camera_view(pose)
    return look_at_pose(eye, target)


def chase_camera_view(
    pose: Pose2D,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    eye = (
        pose.x - 4.0 * math.cos(pose.yaw) - 1.8 * math.sin(pose.yaw),
        pose.y - 4.0 * math.sin(pose.yaw) + 1.8 * math.cos(pose.yaw),
        2.8,
    )
    target = (
        pose.x + 1.0 * math.cos(pose.yaw),
        pose.y + 1.0 * math.sin(pose.yaw),
        0.65,
    )
    return eye, target


def camera_rgb(camera) -> np.ndarray | None:
    rgba = camera.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    image = np.asarray(rgba)[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(
            image * (255.0 if float(image.max()) <= 1.0 else 1.0), 0, 255
        ).astype(np.uint8)
    return image


def add_scene(path: Sequence[tuple[float, float]], target: Pose2D | None) -> None:
    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage_utils.add_reference_to_stage(str(ROOM_USD), "/World/HospitalScene")
    stage = stage_utils.get_current_stage()
    if not args.full_hospital:
        # Hospital.usd references the complete 76 x 42 metre NVIDIA hospital.
        # The first task only uses the reception ROI.  Deactivating disjoint
        # top-level assets prevents RTX from compiling hundreds of invisible
        # room materials while retaining every wall/object intersecting the ROI.
        hospital = stage.GetPrimAtPath("/World/HospitalScene/hospital")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        roi = (-12.0, -3.0, 5.0, 13.0)
        kept = deactivated = 0
        for child in hospital.GetChildren():
            bounds = cache.ComputeWorldBound(child).ComputeAlignedRange()
            if bounds.IsEmpty():
                kept += 1
                continue
            lower, upper = bounds.GetMin(), bounds.GetMax()
            disjoint = (
                upper[0] < roi[0]
                or lower[0] > roi[2]
                or upper[1] < roi[1]
                or lower[1] > roi[3]
            )
            if disjoint:
                child.SetActive(False)
                deactivated += 1
            else:
                kept += 1
        print(f"Hospital reception ROI: kept={kept}, deactivated={deactivated}")
    dome = UsdLux.DomeLight.Define(stage, "/World/VLN/DomeLight")
    dome.CreateIntensityAttr(350.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    if target is not None:
        marker = UsdGeom.Cylinder.Define(stage, "/World/VLN/Goal")
        marker.CreateAxisAttr("Z")
        marker.CreateRadiusAttr(0.22)
        marker.CreateHeightAttr(0.04)
        marker.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.05, 0.05)])
        UsdGeom.Xformable(marker).AddTranslateOp().Set(
            Gf.Vec3d(target.x, target.y, FLOOR_Z_M + 0.025)
        )
    route = UsdGeom.BasisCurves.Define(stage, "/World/VLN/PlannedPath")
    route.CreateTypeAttr("linear")
    route.CreateCurveVertexCountsAttr([len(path)])
    route.CreatePointsAttr([Gf.Vec3f(x, y, FLOOR_Z_M + 0.05) for x, y in path])
    route.CreateWidthsAttr([0.06])
    route.SetWidthsInterpolation("constant")
    route.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])


class SurveyRecorder:
    def __init__(
        self,
        output: Path,
        camera,
        intrinsics: np.ndarray,
        *,
        distance_step_m: float,
        angle_step_rad: float,
    ) -> None:
        self.root = output / "survey"
        self.rgb_dir = self.root / "rgb"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.rgb_dir.glob("*.png"):
            stale.unlink()
        self.camera = camera
        self.intrinsics = np.asarray(intrinsics).tolist()
        self.frames: list[dict] = []
        self.last_pose: Pose2D | None = None
        self.distance_step_m = distance_step_m
        self.angle_step_rad = angle_step_rad

    def capture(self, pose: Pose2D, *, force: bool = False) -> bool:
        if self.last_pose is not None and not force:
            distance = math.dist((pose.x, pose.y), (self.last_pose.x, self.last_pose.y))
            angle = abs(math.atan2(math.sin(pose.yaw - self.last_pose.yaw), math.cos(pose.yaw - self.last_pose.yaw)))
            if distance < self.distance_step_m and angle < self.angle_step_rad:
                return False
        image = camera_rgb(self.camera)
        if image is None:
            return False
        from PIL import Image

        index = len(self.frames)
        name = f"{index:06d}.png"
        Image.fromarray(image).save(self.rgb_dir / name)
        camera_position, camera_orientation = camera_world_pose(pose)
        self.frames.append(
            {
                "frame": index,
                "image": f"rgb/{name}",
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
            "scene": "IsaacSim/Hospital.usd reception survey",
            "source_usd": str(ROOM_USD),
            "rgb_is_only_model_input": True,
            "pose_consumer": "offline_metric_alignment_and_evaluation_only",
            "camera": {
                "resolution": [camera_width, camera_height],
                "intrinsics": self.intrinsics,
                "horizontal_fov_deg": math.degrees(
                    2.0 * math.atan(CAMERA_HORIZONTAL_APERTURE_MM / (2.0 * CAMERA_FOCAL_LENGTH_MM))
                ),
                "height_above_floor_m": CAMERA_HEIGHT_M,
                "axes": "+X forward, +Z up",
            },
            "sampling": {
                "minimum_translation_m": self.distance_step_m,
                "minimum_rotation_deg": math.degrees(self.angle_step_rad),
            },
            "frames": self.frames,
        }
        target = self.root / "capture_manifest.json"
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.survey or args.allow_bootstrap:
        grid, places = build_bootstrap_artifacts(args.output_dir)
        map_source = "isaac_geometry_bootstrap"
    else:
        missing = [str(path) for path in (args.map, args.places) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Formal Hospital navigation needs LingBot/SAM3 artifacts: " + ", ".join(missing)
            )
        grid, places = load_lingbot_artifacts(args.map, args.places)
        map_source = "lingbot_map_rgb_only_pointcloud"

    if args.survey:
        target = None
        path = build_survey_path(grid)
        goal_yaw = HOSPITAL_START.yaw
        task = "hospital_reception_rgb_survey"
    else:
        target = resolve_place(args.target_id or args.command, places)
        path = grid.plan(
            (HOSPITAL_START.x, HOSPITAL_START.y), (target.pose.x, target.pose.y)
        )
        goal_yaw = target.pose.yaw
        task = target.place_id

    print(f"Map source: {map_source}")
    print(f"Task: {task}")
    if target is not None:
        source = args.target_id or args.command
        print(f"Resolved {source!r} -> {target.place_id}")
    print(f"Planned {len(path)} waypoints, length={path_length(path):.3f} m")

    live = None
    if args.live_dir is not None:
        live = LivePublisher(
            args.live_dir,
            command="" if args.survey else args.command,
            task=task,
            map_source=map_source,
            path=path,
        )
        live.publish_state(
            state="loading",
            message="正在加载 Hospital 场景、G1-D 和 RTX 相机…",
            frame=0,
            action="loading",
            pose=HOSPITAL_START,
            linear=0.0,
            angular=0.0,
            waypoint=0,
            waypoint_count=max(0, len(path) - 1),
        )

    add_scene(path, None if target is None else target.pose)
    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT],
        usd_path=str(ROBOT_USD),
        positions=[HOSPITAL_START.x, HOSPITAL_START.y, ROBOT_ROOT_Z_M],
    )

    camera = None
    if not args.no_camera:
        from isaacsim.sensors.camera import Camera

        camera = Camera(
            prim_path="/World/G1DRgbCamera",
            position=camera_world_pose(HOSPITAL_START)[0],
            orientation=camera_world_pose(HOSPITAL_START)[1],
            frequency=args.rgb_fps,
            resolution=(camera_width, camera_height),
        )

    overview_camera = None
    gif_frames = []
    if args.record_gif is not None or live is not None:
        if not 1 <= args.record_fps <= PHYSICS_HZ:
            raise ValueError("--record-fps must be between 1 and 60")
        from isaacsim.sensors.camera import Camera

        overview_fps = max(args.record_fps if args.record_gif is not None else 1, args.live_fps)
        overview_resolution = (
            (live_width, live_height) if live is not None else (480, 270)
        )
        overview_camera = Camera(
            prim_path="/World/VLNChaseCamera",
            position=chase_camera_pose(HOSPITAL_START)[0],
            orientation=chase_camera_pose(HOSPITAL_START)[1],
            frequency=overview_fps,
            resolution=overview_resolution,
        )

    if not args.headless:
        if args.viewport_mode == "chase":
            eye, view_target = chase_camera_view(HOSPITAL_START)
        else:
            eye, view_target = [-5.5, -6.5, 4.0], [-1.5, 2.5, 0.6]
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=eye,
            target=view_target,
        )

    SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ, device="cpu")
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    app_utils.play()
    app_utils.update_app(steps=12)
    configure_joint_drives(robot)

    pose = HOSPITAL_START
    if not args.wheel_physics_only:
        set_assisted_pose(robot, pose, 0.0, 0.0)
    recorder = None
    if camera is not None:
        camera.initialize()
        camera.set_focal_length(CAMERA_FOCAL_LENGTH_MM)
        camera.set_horizontal_aperture(CAMERA_HORIZONTAL_APERTURE_MM)
        camera.set_world_pose(*camera_world_pose(pose), camera_axes="world")
        app_utils.update_app(steps=12)
        if args.survey:
            if args.survey_distance_step <= 0.0 or args.survey_angle_step_deg <= 0.0:
                raise ValueError("survey sampling steps must be positive")
            recorder = SurveyRecorder(
                args.output_dir,
                camera,
                camera.get_intrinsics_matrix(),
                distance_step_m=args.survey_distance_step,
                angle_step_rad=math.radians(args.survey_angle_step_deg),
            )
            recorder.capture(pose, force=True)
    if overview_camera is not None:
        overview_camera.initialize()
        overview_camera.set_focal_length(16.0)
        overview_camera.set_horizontal_aperture(28.0)
        overview_camera.set_world_pose(*chase_camera_pose(pose), camera_axes="world")
        app_utils.update_app(steps=12)

    follower = PathFollower(
        path,
        goal_yaw=goal_yaw,
        max_linear=0.72 if args.survey else 0.55,
        max_angular=1.20,
    )
    frame = 0
    last_label = "start"
    if live is not None:
        live.publish_state(
            state="running",
            message="场景就绪，机器人开始执行导航指令。",
            frame=0,
            action="start",
            pose=pose,
            linear=0.0,
            angular=0.0,
            waypoint=follower.index,
            waypoint_count=max(0, len(path) - 1),
        )
    while simulation_app.is_running():
        observed = robot_pose(robot) if args.wheel_physics_only else pose
        linear, angular, label = follower.command(observed)
        robot.apply_wheel_actions(command_to_wheels(linear, angular))
        if not args.wheel_physics_only:
            pose = assisted_step(pose, linear, angular)
            set_assisted_pose(robot, pose, linear, angular)
        current = robot_pose(robot) if args.wheel_physics_only else pose
        if camera is not None:
            camera.set_world_pose(*camera_world_pose(current), camera_axes="world")
        if overview_camera is not None:
            overview_camera.set_world_pose(*chase_camera_pose(current), camera_axes="world")
        if not args.headless and args.viewport_mode == "chase":
            eye, view_target = chase_camera_view(current)
            ViewportManager.set_camera_view(
                "/OmniverseKit_Persp", eye=eye, target=view_target
            )
        simulation_app.update()
        if recorder is not None:
            recorder.capture(current)
        live_due = live is not None and frame % max(1, PHYSICS_HZ // args.live_fps) == 0
        gif_due = (
            args.record_gif is not None
            and frame % max(1, PHYSICS_HZ // args.record_fps) == 0
        )
        if overview_camera is not None and (live_due or gif_due):
            image = camera_rgb(overview_camera)
            if image is not None:
                if live_due:
                    live.publish_image(image)
                if gif_due:
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
        if label != last_label or frame % 240 == 0:
            print(
                f"frame={frame:4d} state={label:7s} waypoint={follower.index}/{len(path)-1} "
                f"pose=({current.x:+.2f},{current.y:+.2f},{current.yaw:+.2f}) "
                f"cmd=({linear:+.2f},{angular:+.2f})"
            )
            last_label = label
        frame += 1
        if follower.done or (args.steps > 0 and frame >= args.steps):
            break

    robot.apply_wheel_actions(np.zeros(2, dtype=np.float32))
    app_utils.update_app(steps=5)
    final_pose = robot_pose(robot) if args.wheel_physics_only else pose
    position_error = math.dist((final_pose.x, final_pose.y), path[-1])
    yaw_error = abs(
        math.atan2(math.sin(goal_yaw - final_pose.yaw), math.cos(goal_yaw - final_pose.yaw))
    )
    manifest = None
    if recorder is not None:
        recorder.capture(final_pose, force=True)
        manifest = recorder.finish()
        print(f"Survey frames: {len(recorder.frames)} -> {manifest}")
    elif camera is not None:
        image = camera_rgb(camera)
        if image is not None:
            from PIL import Image

            Image.fromarray(image).save(args.output_dir / "arrival_rgb.png")

    if args.record_gif is not None:
        if not gif_frames:
            raise RuntimeError("chase camera produced no GIF frames")
        args.record_gif.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            args.record_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=round(1000.0 / args.record_fps),
            loop=0,
            optimize=False,
        )
        print(f"Isaac movement GIF: {args.record_gif} ({len(gif_frames)} frames)")

    result = {
        "task": task,
        "command": None if args.survey else args.command,
        "map_source": map_source,
        "execution_mode": "wheel_physics_only" if args.wheel_physics_only else "stable_assisted",
        "camera_enabled": camera is not None,
        "success": follower.done,
        "frames": frame,
        "path_length_m": path_length(path),
        "final_pose": {"x": final_pose.x, "y": final_pose.y, "yaw": final_pose.yaw},
        "position_error_m": position_error,
        "yaw_error_rad": yaw_error,
        "survey_manifest": str(manifest) if manifest else None,
        "isaac_movement_gif": str(args.record_gif) if args.record_gif else None,
    }
    summary_name = "survey_summary.json" if args.survey else "run_summary.json"
    summary_path = args.output_dir / summary_name
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if live is not None:
        live.publish_state(
            state="succeeded" if follower.done else "failed",
            message=(
                f"已到达 {task}，位置误差 {position_error:.3f} m。"
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

    if args.arrival_hold_seconds > 0:
        import time

        deadline = time.monotonic() + args.arrival_hold_seconds
        while simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
    if args.keep_open and not args.headless:
        print("Demo complete. Close the Isaac Sim window to exit.")
        while simulation_app.is_running():
            simulation_app.update()
    app_utils.stop()
    if args.test and (not follower.done or position_error > 0.20):
        print("TEST FAILED: G1-D did not reach the Hospital target", file=sys.stderr)
        return 2
    if args.test:
        connected = "Hospital scene and G1-D navigation"
        if camera is not None:
            connected += " with RGB camera"
        print(f"TEST PASSED: {connected} are connected")
    return 0


try:
    exit_code = main()
except Exception as exc:
    if args.live_dir is not None:
        publish_failure(
            args.live_dir,
            command=args.command,
            message=f"{type(exc).__name__}: {exc}",
            pose=HOSPITAL_START,
        )
    raise
finally:
    simulation_app.close()

raise SystemExit(exit_code)
