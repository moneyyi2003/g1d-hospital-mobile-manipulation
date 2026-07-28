r"""Navigate the real G1-D USD in MobileManiBench's multi-shelf warehouse.

The default bootstrap mode rasterizes the composed Isaac collision geometry.
It validates scene loading, semantic-place routing, path planning and the G1-D
control boundary.  Formal navigation must replace that bootstrap with the
project's LingBot RGB-only occupancy map and an approved place catalog.
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
ROBOT_USD = ROOT / "Assets/g1_d_robot/g1_d.usd"
DEFAULT_OUTPUT = ROOT / "outputs/warehouse_vln"
DEFAULT_MAP = DEFAULT_OUTPUT / "lingbot_map/map.yaml"
DEFAULT_PLACES = DEFAULT_OUTPUT / "places_formal.json"

ROBOT_PRIM_PATH = "/World/G1_D"
SCENE_PRIM_PATH = "/World/Warehouse"
LEFT_WHEEL_JOINT = "Left_Wheel_Joint"
RIGHT_WHEEL_JOINT = "Right_Wheel_Joint"
PHYSICS_HZ = 60
FLOOR_Z_M = 0.0
ROBOT_ROOT_Z_M = 0.1065
CAMERA_HEIGHT_M = 1.34
CAMERA_FORWARD_M = 0.18
CAMERA_PITCH_RAD = math.radians(16.0)
CAMERA_FOCAL_LENGTH_MM = 16.0
CAMERA_HORIZONTAL_APERTURE_MM = 28.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--survey", action="store_true")
    parser.add_argument("--command", default="请带我到东侧货架通道")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--places", type=Path, default=DEFAULT_PLACES)
    parser.add_argument(
        "--scene",
        default="",
        help="Override the MobileManiBench warehouse USD path or URL",
    )
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Use collision-derived Isaac ground truth instead of LingBot/SAM3 artifacts",
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--wheel-physics-only", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--resolution", default="480x270")
    parser.add_argument("--record-gif", type=Path)
    parser.add_argument("--record-fps", type=int, default=5)
    parser.add_argument("--position-tolerance", type=float, default=0.15)
    parser.add_argument("--yaw-tolerance", type=float, default=0.15)
    parser.add_argument("--survey-distance-step", type=float, default=0.18)
    parser.add_argument("--survey-angle-step-deg", type=float, default=10.0)
    return parser.parse_args()


args = parse_args()
if args.test:
    args.headless = True
    args.allow_bootstrap = True
    if args.steps <= 0:
        args.steps = 4200
if args.survey:
    args.allow_bootstrap = True
    args.no_camera = False
    if args.steps <= 0:
        args.steps = 9000
if args.record_gif is not None and not args.record_gif.is_absolute():
    args.record_gif = ROOT / args.record_gif
if not 0.02 <= args.position_tolerance <= 0.40:
    raise SystemExit("--position-tolerance must be between 0.02 and 0.40 m")
if not 0.02 <= args.yaw_tolerance <= 0.60:
    raise SystemExit("--yaw-tolerance must be between 0.02 and 0.60 rad")
if not 1 <= args.record_fps <= PHYSICS_HZ:
    raise SystemExit("--record-fps must be between 1 and 60")
try:
    camera_width, camera_height = (
        int(value) for value in args.resolution.lower().split("x", 1)
    )
except (TypeError, ValueError) as exc:
    raise SystemExit("--resolution must be WIDTHxHEIGHT") from exc
if camera_width <= 0 or camera_height <= 0:
    raise SystemExit("--resolution dimensions must be positive")
if not ROBOT_USD.is_file():
    raise FileNotFoundError(ROBOT_USD)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from warehouse_vln.artifacts import (
    CollisionBounds,
    WAREHOUSE_SCENE_URL,
    WAREHOUSE_START,
    build_bootstrap_artifacts,
    build_survey_path,
)
from warehouse_vln.kinematics import (
    ROOT_FROM_NAVIGATION_YAW_RAD,
    USD_ANGULAR_SIGN,
    WHEEL_BASE_M,
    WHEEL_RADIUS_M,
    navigation_twist_to_wheel_speeds,
    navigation_yaw_to_root_yaw,
    root_yaw_to_navigation_yaw,
)


scene_source = args.scene or WAREHOUSE_SCENE_URL
if "://" not in scene_source:
    scene_path = Path(scene_source).expanduser()
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    scene_source = str(scene_path.resolve())

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 1,
        "disable_viewport_updates": bool(
            args.headless
            and args.no_camera
            and args.record_gif is None
        ),
    }
)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from simple_room_vln.artifacts import load_lingbot_artifacts
from simple_room_vln.core import (
    GridMap,
    PathFollower,
    Place,
    Pose2D,
    path_length,
    resolve_place,
)


def command_to_wheels(linear: float, angular: float) -> np.ndarray:
    left, right = navigation_twist_to_wheel_speeds(linear, angular)
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
    wheel_indices = robot.get_dof_indices(
        [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]
    ).numpy().tolist()
    robot.set_dof_max_efforts([40.0, 40.0], dof_indices=wheel_indices)
    robot.set_dof_position_targets(robot.get_dof_positions().numpy()[0])


def assisted_step(
    pose: Pose2D,
    linear: float,
    angular: float,
) -> Pose2D:
    dt = 1.0 / PHYSICS_HZ
    yaw = pose.yaw + angular * dt
    return Pose2D(
        pose.x + linear * math.cos(yaw) * dt,
        pose.y + linear * math.sin(yaw) * dt,
        yaw,
    )


def set_assisted_pose(
    robot: WheeledRobot,
    pose: Pose2D,
    linear: float,
    angular: float,
) -> None:
    root_yaw = navigation_yaw_to_root_yaw(pose.yaw)
    orientation = np.array(
        [math.cos(root_yaw / 2.0), 0.0, 0.0, math.sin(root_yaw / 2.0)],
        dtype=np.float32,
    )
    robot.set_world_poses(
        positions=np.array(
            [pose.x, pose.y, ROBOT_ROOT_Z_M],
            dtype=np.float32,
        ),
        orientations=orientation,
    )
    robot.set_velocities(
        linear_velocities=[
            linear * math.cos(pose.yaw),
            linear * math.sin(pose.yaw),
            0.0,
        ],
        angular_velocities=[0.0, 0.0, angular],
    )


def robot_pose(robot: WheeledRobot) -> Pose2D:
    positions, orientations = robot.get_world_poses()
    position = positions.numpy()[0]
    quaternion = orientations.numpy()[0]
    root_yaw = math.atan2(
        2.0
        * (
            quaternion[0] * quaternion[3]
            + quaternion[1] * quaternion[2]
        ),
        1.0
        - 2.0
        * (
            quaternion[2] ** 2
            + quaternion[3] ** 2
        ),
    )
    return Pose2D(
        float(position[0]),
        float(position[1]),
        root_yaw_to_navigation_yaw(float(root_yaw)),
    )


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
    cp, sp = math.cos(CAMERA_PITCH_RAD / 2.0), math.sin(
        CAMERA_PITCH_RAD / 2.0
    )
    orientation = np.array(
        [cy * cp, -sy * sp, cy * sp, sy * cp],
        dtype=np.float32,
    )
    return position, orientation


def look_at_pose(
    eye: Sequence[float],
    target: Sequence[float],
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
        scale = math.sqrt(
            1.0
            + rotation[index, index]
            - rotation[j, j]
            - rotation[k, k]
        ) * 2.0
        quaternion = np.zeros(4, dtype=np.float32)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (
            rotation[k, j] - rotation[j, k]
        ) / scale
        quaternion[j + 1] = (
            rotation[j, index] + rotation[index, j]
        ) / scale
        quaternion[k + 1] = (
            rotation[k, index] + rotation[index, k]
        ) / scale
    return eye_array.astype(np.float32), quaternion


def chase_camera_pose(pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
    eye = (
        pose.x - 4.0 * math.cos(pose.yaw) - 1.8 * math.sin(pose.yaw),
        pose.y - 4.0 * math.sin(pose.yaw) + 1.8 * math.cos(pose.yaw),
        2.8,
    )
    target = (
        pose.x + math.cos(pose.yaw),
        pose.y + math.sin(pose.yaw),
        0.65,
    )
    return look_at_pose(eye, target)


def camera_rgb(camera) -> np.ndarray | None:
    rgba = camera.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    image = np.asarray(rgba)[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(
            image
            * (
                255.0
                if float(image.max()) <= 1.0
                else 1.0
            ),
            0,
            255,
        ).astype(np.uint8)
    return image


def compose_scene() -> dict:
    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage_utils.add_reference_to_stage(scene_source, SCENE_PRIM_PATH)
    app_utils.update_app(steps=120)
    stage = stage_utils.get_current_stage()
    scene = stage.GetPrimAtPath(SCENE_PRIM_PATH)
    if not scene.IsValid() or not scene.GetChildren():
        raise RuntimeError(f"warehouse scene did not compose: {scene_source}")

    prims = list(stage.Traverse())
    collision_count = sum(
        prim.HasAPI(UsdPhysics.CollisionAPI)
        for prim in prims
    )
    mesh_count = sum(prim.IsA(UsdGeom.Mesh) for prim in prims)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )
    scene_range = cache.ComputeWorldBound(scene).ComputeAlignedRange()
    if scene_range.IsEmpty():
        raise RuntimeError("warehouse scene bounds are empty")
    return {
        "prim_count": len(prims),
        "mesh_count": mesh_count,
        "collision_prim_count": collision_count,
        "bounds": {
            "minimum": [
                float(value)
                for value in scene_range.GetMin()
            ],
            "maximum": [
                float(value)
                for value in scene_range.GetMax()
            ],
        },
    }


def collect_collision_bounds() -> list[CollisionBounds]:
    stage = stage_utils.get_current_stage()
    scene = stage.GetPrimAtPath(SCENE_PRIM_PATH)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )
    result = []
    for prim in scene.GetChildren():
        if not prim.GetName().startswith(("Shelf_", "PalletBin_")):
            continue
        if not any(
            descendant.HasAPI(UsdPhysics.CollisionAPI)
            for descendant in Usd.PrimRange(prim)
        ):
            continue
        path = str(prim.GetPath())
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if value.IsEmpty():
            continue
        minimum = tuple(float(item) for item in value.GetMin())
        maximum = tuple(float(item) for item in value.GetMax())
        if not all(
            math.isfinite(item)
            for item in (*minimum, *maximum)
        ):
            continue
        if any(left > right for left, right in zip(minimum, maximum)):
            continue
        result.append(
            CollisionBounds(
                path,
                minimum,
                maximum,
            )
        )
    return result


def add_navigation_overlay(
    path: Sequence[tuple[float, float]],
    target: Pose2D | None,
) -> None:
    stage = stage_utils.get_current_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/VLN/DomeLight")
    dome.CreateIntensityAttr(250.0)
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
    route = UsdGeom.BasisCurves.Define(
        stage,
        "/World/VLN/PlannedPath",
    )
    route.CreateTypeAttr("linear")
    route.CreateCurveVertexCountsAttr([len(path)])
    route.CreatePointsAttr(
        [
            Gf.Vec3f(x, y, FLOOR_Z_M + 0.05)
            for x, y in path
        ]
    )
    route.CreateWidthsAttr([0.06])
    route.SetWidthsInterpolation("constant")
    route.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])


class SurveyRecorder:
    def __init__(
        self,
        output: Path,
        camera,
        intrinsics: np.ndarray,
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

    def capture(self, pose: Pose2D, *, force: bool = False) -> bool:
        if self.last_pose is not None and not force:
            distance = math.dist(
                (pose.x, pose.y),
                (self.last_pose.x, self.last_pose.y),
            )
            angle = abs(
                math.atan2(
                    math.sin(pose.yaw - self.last_pose.yaw),
                    math.cos(pose.yaw - self.last_pose.yaw),
                )
            )
            if (
                distance < args.survey_distance_step
                and angle < math.radians(args.survey_angle_step_deg)
            ):
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
                "robot_pose": {
                    "x": pose.x,
                    "y": pose.y,
                    "yaw": pose.yaw,
                },
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
            "scene": "MobileManiBench warehouse_multiple_shelves",
            "source_usd": scene_source,
            "rgb_is_only_model_input": True,
            "pose_consumer": (
                "offline_metric_alignment_and_evaluation_only"
            ),
            "camera": {
                "resolution": [camera_width, camera_height],
                "intrinsics": self.intrinsics,
                "horizontal_fov_deg": math.degrees(
                    2.0
                    * math.atan(
                        CAMERA_HORIZONTAL_APERTURE_MM
                        / (2.0 * CAMERA_FOCAL_LENGTH_MM)
                    )
                ),
                "height_above_floor_m": CAMERA_HEIGHT_M,
                "axes": "+X forward, +Z up",
            },
            "sampling": {
                "minimum_translation_m": args.survey_distance_step,
                "minimum_rotation_deg": args.survey_angle_step_deg,
            },
            "frames": self.frames,
        }
        target = self.root / "capture_manifest.json"
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


def write_plan_summary(
    *,
    map_source: str,
    task: str,
    start: Pose2D,
    target: Pose2D | None,
    path: Sequence[tuple[float, float]],
    scene_metrics: dict,
    source_collision_count: int,
    used_collision_count: int,
) -> Path:
    payload = {
        "schema_version": 1,
        "scene": "MobileManiBench/warehouse_multiple_shelves",
        "scene_source": scene_source,
        "robot": {
            "model": "G1-D",
            "usd": str(ROBOT_USD),
            "wheel_joints": [
                LEFT_WHEEL_JOINT,
                RIGHT_WHEEL_JOINT,
            ],
            "root_from_navigation_yaw_rad": (
                ROOT_FROM_NAVIGATION_YAW_RAD
            ),
            "usd_angular_sign": USD_ANGULAR_SIGN,
        },
        "map_source": map_source,
        "task": task,
        "start_pose": {
            "x": start.x,
            "y": start.y,
            "yaw": start.yaw,
        },
        "target_pose": (
            {
                "x": target.x,
                "y": target.y,
                "yaw": target.yaw,
            }
            if target is not None
            else None
        ),
        "path": [[x, y] for x, y in path],
        "path_length_m": path_length(path),
        "scene_metrics": scene_metrics,
        "source_collision_count": source_collision_count,
        "used_collision_count": used_collision_count,
    }
    target_path = args.output_dir / "navigation_plan.json"
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_metrics = compose_scene()
    print(
        "Warehouse scene: "
        f"prims={scene_metrics['prim_count']}, "
        f"meshes={scene_metrics['mesh_count']}, "
        f"collisions={scene_metrics['collision_prim_count']}"
    )

    source_collision_count = 0
    used_collision_count = 0
    if args.allow_bootstrap:
        collision_bounds = collect_collision_bounds()
        source_collision_count = len(collision_bounds)
        grid, places, start, used_bounds = build_bootstrap_artifacts(
            args.output_dir,
            collision_bounds,
        )
        used_collision_count = len(used_bounds)
        map_source = "isaac_collision_aabb_bootstrap"
    else:
        missing = [
            str(path)
            for path in (args.map, args.places)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Formal Warehouse navigation needs LingBot/SAM3 artifacts: "
                + ", ".join(missing)
                + ". Use --allow-bootstrap only for simulator integration."
            )
        grid, places = load_lingbot_artifacts(
            args.map,
            args.places,
        )
        start = WAREHOUSE_START
        if not grid.is_free(grid.world_to_cell(start.x, start.y)):
            raise ValueError(
                "formal LingBot map does not mark the reviewed start pose free"
            )
        map_source = "lingbot_map_rgb_only_pointcloud"

    target: Place | None
    if args.survey:
        target = None
        path = build_survey_path(grid, start, places)
        goal_yaw = start.yaw
        task = "warehouse_rgb_survey"
    else:
        target = resolve_place(args.command, places)
        path = grid.plan(
            (start.x, start.y),
            (target.pose.x, target.pose.y),
        )
        goal_yaw = target.pose.yaw
        task = target.place_id

    print(f"Map source: {map_source}")
    print(f"Task: {task}")
    if target is not None:
        print(
            f"Resolved {args.command!r} -> {target.place_id}"
        )
    print(
        f"Planned {len(path)} waypoints, "
        f"length={path_length(path):.3f} m"
    )
    add_navigation_overlay(
        path,
        None if target is None else target.pose,
    )
    plan_path = write_plan_summary(
        map_source=map_source,
        task=task,
        start=start,
        target=None if target is None else target.pose,
        path=path,
        scene_metrics=scene_metrics,
        source_collision_count=source_collision_count,
        used_collision_count=used_collision_count,
    )
    print(f"Plan: {plan_path}")
    if args.plan_only:
        return 0

    robot = WheeledRobot(
        paths=ROBOT_PRIM_PATH,
        wheel_dof_names=[
            LEFT_WHEEL_JOINT,
            RIGHT_WHEEL_JOINT,
        ],
        usd_path=str(ROBOT_USD),
        positions=[start.x, start.y, ROBOT_ROOT_Z_M],
    )

    camera = None
    if not args.no_camera:
        from isaacsim.sensors.camera import Camera

        camera = Camera(
            prim_path="/World/G1DRgbCamera",
            position=camera_world_pose(start)[0],
            orientation=camera_world_pose(start)[1],
            frequency=8 if args.survey else 2,
            resolution=(camera_width, camera_height),
        )

    chase_camera = None
    gif_frames = []
    if args.record_gif is not None:
        from isaacsim.sensors.camera import Camera

        chase_camera = Camera(
            prim_path="/World/VLNChaseCamera",
            position=chase_camera_pose(start)[0],
            orientation=chase_camera_pose(start)[1],
            frequency=args.record_fps,
            resolution=(480, 270),
        )

    if not args.headless:
        eye, orientation = chase_camera_pose(start)
        del orientation
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=eye,
            target=[
                start.x + 1.0,
                start.y,
                0.65,
            ],
        )

    SimulationManager.setup_simulation(
        dt=1.0 / PHYSICS_HZ,
        device="cpu",
    )
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    app_utils.play()
    app_utils.update_app(steps=18)
    configure_joint_drives(robot)

    pose = start
    set_assisted_pose(robot, pose, 0.0, 0.0)
    app_utils.update_app(steps=5)

    recorder = None
    if camera is not None:
        camera.initialize()
        camera.set_focal_length(CAMERA_FOCAL_LENGTH_MM)
        camera.set_horizontal_aperture(
            CAMERA_HORIZONTAL_APERTURE_MM
        )
        camera.set_world_pose(
            *camera_world_pose(pose),
            camera_axes="world",
        )
        app_utils.update_app(steps=18)
        if args.survey:
            if (
                args.survey_distance_step <= 0.0
                or args.survey_angle_step_deg <= 0.0
            ):
                raise ValueError("survey sampling steps must be positive")
            recorder = SurveyRecorder(
                args.output_dir,
                camera,
                camera.get_intrinsics_matrix(),
            )
            recorder.capture(pose, force=True)
    if chase_camera is not None:
        chase_camera.initialize()
        chase_camera.set_focal_length(16.0)
        chase_camera.set_horizontal_aperture(28.0)
        chase_camera.set_world_pose(
            *chase_camera_pose(pose),
            camera_axes="world",
        )
        app_utils.update_app(steps=18)

    follower = PathFollower(
        path,
        goal_yaw=goal_yaw,
        max_linear=0.75,
        max_angular=1.20,
        position_tolerance=args.position_tolerance,
        yaw_tolerance=args.yaw_tolerance,
    )
    frame = 0
    last_label = "start"
    while simulation_app.is_running():
        observed = (
            robot_pose(robot)
            if args.wheel_physics_only
            else pose
        )
        linear, angular, label = follower.command(observed)
        robot.apply_wheel_actions(
            command_to_wheels(linear, angular)
        )
        if not args.wheel_physics_only:
            pose = assisted_step(pose, linear, angular)
            set_assisted_pose(robot, pose, linear, angular)
        current = (
            robot_pose(robot)
            if args.wheel_physics_only
            else pose
        )
        if camera is not None:
            camera.set_world_pose(
                *camera_world_pose(current),
                camera_axes="world",
            )
        if chase_camera is not None:
            chase_camera.set_world_pose(
                *chase_camera_pose(current),
                camera_axes="world",
            )
        simulation_app.update()
        if recorder is not None:
            recorder.capture(current)
        if (
            chase_camera is not None
            and frame
            % max(1, PHYSICS_HZ // args.record_fps)
            == 0
        ):
            image = camera_rgb(chase_camera)
            if image is not None:
                from PIL import Image

                gif_frames.append(Image.fromarray(image).copy())
        if label != last_label or frame % 240 == 0:
            print(
                f"frame={frame:4d} state={label:7s} "
                f"waypoint={follower.index}/{len(path)-1} "
                f"pose=({current.x:+.2f},{current.y:+.2f},"
                f"{current.yaw:+.2f}) "
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
    final_pose = (
        robot_pose(robot)
        if args.wheel_physics_only
        else pose
    )
    position_error = math.dist(
        (final_pose.x, final_pose.y),
        path[-1],
    )
    yaw_error = abs(
        math.atan2(
            math.sin(goal_yaw - final_pose.yaw),
            math.cos(goal_yaw - final_pose.yaw),
        )
    )
    manifest = None
    if recorder is not None:
        recorder.capture(final_pose, force=True)
        manifest = recorder.finish()
        print(
            f"Survey frames: {len(recorder.frames)} -> {manifest}"
        )
    elif camera is not None:
        image = camera_rgb(camera)
        if image is not None:
            from PIL import Image

            Image.fromarray(image).save(
                args.output_dir / "arrival_rgb.png"
            )

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
        print(
            f"Isaac movement GIF: {args.record_gif} "
            f"({len(gif_frames)} frames)"
        )

    result = {
        "schema_version": 1,
        "task": task,
        "command": None if args.survey else args.command,
        "scene": "MobileManiBench/warehouse_multiple_shelves",
        "scene_source": scene_source,
        "scene_metrics": scene_metrics,
        "robot": {
            "model": "G1-D",
            "usd": str(ROBOT_USD),
            "wheel_joints": [
                LEFT_WHEEL_JOINT,
                RIGHT_WHEEL_JOINT,
            ],
            "wheel_radius_m": WHEEL_RADIUS_M,
            "wheel_base_m": WHEEL_BASE_M,
            "root_from_navigation_yaw_rad": (
                ROOT_FROM_NAVIGATION_YAW_RAD
            ),
            "usd_angular_sign": USD_ANGULAR_SIGN,
        },
        "map_source": map_source,
        "execution_mode": (
            "wheel_physics_only"
            if args.wheel_physics_only
            else "stable_assisted"
        ),
        "camera_enabled": camera is not None,
        "success": follower.done,
        "frames": frame,
        "path_length_m": path_length(path),
        "final_pose": {
            "x": final_pose.x,
            "y": final_pose.y,
            "yaw": final_pose.yaw,
        },
        "target_pose": {
            "x": path[-1][0],
            "y": path[-1][1],
            "yaw": goal_yaw,
        },
        "position_error_m": position_error,
        "yaw_error_rad": yaw_error,
        "source_collision_count": source_collision_count,
        "used_collision_count": used_collision_count,
        "survey_manifest": str(manifest) if manifest else None,
        "isaac_movement_gif": (
            str(args.record_gif)
            if args.record_gif
            else None
        ),
    }
    summary_name = (
        "survey_summary.json"
        if args.survey
        else "run_summary.json"
    )
    summary_path = args.output_dir / summary_name
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Result: success={follower.done} "
        f"position_error={position_error:.3f} m "
        f"yaw_error={yaw_error:.3f} rad"
    )
    print(f"Summary: {summary_path}")

    app_utils.stop()
    if args.test:
        complex_scene_ok = (
            scene_metrics["prim_count"] >= 8000
            and scene_metrics["collision_prim_count"] >= 1800
        )
        if (
            not follower.done
            or position_error > 0.20
            or not complex_scene_ok
        ):
            print(
                "TEST FAILED: G1-D warehouse navigation "
                "or scene-complexity assertion failed",
                file=sys.stderr,
            )
            return 2
        print(
            "TEST PASSED: multi-shelf Warehouse and G1-D "
            "navigation are connected"
        )
    return 0


try:
    exit_code = main()
except Exception:
    # SimulationApp.close() may terminate native Kit before Python emits an
    # unhandled exception, so print the actionable error first.
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()

raise SystemExit(exit_code)
