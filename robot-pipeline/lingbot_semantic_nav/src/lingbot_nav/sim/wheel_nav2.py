"""Nav2-driven differential-wheel simulation on a LingBot occupancy map.

The simulator integrates ``/cmd_vel`` in the LingBot map frame and publishes
odometry/TF.  It deliberately has no Habitat imports and never queries a
navmesh, simulator scene, recorded pose, depth image, or semantic ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable

from ..errors import ConfigurationError
from ..mission import MissionResolver
from ..models import Mission
from .frame_writer import RealtimePacer
from .occupancy_planner import OccupancyPathPlanner, OccupancyPlannerConfig


@dataclass(frozen=True)
class WheelNav2Config:
    map_yaml: Path
    output_dir: Path
    instruction: str
    start_x: float
    start_y: float
    start_yaw: float = 0.0
    robot_radius: float = 0.08
    control_hz: float = 20.0
    realtime_factor: float = 1.0
    max_navigation_time: float = 180.0
    action_server_timeout: float = 45.0

    def validate(self) -> None:
        values = (
            self.start_x,
            self.start_y,
            self.start_yaw,
            self.robot_radius,
            self.control_hz,
            self.realtime_factor,
            self.max_navigation_time,
            self.action_server_timeout,
        )
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("Wheel simulation config contains non-finite values")
        if min(
            self.robot_radius,
            self.control_hz,
            self.realtime_factor,
            self.max_navigation_time,
            self.action_server_timeout,
        ) <= 0.0:
            raise ConfigurationError("Wheel simulation rates, radius and timeouts must be positive")
        if not self.map_yaml.expanduser().resolve().is_file():
            raise ConfigurationError(f"LingBot occupancy map not found: {self.map_yaml}")


def integrate_differential_drive(
    x: float,
    y: float,
    yaw: float,
    linear_velocity: float,
    angular_velocity: float,
    dt: float,
) -> tuple[float, float, float]:
    """Integrate one planar unicycle/differential-drive command."""
    if dt < 0.0 or not all(
        math.isfinite(value)
        for value in (x, y, yaw, linear_velocity, angular_velocity, dt)
    ):
        raise ConfigurationError("Invalid differential-drive integration input")
    if abs(angular_velocity) < 1e-9:
        return (
            x + linear_velocity * math.cos(yaw) * dt,
            y + linear_velocity * math.sin(yaw) * dt,
            math.atan2(math.sin(yaw), math.cos(yaw)),
        )
    next_yaw = yaw + angular_velocity * dt
    radius = linear_velocity / angular_velocity
    next_x = x + radius * (math.sin(next_yaw) - math.sin(yaw))
    next_y = y - radius * (math.cos(next_yaw) - math.cos(yaw))
    return next_x, next_y, math.atan2(math.sin(next_yaw), math.cos(next_yaw))


def _ros_imports():
    try:
        import rclpy
        from action_msgs.msg import GoalStatus
        from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
        from nav2_msgs.action import NavigateToPose
        from nav_msgs.msg import Odometry, Path as RosPath
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from std_msgs.msg import Bool
        from tf2_ros import TransformBroadcaster
    except ImportError as exc:
        raise ConfigurationError("Wheel Nav2 simulation requires ROS 2 and Nav2 Python packages") from exc
    return (
        rclpy,
        GoalStatus,
        PoseStamped,
        TransformStamped,
        Twist,
        NavigateToPose,
        Odometry,
        RosPath,
        ActionClient,
        Node,
        TransformBroadcaster,
        Bool,
    )


class WheelNav2Runtime:
    """Long-lived 8083 odom/TF bridge shared by consecutive missions."""

    def __init__(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        *,
        publish_hz: float = 20.0,
    ) -> None:
        if publish_hz <= 0.0 or not all(
            math.isfinite(value)
            for value in (start_x, start_y, start_yaw, publish_hz)
        ):
            raise ConfigurationError("Invalid persistent wheel bridge pose or rate")
        (
            rclpy,
            GoalStatus,
            PoseStamped,
            TransformStamped,
            Twist,
            NavigateToPose,
            Odometry,
            RosPath,
            ActionClient,
            Node,
            TransformBroadcaster,
            Bool,
        ) = _ros_imports()
        if not rclpy.ok():
            rclpy.init(args=None)

        class WheelBridge(Node):
            def __init__(self) -> None:
                super().__init__("lingbot_8083_wheel_nav2_bridge")
                self.command = (0.0, 0.0)
                self.command_lock = threading.Lock()
                self.pose_lock = threading.Lock()
                self.retained_pose = (start_x, start_y, start_yaw, 0.0, 0.0)
                self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
                self.active_pub = self.create_publisher(
                    Bool, "/lingbot_wheel_bridge_active", 10
                )
                self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
                self.create_subscription(RosPath, "/plan", self._on_plan, 10)
                self.tf = TransformBroadcaster(self)
                self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
                self.latest_plan: list[list[float]] = []
                self.create_timer(1.0 / publish_hz, self._publish_retained_pose)
                self.publish_pose(start_x, start_y, start_yaw, 0.0, 0.0)

            def announce_active(self, value: bool) -> None:
                message = Bool()
                message.data = value
                self.active_pub.publish(message)

            def _on_cmd(self, message) -> None:
                with self.command_lock:
                    self.command = (
                        float(message.linear.x),
                        float(message.angular.z),
                    )

            def velocity(self) -> tuple[float, float]:
                with self.command_lock:
                    return self.command

            def stop(self) -> None:
                with self.command_lock:
                    self.command = (0.0, 0.0)
                with self.pose_lock:
                    x, y, yaw, _, _ = self.retained_pose
                    self.retained_pose = (x, y, yaw, 0.0, 0.0)

            def current_pose(self) -> tuple[float, float, float]:
                with self.pose_lock:
                    return self.retained_pose[:3]

            def _on_plan(self, message) -> None:
                self.latest_plan = [
                    [float(item.pose.position.x), 0.0, float(item.pose.position.y)]
                    for item in message.poses
                ]

            def _emit_pose(
                self, x: float, y: float, yaw: float, v: float, omega: float
            ) -> None:
                # Repeated active announcements let the startup publisher hand
                # off even when ROS discovery occurs after this node starts.
                self.announce_active(True)
                stamp = self.get_clock().now().to_msg()
                qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
                odom = Odometry()
                odom.header.stamp = stamp
                odom.header.frame_id = "odom"
                odom.child_frame_id = "base_link"
                odom.pose.pose.position.x = x
                odom.pose.pose.position.y = y
                odom.pose.pose.orientation.z = qz
                odom.pose.pose.orientation.w = qw
                odom.twist.twist.linear.x = v
                odom.twist.twist.angular.z = omega
                self.odom_pub.publish(odom)
                transform = TransformStamped()
                transform.header.stamp = stamp
                transform.header.frame_id = "odom"
                transform.child_frame_id = "base_link"
                transform.transform.translation.x = x
                transform.transform.translation.y = y
                transform.transform.rotation.z = qz
                transform.transform.rotation.w = qw
                self.tf.sendTransform(transform)

            def _publish_retained_pose(self) -> None:
                with self.pose_lock:
                    pose = self.retained_pose
                self._emit_pose(*pose)

            def publish_pose(
                self, x: float, y: float, yaw: float, v: float, omega: float
            ) -> None:
                with self.pose_lock:
                    self.retained_pose = (x, y, yaw, v, omega)
                self._emit_pose(x, y, yaw, v, omega)

            def goal(self, x: float, y: float, yaw: float, feedback_callback):
                stamped = PoseStamped()
                stamped.header.frame_id = "map"
                stamped.header.stamp = self.get_clock().now().to_msg()
                stamped.pose.position.x = x
                stamped.pose.position.y = y
                stamped.pose.orientation.z = math.sin(yaw / 2.0)
                stamped.pose.orientation.w = math.cos(yaw / 2.0)
                goal = NavigateToPose.Goal()
                goal.pose = stamped
                return self.nav.send_goal_async(
                    goal, feedback_callback=feedback_callback
                )

        self._rclpy = rclpy
        self.goal_status = GoalStatus
        self.node = WheelBridge()
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="lingbot-8083-wheel-bridge",
            daemon=True,
        )
        self._closed = False
        self._spin_thread.start()

    def pose(self) -> tuple[float, float, float]:
        return self.node.current_pose()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.node.stop()
        self.node._publish_retained_pose()
        self.node.announce_active(False)
        time.sleep(0.1)
        self._executor.shutdown()
        self.node.destroy_node()
        self._spin_thread.join(timeout=2.0)


def run_wheel_nav2_route(
    config: WheelNav2Config,
    resolver: MissionResolver,
    mission: Mission | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    runtime: WheelNav2Runtime | None = None,
) -> dict[str, object]:
    """Execute one semantic mission with Nav2 and LingBot-only wheel kinematics."""
    config.validate()
    mission = mission or resolver.resolve(config.instruction)
    if not mission.steps:
        raise ConfigurationError("Resolved wheel mission has no navigation steps")
    occupancy = OccupancyPathPlanner(
        config.map_yaml,
        OccupancyPlannerConfig(
            robot_radius=config.robot_radius,
            max_snap_distance=0.75,
            unknown_is_occupied=True,
        ),
    )
    dt = 1.0 / config.control_hz
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    owned_runtime = runtime is None
    runtime = runtime or WheelNav2Runtime(
        config.start_x, config.start_y, config.start_yaw, publish_hz=config.control_hz
    )
    node = runtime.node
    retained_pose = runtime.pose()
    start_error = math.hypot(
        retained_pose[0] - config.start_x, retained_pose[1] - config.start_y
    )
    yaw_error = abs(
        math.atan2(
            math.sin(retained_pose[2] - config.start_yaw),
            math.cos(retained_pose[2] - config.start_yaw),
        )
    )
    if not owned_runtime and (start_error > 1e-4 or yaw_error > 1e-4):
        raise ConfigurationError(
            "Dashboard pose diverged from the persistent wheel odometry"
        )
    node.latest_plan = []
    pose = list(retained_pose)
    trace: list[dict[str, object]] = []
    collisions = 0
    active_handle = None
    started = time.monotonic()

    def traversable(x: float, y: float) -> bool:
        row = math.floor((y - occupancy.grid.origin_y) / occupancy.grid.resolution)
        col = math.floor((x - occupancy.grid.origin_x) / occupancy.grid.resolution)
        return bool(
            0 <= row < occupancy.free.shape[0]
            and 0 <= col < occupancy.free.shape[1]
            and occupancy.free[row, col]
        )

    def feedback(_message) -> None:
        return

    def publish_sample(action: str, v: float, omega: float, collided: bool, step_index: int) -> None:
        sample = {
            "frame": len(trace),
            "action": action,
            "linear_velocity_mps": v,
            "angular_velocity_rps": omega,
            "map_position": [pose[0], pose[1]],
            "map_yaw": pose[2],
            "collided": collided,
            "step_index": step_index,
            "step_count": len(mission.steps),
        }
        trace.append(sample)
        node.publish_pose(pose[0], pose[1], pose[2], v, omega)
        if progress_callback:
            progress_callback({"kind": "frame", "sample": sample})

    try:
        publish_sample("nav2_wait", 0.0, 0.0, False, 0)
        deadline = time.monotonic() + config.action_server_timeout
        while not node.nav.server_is_ready():
            node.publish_pose(*pose, 0.0, 0.0)
            if time.monotonic() >= deadline:
                raise ConfigurationError("Nav2 NavigateToPose server did not become ready")
            time.sleep(0.1)

        for step_index, step in enumerate(mission.steps, start=1):
            target = step.place.entrance_pose
            if progress_callback:
                progress_callback({
                    "kind": "goal",
                    "destination": step.place.place_id,
                    "destination_name": step.place.name,
                    "route": [
                        {
                            "action": item.action.value,
                            "id": item.place.place_id,
                            "name": item.place.name,
                        }
                        for item in mission.steps
                    ],
                    "target": step.place.to_dict(),
                    "step_index": step_index,
                    "step_count": len(mission.steps),
                })
            send_future = node.goal(target.x, target.y, target.yaw, feedback)
            while not send_future.done():
                node.publish_pose(*pose, 0.0, 0.0)
                time.sleep(0.05)
            active_handle = send_future.result()
            if not active_handle.accepted:
                raise ConfigurationError("Nav2 rejected NavigateToPose goal")
            result_future = active_handle.get_result_async()
            published_plan: list[list[float]] = []
            pacer = RealtimePacer(dt / config.realtime_factor)
            while not result_future.done():
                if cancel_check and cancel_check():
                    active_handle.cancel_goal_async()
                if time.monotonic() - started > config.max_navigation_time:
                    active_handle.cancel_goal_async()
                    raise ConfigurationError("Nav2 wheel navigation timeout")
                v, omega = node.velocity()
                proposed = integrate_differential_drive(*pose, v, omega, dt)
                collided = not traversable(proposed[0], proposed[1])
                if collided:
                    collisions += 1
                    v = 0.0
                else:
                    pose[:] = proposed
                publish_sample("nav2_cmd_vel", v, omega, collided, step_index)
                if node.latest_plan and node.latest_plan != published_plan:
                    published_plan = list(node.latest_plan)
                    if progress_callback:
                        progress_callback({
                            "kind": "planned",
                            "points": published_plan,
                            "planner": "nav2",
                            "controller": "nav2_regulated_pure_pursuit",
                        })
                pacer.wait()
            status = result_future.result().status
            if (
                status == runtime.goal_status.STATUS_CANCELED
                and cancel_check
                and cancel_check()
            ):
                raise ConfigurationError("Nav2 wheel navigation cancelled")
            if status != runtime.goal_status.STATUS_SUCCEEDED:
                raise ConfigurationError(f"Nav2 NavigateToPose status={status}")
        node.stop()
        publish_sample("nav2_arrived", 0.0, 0.0, False, len(mission.steps))
    finally:
        if active_handle is not None and cancel_check and cancel_check():
            active_handle.cancel_goal_async()
        node.stop()
        node.publish_pose(pose[0], pose[1], pose[2], 0.0, 0.0)
        if owned_runtime:
            runtime.close()

    manifest = {
        "schema_version": 1,
        "runtime": "nav2+differential_wheel_kinematics",
        "instruction": config.instruction,
        "mission": mission.to_dict(),
        "frames": len(trace),
        "collisions": collisions,
        "planning_inputs": {
            "planner": "nav2",
            "map": str(config.map_yaml.expanduser().resolve()),
            "map_source": "lingbot_map_rgb_only_occupancy",
            "habitat_navmesh": False,
            "habitat_depth": False,
            "habitat_semantics": False,
            "habitat_camera_poses": False,
            "simulator_ground_truth_map": False,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    (output / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "WheelNav2Config",
    "WheelNav2Runtime",
    "integrate_differential_drive",
    "run_wheel_nav2_route",
]
