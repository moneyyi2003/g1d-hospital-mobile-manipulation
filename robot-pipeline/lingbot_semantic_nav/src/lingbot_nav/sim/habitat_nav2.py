"""Habitat robot adapter driven by Nav2 NavigateToPose and /cmd_vel."""

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
from .habitat_collector import _camera_spec, _imports
from .frame_writer import AsyncPngWriter, RealtimePacer
from .habitat_route import executable_navigation_steps


@dataclass(frozen=True)
class HabitatNav2Config:
    scene: str | Path
    output_dir: Path
    instruction: str
    simulation_start: tuple[float, float, float]
    scene_dataset_config: Path | None = None
    map_unit_to_sim_meter: float = 1.0
    width: int = 640
    height: int = 480
    sensor_height: float = 1.0
    hfov_degrees: float = 90.0
    # Match Nav2's default controller_frequency so each cmd_vel update is
    # integrated and rendered instead of sampling only every other command.
    control_hz: float = 20.0
    action_server_timeout: float = 30.0
    max_navigation_time: float = 180.0
    realtime_factor: float = 1.0
    seed: int = 7


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _habitat_yaw(rotation) -> float:
    return math.atan2(
        2.0 * (rotation.w * rotation.y + rotation.x * rotation.z),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


def _ros_imports():
    try:
        import rclpy
        from action_msgs.msg import GoalStatus
        from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
        from nav2_msgs.action import NavigateToPose
        from nav_msgs.msg import Odometry, Path as RosPath
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from tf2_ros import TransformBroadcaster
        from std_msgs.msg import Bool
    except ImportError as exc:
        raise ConfigurationError(
            "Nav2 backend needs the ROS 2 Humble environment"
        ) from exc
    return (
        rclpy, GoalStatus, PoseStamped, TransformStamped, Twist,
        NavigateToPose, Odometry, RosPath, ActionClient, Node,
        TransformBroadcaster, Bool,
    )


def run_habitat_nav2_route(
    config: HabitatNav2Config,
    resolver: MissionResolver,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, object]:
    habitat_sim, np, quaternion, Image = _imports()
    (
        rclpy, GoalStatus, PoseStamped, TransformStamped, Twist,
        NavigateToPose, Odometry, RosPath, ActionClient, Node,
        TransformBroadcaster, Bool,
    ) = _ros_imports()
    import magnum as mn
    from ament_index_python.packages import get_package_share_directory

    scene_path = Path(config.scene).expanduser()
    dataset_path = config.scene_dataset_config
    scene_is_file = scene_path.is_file()
    if (
        config.map_unit_to_sim_meter <= 0
        or (not scene_is_file and dataset_path is None)
        or (dataset_path is not None and not dataset_path.is_file())
    ):
        raise ConfigurationError("Invalid Habitat scene or map scale for Nav2")
    mission = resolver.resolve(config.instruction)
    if not mission.steps:
        raise ConfigurationError("Resolved Nav2 mission has no steps")
    navigation_steps = executable_navigation_steps(mission.steps)
    start_place = resolver.places.resolve(resolver.topology_start).place
    map_start = (start_place.entrance_pose.x, start_place.entrance_pose.y)
    start_yaw = start_place.entrance_pose.yaw
    dt = 1.0 / config.control_hz
    behavior_tree = str(
        Path(get_package_share_directory("nav2_bt_navigator"))
        / "behavior_trees"
        / "navigate_w_replanning_only_if_path_becomes_invalid.xml"
    )
    output = config.output_dir
    rgb_dir = output / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    if not rclpy.ok():
        rclpy.init(args=None)

    class HabitatNav2Bridge(Node):
        def __init__(self) -> None:
            super().__init__("habitat_dashboard_nav2_bridge")
            self.command = (0.0, 0.0)
            self.command_lock = threading.Lock()
            self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
            self.active_pub = self.create_publisher(Bool, "/habitat_bridge_active", 10)
            self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
            self.create_subscription(RosPath, "/plan", self._on_plan, 10)
            self.tf = TransformBroadcaster(self)
            self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
            self.latest_plan: list[list[float]] = []
            active = Bool()
            active.data = True
            self.active_pub.publish(active)

        def _on_cmd(self, message) -> None:
            with self.command_lock:
                self.command = (float(message.linear.x), float(message.angular.z))

        def velocity(self) -> tuple[float, float]:
            with self.command_lock:
                return self.command

        def stop(self) -> None:
            with self.command_lock:
                self.command = (0.0, 0.0)

        def announce_active(self, value: bool) -> None:
            active = Bool()
            active.data = value
            self.active_pub.publish(active)

        def _on_plan(self, message) -> None:
            self.latest_plan = [
                [float(item.pose.position.x), 0.0, float(item.pose.position.y)]
                for item in message.poses
            ]

        def publish_pose(self, x: float, y: float, yaw: float, v: float, omega: float) -> None:
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
            # The stock recovery tree repeatedly spins and backs up when DWB
            # reports a blocked doorway.  This static-map bridge instead uses
            # path-validity replanning and fails promptly if no safe command
            # exists, avoiding long left/right recovery oscillations.
            goal.behavior_tree = behavior_tree
            return self.nav.send_goal_async(goal, feedback_callback=feedback_callback)

    node = HabitatNav2Bridge()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(config.scene)
    if dataset_path is not None:
        sim_cfg.scene_dataset_config_file = str(dataset_path.resolve())
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _camera_spec(habitat_sim, "color_sensor", habitat_sim.SensorType.COLOR, config)
    ]
    trace: list[dict[str, object]] = []
    collisions = 0
    planned_path: list[list[float]] = []
    active_handle = None
    started = time.monotonic()

    def feedback(_message) -> None:
        return

    try:
        with AsyncPngWriter(Image) as image_writer, habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        ) as sim:
            sim.seed(config.seed)
            agent = sim.initialize_agent(0)
            state = agent.get_state()
            state.position = np.asarray(config.simulation_start, dtype=np.float32)
            # Habitat yaw zero looks along map -Y, while ROS yaw +pi/2 is +Y.
            habitat_start_yaw = -start_yaw - math.pi / 2.0
            state.rotation = quaternion.from_rotation_vector(
                np.asarray([0.0, habitat_start_yaw, 0.0], dtype=np.float64)
            )
            agent.set_state(state)
            velocity = habitat_sim.physics.VelocityControl()
            velocity.controlling_lin_vel = True
            velocity.lin_vel_is_local = True
            velocity.controlling_ang_vel = True
            velocity.ang_vel_is_local = True

            def map_pose():
                current = agent.get_state()
                x = map_start[0] + (
                    float(current.position[0]) - config.simulation_start[0]
                ) / config.map_unit_to_sim_meter
                y = map_start[1] + (
                    float(current.position[2]) - config.simulation_start[2]
                ) / config.map_unit_to_sim_meter
                yaw = _wrap(-_habitat_yaw(current.rotation) - math.pi / 2.0)
                return x, y, yaw

            def publish_frame(action: str, v: float, omega: float, collided: bool, step_index: int):
                nonlocal collisions
                observations = sim.get_sensor_observations()
                frame = len(trace)
                rgb_path = rgb_dir / f"{frame:06d}.png"
                color = np.asarray(observations["color_sensor"])[..., :3].astype(np.uint8)
                current = agent.get_state()
                x, y, yaw = map_pose()
                rotation = current.rotation
                sample = {
                    "frame": frame,
                    "action": action,
                    "linear_velocity_mps": v,
                    "angular_velocity_rps": omega,
                    "position": [float(item) for item in current.position],
                    "map_position": [x, y],
                    "rotation_xyzw": [
                        float(rotation.x), float(rotation.y),
                        float(rotation.z), float(rotation.w),
                    ],
                    "map_yaw": yaw,
                    "collided": collided,
                    "step_index": step_index,
                    "step_count": len(navigation_steps),
                }
                collisions += int(collided)
                trace.append(sample)
                node.publish_pose(x, y, yaw, v, omega)
                image_writer.submit(
                    color,
                    rgb_path,
                    (
                        lambda saved_path, jpeg, current_sample=sample: progress_callback(
                            {
                                "kind": "frame",
                                "sample": current_sample,
                                "rgb_path": str(saved_path),
                                "jpeg_bytes": jpeg,
                            }
                        )
                    )
                    if progress_callback
                    else None,
                )

            publish_frame("nav2_wait", 0.0, 0.0, False, 0)
            wait_deadline = time.monotonic() + config.action_server_timeout
            while not node.nav.server_is_ready():
                x, y, yaw = map_pose()
                node.publish_pose(x, y, yaw, 0.0, 0.0)
                if time.monotonic() >= wait_deadline:
                    raise ConfigurationError("Nav2 NavigateToPose server did not become ready")
                time.sleep(0.1)

            # RGB scan poses are evidence-collection camera locations, not
            # mandatory motion goals.  Replaying all of them introduced tight
            # doorway segments and forced a full stop at every scan.  Keep the
            # semantic region chain and explicit user waypoints, then finish at
            # the requested place/object.
            for step_index, step in enumerate(navigation_steps, start=1):
                target = step.place.entrance_pose
                if step_index < len(navigation_steps):
                    following = navigation_steps[step_index].place.entrance_pose
                    goal_yaw = math.atan2(
                        following.y - target.y,
                        following.x - target.x,
                    )
                else:
                    goal_yaw = target.yaw
                if progress_callback:
                    progress_callback({
                        "kind": "planned",
                        "instruction": config.instruction,
                        "destination": mission.place.place_id,
                        "destination_name": mission.place.name,
                        "route": [
                            {"action": item.action.value, "id": item.place.place_id, "name": item.place.name}
                            for item in mission.steps
                        ],
                        "points": node.latest_plan or [[map_pose()[0], 0.0, map_pose()[1]], [target.x, 0.0, target.y]],
                        "controller": "nav2_navigate_to_pose",
                        "planner": "nav2",
                    })
                # RGB scan waypoints store the camera heading used during map
                # collection.  Replaying those headings made the robot turn
                # back and forth at every intermediate stop.  Face the next
                # waypoint instead; only the final goal keeps its semantic yaw.
                send_future = node.goal(target.x, target.y, goal_yaw, feedback)
                while not send_future.done():
                    x, y, yaw = map_pose()
                    node.publish_pose(x, y, yaw, 0.0, 0.0)
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
                        raise ConfigurationError("Nav2 navigation timeout")
                    v, omega = node.velocity()
                    old = agent.get_state()
                    rigid = habitat_sim.RigidState(
                        mn.Quaternion(
                            mn.Vector3(old.rotation.x, old.rotation.y, old.rotation.z),
                            old.rotation.w,
                        ),
                        mn.Vector3(*[float(item) for item in old.position]),
                    )
                    # Nav2 owns both path planning and collision avoidance.  The
                    # Habitat navmesh/step_filter must not alter a route planned
                    # on the LingBot occupancy map; Habitat is only the RGB
                    # renderer and cmd_vel motion adapter in this backend.
                    velocity.linear_velocity = mn.Vector3(
                        0.0, 0.0, -v * config.map_unit_to_sim_meter
                    )
                    velocity.angular_velocity = mn.Vector3(0.0, -omega, 0.0)
                    integrated = velocity.integrate_transform(dt, rigid)
                    next_state = agent.get_state()
                    next_state.position = np.asarray(
                        integrated.translation, dtype=np.float32
                    )
                    next_state.rotation = quaternion.quaternion(
                        integrated.rotation.scalar,
                        integrated.rotation.vector.x,
                        integrated.rotation.vector.y,
                        integrated.rotation.vector.z,
                    )
                    agent.set_state(next_state)
                    publish_frame("nav2_cmd_vel", v, omega, False, step_index)
                    if node.latest_plan and node.latest_plan != published_plan:
                        planned_path = list(node.latest_plan)
                        published_plan = list(node.latest_plan)
                        if progress_callback:
                            progress_callback({
                                "kind": "planned",
                                "instruction": config.instruction,
                                "destination": mission.place.place_id,
                                "destination_name": mission.place.name,
                                "route": [
                                    {"action": item.action.value, "id": item.place.place_id, "name": item.place.name}
                                    for item in mission.steps
                                ],
                                "points": planned_path,
                                "controller": "nav2_navigate_to_pose",
                                "planner": "nav2_navfn",
                            })
                    pacer.wait()
                status = result_future.result().status
                if status != GoalStatus.STATUS_SUCCEEDED:
                    raise ConfigurationError(f"Nav2 NavigateToPose status={status}")
            node.stop()
            publish_frame("nav2_arrived", 0.0, 0.0, False, len(navigation_steps))
    finally:
        if active_handle is not None and cancel_check and cancel_check():
            active_handle.cancel_goal_async()
        node.announce_active(False)
        time.sleep(0.1)
        executor.shutdown()
        node.destroy_node()
        spin_thread.join(timeout=2.0)

    manifest = {
        "schema_version": 1,
        "runtime": "nav2_navigate_to_pose+habitat_cmd_vel_adapter",
        "scene": str(config.scene),
        "instruction": config.instruction,
        "mission": mission.to_dict(),
        "frames": len(trace),
        "collisions": collisions,
        "planned_path": planned_path,
        "planning_inputs": {
            "planner": "nav2",
            "map": "lingbot_map_occupancy",
            "semantic_goal_frame": "lingbot_map_local",
            "habitat_navmesh_queries": 0,
            "habitat_ground_truth_used_for_planning": False,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


__all__ = ["HabitatNav2Config", "run_habitat_nav2_route"]
