"""Continuous unicycle control for reviewed Habitat semantic routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

from ..errors import ConfigurationError
from ..mission import MissionResolver
from .habitat_collector import _camera_pose_matrix, _camera_spec, _imports
from .habitat_route import _habitat_position, executable_navigation_steps
from .frame_writer import AsyncPngWriter, RealtimePacer
from .occupancy_planner import OccupancyPathPlanner, OccupancyPlannerConfig


@dataclass(frozen=True)
class HabitatContinuousConfig:
    scene: Path
    output_dir: Path
    instruction: str
    width: int = 640
    height: int = 480
    sensor_height: float = 1.0
    hfov_degrees: float = 90.0
    control_hz: float = 20.0
    linear_speed_mps: float = 0.40
    angular_speed_rps: float = 1.20
    heading_gain: float = 2.2
    turn_in_place_threshold_rad: float = 0.35
    waypoint_radius: float = 0.16
    goal_radius: float = 0.20
    max_control_steps: int = 1500
    realtime_factor: float = 1.0
    seed: int = 7
    map_yaml: Path | None = None
    robot_radius: float = 0.14
    simulation_start: tuple[float, float, float] | None = None
    map_unit_to_sim_meter: float = 1.0
    unknown_is_occupied: bool = True
    use_habitat_navmesh: bool = False


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw(rotation) -> float:
    return math.atan2(
        2.0 * (rotation.w * rotation.y + rotation.x * rotation.z),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


def run_habitat_continuous_route(
    config: HabitatContinuousConfig,
    resolver: MissionResolver,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, object]:
    habitat_sim, np, quaternion, Image = _imports()
    import magnum as mn

    if config.control_hz <= 0 or config.linear_speed_mps <= 0 or config.angular_speed_rps <= 0:
        raise ConfigurationError("Continuous Habitat controller needs positive rates and speeds")
    if not config.scene.is_file():
        raise ConfigurationError(f"Habitat scene not found: {config.scene}")
    mission = resolver.resolve(config.instruction)
    navigation_steps = executable_navigation_steps(mission.steps)
    if config.map_yaml is None:
        raise ConfigurationError("Generated occupancy map is required for Habitat navigation")
    planner = None
    if not config.use_habitat_navmesh:
        planner = OccupancyPathPlanner(
            config.map_yaml,
            OccupancyPlannerConfig(
                robot_radius=config.robot_radius,
                unknown_is_occupied=config.unknown_is_occupied,
            ),
        )
    dt = 1.0 / config.control_hz

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(config.scene)
    sensors = [
        _camera_spec(habitat_sim, "color_sensor", habitat_sim.SensorType.COLOR, config),
        _camera_spec(habitat_sim, "depth_sensor", habitat_sim.SensorType.DEPTH, config),
    ]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    output = config.output_dir
    rgb_dir = output / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    poses = []
    collisions = 0

    with AsyncPngWriter(Image) as image_writer, habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    ) as sim:
        sim.seed(config.seed)
        agent = sim.initialize_agent(0)
        start_place = resolver.places.resolve(resolver.topology_start).place
        map_start = (float(start_place.entrance_pose.x), float(start_place.entrance_pose.y))
        start = (
            np.asarray(config.simulation_start, dtype=np.float32)
            if config.simulation_start is not None
            else _habitat_position(start_place, np)
        )
        if config.map_unit_to_sim_meter <= 0:
            raise ConfigurationError("map_unit_to_sim_meter must be positive")

        def map_to_sim(point: tuple[float, float]):
            return np.asarray(
                [
                    start[0] + (point[0] - map_start[0]) * config.map_unit_to_sim_meter,
                    start[1],
                    start[2] + (point[1] - map_start[1]) * config.map_unit_to_sim_meter,
                ],
                dtype=np.float32,
            )

        def sim_to_map(position) -> list[float]:
            return [
                map_start[0] + (float(position[0]) - float(start[0])) / config.map_unit_to_sim_meter,
                map_start[1] + (float(position[2]) - float(start[2])) / config.map_unit_to_sim_meter,
            ]
        state = agent.get_state()
        state.position = start
        state.rotation = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        agent.set_state(state)

        planned_path: list[list[float]] = []
        display_path: list[list[float]] = []
        segment_paths: list[tuple[object, list[list[float]]]] = []
        cursor = map_start
        simulation_cursor = np.asarray(start, dtype=np.float32)
        for step in navigation_steps:
            goal = (float(step.place.entrance_pose.x), float(step.place.entrance_pose.y))
            if config.use_habitat_navmesh:
                shortest = habitat_sim.ShortestPath()
                shortest.requested_start = simulation_cursor
                shortest.requested_end = map_to_sim(goal)
                if not sim.pathfinder.find_path(shortest):
                    raise ConfigurationError(
                        f"No Habitat geodesic route to {step.place.place_id}"
                    )
                points = [np.asarray(point, dtype=np.float32).tolist() for point in shortest.points]
                shown = [
                    [map_point[0], 0.0, map_point[1]]
                    for map_point in (sim_to_map(point) for point in shortest.points)
                ]
                simulation_cursor = np.asarray(shortest.points[-1], dtype=np.float32)
            else:
                map_points = planner.plan(cursor, goal)
                points = [map_to_sim(point).tolist() for point in map_points]
                shown = [[x, 0.0, y] for x, y in map_points]
            if planned_path:
                planned_path.extend(points[1:])
                display_path.extend(shown[1:])
            else:
                planned_path.extend(points)
                display_path.extend(shown)
            segment_paths.append((step, points))
            cursor = goal
        if progress_callback:
            progress_callback(
                {
                    "kind": "planned",
                    "instruction": config.instruction,
                    "destination": mission.place.place_id,
                    "destination_name": mission.place.name,
                    "route": [
                        {"action": step.action.value, "id": step.place.place_id, "name": step.place.name}
                        for step in mission.steps
                    ],
                    "points": display_path,
                    "controller": "habitat_navmesh_geodesic+continuous_unicycle_velocity_control"
                    if config.use_habitat_navmesh else
                    "occupancy_grid_astar+continuous_unicycle_velocity_control",
                    "planner": "habitat_navmesh_geodesic"
                    if config.use_habitat_navmesh else "lingbot_occupancy_astar",
                    "map_yaml": str(config.map_yaml),
                }
            )

        velocity = habitat_sim.physics.VelocityControl()
        velocity.controlling_lin_vel = True
        velocity.lin_vel_is_local = True
        velocity.controlling_ang_vel = True
        velocity.ang_vel_is_local = True

        def save_frame(v: float, omega: float, collided: bool, step_index: int, target: str):
            nonlocal collisions
            observations = sim.get_sensor_observations()
            frame = len(trace)
            rgb_path = rgb_dir / f"{frame:06d}.png"
            color = np.asarray(observations["color_sensor"])[..., :3].astype(np.uint8)
            agent_state = agent.get_state()
            sensor_state = agent_state.sensor_states["color_sensor"]
            poses.append(_camera_pose_matrix(sensor_state, np, quaternion))
            rotation = agent_state.rotation
            sample = {
                "frame": frame,
                "action": "continuous_velocity",
                "linear_velocity_mps": v,
                "angular_velocity_rps": omega,
                "control_dt_sec": dt,
                "step_index": step_index,
                "step_count": len(navigation_steps),
                "destination": target,
                "position": [float(value) for value in agent_state.position],
                "map_position": sim_to_map(agent_state.position),
                "rotation_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
                "collided": collided,
            }
            collisions += int(collided)
            trace.append(sample)
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

        save_frame(0.0, 0.0, False, 0, "")
        control_steps = 0
        events.append({
            "state": "controller_started",
            "controller": "continuous_unicycle_velocity_control",
            "planner": "habitat_navmesh_geodesic"
            if config.use_habitat_navmesh else "lingbot_occupancy_astar",
            "map_yaml": str(config.map_yaml),
        })
        for step_index, (step, points) in enumerate(segment_paths, start=1):
            waypoint_index = 1 if len(points) > 1 else 0
            pacer = RealtimePacer(dt / config.realtime_factor)
            while control_steps < config.max_control_steps:
                if cancel_check and cancel_check():
                    raise ConfigurationError("Habitat continuous navigation cancelled")
                current = np.asarray(agent.get_state().position, dtype=np.float64)
                goal = np.asarray(points[-1], dtype=np.float64)
                if float(np.linalg.norm(current[[0, 2]] - goal[[0, 2]])) <= config.goal_radius:
                    break
                while waypoint_index + 1 < len(points):
                    waypoint = np.asarray(points[waypoint_index], dtype=np.float64)
                    if float(np.linalg.norm(current[[0, 2]] - waypoint[[0, 2]])) > config.waypoint_radius:
                        break
                    waypoint_index += 1
                target = np.asarray(points[waypoint_index], dtype=np.float64)
                dx, dz = float(target[0] - current[0]), float(target[2] - current[2])
                desired_yaw = math.atan2(-dx, -dz)
                heading_error = _wrap(desired_yaw - _yaw(agent.get_state().rotation))
                omega = max(
                    -config.angular_speed_rps,
                    min(config.angular_speed_rps, config.heading_gain * heading_error),
                )
                v = config.linear_speed_mps * max(0.0, math.cos(heading_error))
                if abs(heading_error) > config.turn_in_place_threshold_rad:
                    v = 0.0
                velocity.linear_velocity = mn.Vector3(0.0, 0.0, -v)
                velocity.angular_velocity = mn.Vector3(0.0, omega, 0.0)
                old_state = agent.get_state()
                rigid = habitat_sim.RigidState(
                    mn.Quaternion(
                        mn.Vector3(old_state.rotation.x, old_state.rotation.y, old_state.rotation.z),
                        old_state.rotation.w,
                    ),
                    mn.Vector3(*[float(value) for value in old_state.position]),
                )
                integrated = velocity.integrate_transform(dt, rigid)
                filtered = (
                    sim.step_filter(rigid.translation, integrated.translation)
                    if config.use_habitat_navmesh
                    else integrated.translation
                )
                intended = float((integrated.translation - rigid.translation).length())
                moved = float((filtered - rigid.translation).length())
                collided = intended > 1e-5 and moved < intended * 0.2
                next_state = agent.get_state()
                next_state.position = np.asarray(filtered, dtype=np.float32)
                next_state.rotation = quaternion.quaternion(
                    integrated.rotation.scalar,
                    integrated.rotation.vector.x,
                    integrated.rotation.vector.y,
                    integrated.rotation.vector.z,
                )
                agent.set_state(next_state)
                control_steps += 1
                save_frame(v, omega, collided, step_index, step.place.place_id)
                pacer.wait()
            else:
                raise ConfigurationError("Habitat continuous controller exceeded max_control_steps")
            error = float(
                np.linalg.norm(
                    np.asarray(agent.get_state().position, dtype=np.float64)[[0, 2]]
                    - np.asarray(points[-1], dtype=np.float64)[[0, 2]]
                )
            )
            save_frame(0.0, 0.0, False, step_index, step.place.place_id)
            events.append(
                {
                    "state": "arrived" if step_index == len(segment_paths) else "waypoint_reached",
                    "destination": step.place.place_id,
                    "position_error_m": error,
                    "frame": len(trace) - 1,
                }
            )

        with (output / "poses.txt").open("w", encoding="utf-8") as stream:
            for pose in poses:
                stream.write(" ".join(f"{float(value):.9g}" for value in pose.reshape(-1)) + "\n")
        (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
        (output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "runtime": (
                "habitat_navmesh_geodesic+habitat_sim_continuous_unicycle_velocity_control"
                if config.use_habitat_navmesh else
                "lingbot_occupancy_astar+habitat_sim_continuous_unicycle_velocity_control"
            ),
            "scene": str(config.scene),
            "instruction": config.instruction,
            "mission": mission.to_dict(),
            "frames": len(trace),
            "control_steps": control_steps,
            "collisions": collisions,
            "planned_path": planned_path,
            "terminal": events[-1],
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
