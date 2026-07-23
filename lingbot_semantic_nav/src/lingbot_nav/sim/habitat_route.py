"""Execute a reviewed semantic route with Habitat's discrete wheeled actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

from ..errors import ConfigurationError
from ..mission import MissionResolver
from ..models import Mission
from .habitat_collector import _camera_pose_matrix, _camera_spec, _imports
from .frame_writer import AsyncPngWriter, RealtimePacer
from .occupancy_planner import OccupancyPathPlanner, OccupancyPlannerConfig


@dataclass(frozen=True)
class HabitatRouteConfig:
    scene: Path
    output_dir: Path
    instruction: str
    width: int = 640
    height: int = 480
    sensor_height: float = 1.0
    hfov_degrees: float = 90.0
    forward_step: float = 0.25
    turn_degrees: float = 10.0
    goal_radius: float = 0.30
    max_steps: int = 500
    seed: int = 7
    save_depth: bool = True
    step_delay: float = 0.0
    map_yaml: Path | None = None
    robot_radius: float = 0.14
    simulation_start: tuple[float, float, float] | None = None
    map_unit_to_sim_meter: float = 1.0
    unknown_is_occupied: bool = True


def executable_navigation_steps(steps):
    """Exclude map-collection-only scan poses from live navigation goals."""
    values = list(steps)
    executable = [step for step in values if not step.place.metadata.get("internal")]
    return executable or values[-1:]


def _habitat_position(place, np):
    try:
        habitat_y = float(place.metadata["habitat_y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Habitat place {place.place_id!r} needs numeric metadata.habitat_y"
        ) from exc
    return np.asarray(
        [place.entrance_pose.x, habitat_y, place.entrance_pose.y],
        dtype=np.float32,
    )


def run_habitat_route(
    config: HabitatRouteConfig,
    resolver: MissionResolver,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, object]:
    habitat_sim, np, quaternion, Image = _imports()
    if not config.scene.is_file():
        raise ConfigurationError(f"Habitat scene not found: {config.scene}")
    mission: Mission = resolver.resolve(config.instruction)
    if not mission.steps:
        raise ConfigurationError("Resolved Habitat mission has no steps")
    if config.map_yaml is None:
        raise ConfigurationError("Generated occupancy map is required for Habitat navigation")
    planner = OccupancyPathPlanner(
        config.map_yaml,
        OccupancyPlannerConfig(
            robot_radius=config.robot_radius,
            unknown_is_occupied=config.unknown_is_occupied,
        ),
    )

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(config.scene)
    sensors = [
        _camera_spec(habitat_sim, "color_sensor", habitat_sim.SensorType.COLOR, config),
        _camera_spec(habitat_sim, "depth_sensor", habitat_sim.SensorType.DEPTH, config),
    ]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=config.forward_step)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=config.turn_degrees)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=config.turn_degrees)
        ),
    }

    output = config.output_dir
    rgb_dir, depth_dir = output / "rgb", output / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if config.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    poses = []
    action_count = 0
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
                ], dtype=np.float32,
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

        def save_frame(action: str, step_index: int, destination: str, collided: bool) -> None:
            frame_index = len(trace)
            observations = sim.get_sensor_observations()
            color = np.asarray(observations["color_sensor"])[..., :3].astype(np.uint8)
            rgb_path = rgb_dir / f"{frame_index:06d}.png"
            if config.save_depth:
                np.save(depth_dir / f"{frame_index:06d}.npy", observations["depth_sensor"])
            agent_state = agent.get_state()
            sensor_state = agent_state.sensor_states["color_sensor"]
            pose = _camera_pose_matrix(sensor_state, np, quaternion)
            poses.append(pose)
            rotation = agent_state.rotation
            trace.append(
                {
                    "frame": frame_index,
                    "action": action,
                    "step_index": step_index,
                    "step_count": len(mission.steps),
                    "destination": destination,
                    "position": [float(item) for item in agent_state.position],
                    "map_position": sim_to_map(agent_state.position),
                    "rotation_xyzw": [
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ],
                    "collided": collided,
                }
            )
            image_writer.submit(
                color,
                rgb_path,
                (
                    lambda saved_path, jpeg, sample=trace[-1]: progress_callback(
                        {
                            "kind": "frame",
                            "sample": sample,
                            "rgb_path": str(saved_path),
                            "jpeg_bytes": jpeg,
                        }
                    )
                )
                if progress_callback is not None
                else None,
            )

        save_frame("start", 0, "", False)
        events.append({"state": "queued", "instruction": config.instruction, "frame": 0})
        events.append(
            {
                "state": "understanding",
                "frame": 0,
                "route_constraints": [
                    item.value for item in mission.intent.route_constraints
                ],
            }
        )

        planned_path: list[list[float]] = []
        display_path: list[list[float]] = []
        segment_paths: list[tuple[object, list[list[float]]]] = []
        planned_start = map_start
        for step in mission.steps:
            goal = (float(step.place.entrance_pose.x), float(step.place.entrance_pose.y))
            map_points = planner.plan(
                planned_start,
                goal,
            )
            points = [map_to_sim(point).tolist() for point in map_points]
            shown = [[x, 0.0, y] for x, y in map_points]
            if planned_path and points:
                planned_path.extend(points[1:])
                display_path.extend(shown[1:])
            else:
                planned_path.extend(points)
                display_path.extend(shown)
            segment_paths.append((step, points))
            planned_start = goal
        if progress_callback is not None:
            progress_callback(
                {
                    "kind": "planned",
                    "instruction": config.instruction,
                    "destination": mission.place.place_id,
                    "destination_name": mission.place.name,
                    "route": [
                        {
                            "action": step.action.value,
                            "id": step.place.place_id,
                            "name": step.place.name,
                        }
                        for step in mission.steps
                    ],
                    "points": display_path,
                    "planner": "lingbot_occupancy_astar",
                    "map_yaml": str(config.map_yaml),
                }
            )

        pacer = RealtimePacer(config.step_delay) if config.step_delay > 0 else None
        for step_index, (step, points) in enumerate(segment_paths, start=1):
            goal = np.asarray(points[-1], dtype=np.float32)
            events.append(
                {
                    "state": "goal_resolved",
                    "frame": len(trace) - 1,
                    "step_index": step_index,
                    "step_count": len(mission.steps),
                    "destination": step.place.place_id,
                    "destination_name": step.place.name,
                    "route_action": step.action.value,
                    "goal": [float(item) for item in goal],
                }
            )
            waypoint_index = 1 if len(points) > 1 else 0
            while waypoint_index < len(points):
                if cancel_check is not None and cancel_check():
                    raise ConfigurationError("Habitat navigation cancelled")
                if action_count >= config.max_steps:
                    raise ConfigurationError("Habitat route exceeded max_steps")
                current_state = agent.get_state()
                current = np.asarray(current_state.position, dtype=np.float64)
                target = np.asarray(points[waypoint_index], dtype=np.float64)
                distance = float(np.linalg.norm(current[[0, 2]] - target[[0, 2]]))
                radius = config.goal_radius if waypoint_index == len(points) - 1 else config.forward_step * 0.75
                if distance <= radius:
                    waypoint_index += 1
                    continue
                rotation = current_state.rotation
                yaw = math.atan2(
                    2.0 * (rotation.w * rotation.y + rotation.x * rotation.z),
                    1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
                )
                desired_yaw = math.atan2(
                    -float(target[0] - current[0]),
                    -float(target[2] - current[2]),
                )
                heading_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
                turn_threshold = math.radians(config.turn_degrees) * 0.55
                if heading_error > turn_threshold:
                    action = "turn_left"
                elif heading_error < -turn_threshold:
                    action = "turn_right"
                else:
                    action = "move_forward"
                before = np.asarray(agent.get_state().position, dtype=np.float64)
                sim.step(action)
                after = np.asarray(agent.get_state().position, dtype=np.float64)
                moved = float(np.linalg.norm(after - before))
                collided = str(action) == "move_forward" and moved < config.forward_step * 0.1
                collisions += int(collided)
                action_count += 1
                save_frame(str(action), step_index, step.place.place_id, collided)
                if pacer is not None:
                    pacer.wait()
            final_position = np.asarray(agent.get_state().position, dtype=np.float64)
            error = float(np.linalg.norm(final_position - goal.astype(np.float64)))
            if error > config.goal_radius + 1e-3:
                raise ConfigurationError(
                    f"Habitat waypoint {step.place.place_id} ended {error:.3f} m from goal"
                )
            events.append(
                {
                    "state": "waypoint_reached" if step_index < len(mission.steps) else "arrived",
                    "frame": len(trace) - 1,
                    "step_index": step_index,
                    "step_count": len(mission.steps),
                    "destination": step.place.place_id,
                    "position_error_m": error,
                }
            )

        with (output / "poses.txt").open("w", encoding="utf-8") as stream:
            for pose in poses:
                stream.write(
                    " ".join(f"{float(value):.9g}" for value in pose.reshape(-1)) + "\n"
                )
        (output / "trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "events.json").write_text(
            json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "runtime": "lingbot_occupancy_astar+habitat_sim_discrete_wheeled_actions",
            "scene": str(config.scene),
            "instruction": config.instruction,
            "mission": mission.to_dict(),
            "frames": len(trace),
            "actions": action_count,
            "collisions": collisions,
            "planned_path": planned_path,
            "terminal": events[-1],
            "events": events,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
