"""Drive a Habitat agent along a geodesic path and export RGB-D/pose/semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from ..errors import ConfigurationError


@dataclass(frozen=True)
class HabitatCollectConfig:
    scene: Path
    output_dir: Path
    dataset_config: Path | None = None
    width: int = 640
    height: int = 480
    sensor_height: float = 1.0
    hfov_degrees: float = 90.0
    forward_step: float = 0.25
    turn_degrees: float = 10.0
    goal_radius: float = 0.30
    max_steps: int = 2000
    seed: int = 7
    start: Sequence[float] | None = None
    goal: Sequence[float] | None = None
    semantic: bool = True


def _imports():
    try:
        import habitat_sim
        import numpy as np
        import quaternion
        from PIL import Image
    except ImportError as exc:
        raise ConfigurationError(
            "Habitat collection needs habitat-sim, NumPy, numpy-quaternion, and Pillow"
        ) from exc
    return habitat_sim, np, quaternion, Image


def _camera_spec(habitat_sim, uuid: str, sensor_type, config: HabitatCollectConfig):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = [config.height, config.width]
    spec.position = [0.0, config.sensor_height, 0.0]
    spec.hfov = config.hfov_degrees
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return spec


def _camera_pose_matrix(sensor_state, np, quaternion):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    matrix[:3, 3] = np.asarray(sensor_state.position, dtype=np.float64)
    return matrix


def collect_habitat(config: HabitatCollectConfig) -> dict[str, object]:
    habitat_sim, np, quaternion, Image = _imports()
    if not config.scene.exists():
        raise ConfigurationError(f"Habitat scene not found: {config.scene}")
    if config.dataset_config is not None and not config.dataset_config.exists():
        raise ConfigurationError(f"Habitat dataset config not found: {config.dataset_config}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(config.scene)
    if config.dataset_config is not None:
        sim_cfg.scene_dataset_config_file = str(config.dataset_config)

    sensors = [
        _camera_spec(habitat_sim, "color_sensor", habitat_sim.SensorType.COLOR, config),
        _camera_spec(habitat_sim, "depth_sensor", habitat_sim.SensorType.DEPTH, config),
    ]
    if config.semantic:
        sensors.append(
            _camera_spec(
                habitat_sim,
                "semantic_sensor",
                habitat_sim.SensorType.SEMANTIC,
                config,
            )
        )

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
    rgb_dir, depth_dir, semantic_dir = output / "rgb", output / "depth", output / "semantic"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if config.semantic:
        semantic_dir.mkdir(parents=True, exist_ok=True)

    with habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg])) as sim:
        sim.seed(config.seed)
        agent = sim.initialize_agent(0)
        start = np.asarray(
            config.start
            if config.start is not None
            else sim.pathfinder.get_random_navigable_point(),
            dtype=np.float32,
        )
        goal = np.asarray(
            config.goal
            if config.goal is not None
            else sim.pathfinder.get_random_navigable_point(),
            dtype=np.float32,
        )
        if start.shape != (3,) or goal.shape != (3,):
            raise ConfigurationError("Habitat start and goal must each contain x,y,z")
        if not sim.pathfinder.is_navigable(start) or not sim.pathfinder.is_navigable(goal):
            raise ConfigurationError("Habitat start or goal is outside the navmesh")

        state = agent.get_state()
        state.position = start
        agent.set_state(state)
        follower = habitat_sim.nav.GreedyGeodesicFollower(
            sim.pathfinder,
            agent,
            goal_radius=config.goal_radius,
        )
        try:
            actions = follower.find_path(goal)
        except habitat_sim.errors.GreedyFollowerError as exc:
            raise ConfigurationError(f"Habitat geodesic follower failed: {exc}") from exc
        actions = [action for action in actions if action is not None]
        if len(actions) > config.max_steps:
            actions = actions[: config.max_steps]

        poses = []
        action_log = []

        def save_frame(index: int) -> None:
            observations = sim.get_sensor_observations()
            color = np.asarray(observations["color_sensor"])[..., :3].astype(np.uint8)
            Image.fromarray(color).save(rgb_dir / f"{index:06d}.png")
            np.save(depth_dir / f"{index:06d}.npy", observations["depth_sensor"])
            if config.semantic and "semantic_sensor" in observations:
                np.save(semantic_dir / f"{index:06d}.npy", observations["semantic_sensor"])
            sensor_state = agent.get_state().sensor_states["color_sensor"]
            poses.append(_camera_pose_matrix(sensor_state, np, quaternion))

        save_frame(0)
        for index, action in enumerate(actions, start=1):
            sim.step(action)
            action_log.append(str(action))
            save_frame(index)

        poses_path = output / "poses.txt"
        with poses_path.open("w", encoding="utf-8") as stream:
            for pose in poses:
                stream.write(" ".join(f"{float(value):.9g}" for value in pose.reshape(-1)) + "\n")

        fx = config.width / (2.0 * math.tan(math.radians(config.hfov_degrees) / 2.0))
        intrinsics = {
            "width": config.width,
            "height": config.height,
            "fx": fx,
            "fy": fx,
            "cx": (config.width - 1) / 2.0,
            "cy": (config.height - 1) / 2.0,
            "depth_unit": "meter",
            "pose_convention": "camera_to_habitat_world, row-major 4x4",
        }
        (output / "intrinsics.json").write_text(
            json.dumps(intrinsics, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "scene": str(config.scene),
            "dataset_config": str(config.dataset_config) if config.dataset_config else None,
            "start": start.tolist(),
            "goal": goal.tolist(),
            "frames": len(poses),
            "actions": action_log,
            "semantic_enabled": config.semantic,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
