"""RGB-only Habitat renderer driven by an external planar robot pose.

The renderer deliberately exposes no navigation API.  A render-only Sim(2)
calibration maps LingBot map poses into Habitat's X/Z plane.  The calibration
is consumed only in this module and is never published to Nav2, TF, odometry,
goal generation, collision checking, or any occupancy-map component.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from ..errors import ConfigurationError
from .habitat_collector import _camera_spec, _imports


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class HabitatRenderAlignment:
    """Render-only LingBot map -> Habitat X/Z similarity transform."""

    scale: float
    rotation_rad: float
    translation_x: float
    translation_z: float
    yaw_offset_rad: float
    simulator_height: float
    correspondence_count: int
    position_rmse_m: float
    map_start_yaw_rad: float = -math.pi / 2.0
    source: str = "rgb_frame_correspondence_sim2"

    def validate(self) -> None:
        values = (
            self.scale,
            self.rotation_rad,
            self.translation_x,
            self.translation_z,
            self.yaw_offset_rad,
            self.simulator_height,
            self.position_rmse_m,
            self.map_start_yaw_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("Habitat render alignment contains non-finite values")
        if self.scale <= 0.0:
            raise ConfigurationError("Habitat render alignment scale must be positive")
        if self.correspondence_count < 2:
            raise ConfigurationError(
                "Habitat render alignment needs at least two RGB correspondences"
            )
        if self.position_rmse_m < 0.0:
            raise ConfigurationError("Habitat render alignment RMSE cannot be negative")
        if self.source != "rgb_frame_correspondence_sim2":
            raise ConfigurationError("Unsupported Habitat render alignment source")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HabitatRenderAlignment":
        if value.get("artifact_type") != "render_only_lingbot_to_habitat_sim2":
            raise ConfigurationError("Unsupported Habitat render alignment artifact")
        if value.get("consumer") != "habitat_rgb_camera_only":
            raise ConfigurationError("Habitat alignment must be restricted to the RGB camera")
        prohibited = value.get("prohibited_consumers")
        required = {"nav2", "tf", "odom", "goal_generation", "collision_checking"}
        if not isinstance(prohibited, list) or not required.issubset(map(str, prohibited)):
            raise ConfigurationError(
                "Habitat alignment must explicitly prohibit every navigation consumer"
            )
        sim2 = value.get("sim2")
        fit = value.get("fit")
        if not isinstance(sim2, Mapping) or not isinstance(fit, Mapping):
            raise ConfigurationError("Habitat render alignment lacks Sim(2) fit metadata")
        translation = sim2.get("translation_xz_m")
        if not isinstance(translation, Sequence) or len(translation) != 2:
            raise ConfigurationError("Habitat render alignment translation must be [x, z]")
        alignment = cls(
            scale=float(sim2["scale_m_per_lingbot_unit"]),
            rotation_rad=float(sim2["rotation_rad"]),
            translation_x=float(translation[0]),
            translation_z=float(translation[1]),
            yaw_offset_rad=float(sim2["yaw_offset_rad"]),
            simulator_height=float(value["simulator_height_m"]),
            correspondence_count=int(fit["correspondence_count"]),
            position_rmse_m=float(fit["position_rmse_m"]),
            map_start_yaw_rad=float(value["map_start_yaw_rad"]),
            source=str(fit.get("source", "")),
        )
        alignment.validate()
        return alignment

    @classmethod
    def load(cls, path: str | Path) -> "HabitatRenderAlignment":
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise ConfigurationError(f"Habitat render alignment not found: {target}")
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Cannot read Habitat render alignment: {target}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ConfigurationError("Habitat render alignment must be a JSON object")
        return cls.from_mapping(value)


def fit_render_alignment(
    map_points: Sequence[Sequence[float]],
    simulator_points: Sequence[Sequence[float]],
    *,
    map_yaws: Sequence[float],
    simulator_yaws: Sequence[float],
    simulator_height: float,
) -> HabitatRenderAlignment:
    """Least-squares Sim(2) and circular yaw fit from paired RGB frames."""
    if not (
        len(map_points)
        == len(simulator_points)
        == len(map_yaws)
        == len(simulator_yaws)
    ):
        raise ConfigurationError("Render alignment correspondence lengths differ")
    if len(map_points) < 2:
        raise ConfigurationError("Render alignment needs at least two RGB frames")
    try:
        source = [complex(float(point[0]), float(point[1])) for point in map_points]
        target = [
            complex(float(point[0]), float(point[1])) for point in simulator_points
        ]
        map_angles = [float(value) for value in map_yaws]
        simulator_angles = [float(value) for value in simulator_yaws]
    except (IndexError, TypeError, ValueError) as exc:
        raise ConfigurationError("Invalid render alignment correspondence") from exc
    values = [
        component
        for point in (*source, *target)
        for component in (point.real, point.imag)
    ]
    values.extend((*map_angles, *simulator_angles, float(simulator_height)))
    if not all(math.isfinite(value) for value in values):
        raise ConfigurationError("Render alignment correspondence is non-finite")
    source_mean = sum(source) / len(source)
    target_mean = sum(target) / len(target)
    denominator = sum(abs(point - source_mean) ** 2 for point in source)
    if denominator <= 1e-12:
        raise ConfigurationError("Render alignment map correspondences are degenerate")
    coefficient = sum(
        (point - source_mean).conjugate() * (other - target_mean)
        for point, other in zip(source, target)
    ) / denominator
    if abs(coefficient) <= 1e-12:
        raise ConfigurationError("Render alignment fitted a degenerate scale")
    translation = target_mean - coefficient * source_mean
    residuals = [
        abs(coefficient * point + translation - other)
        for point, other in zip(source, target)
    ]
    yaw_vector = sum(
        complex(math.cos(target_yaw + map_yaw), math.sin(target_yaw + map_yaw))
        for map_yaw, target_yaw in zip(map_angles, simulator_angles)
    )
    if abs(yaw_vector) <= 1e-12:
        raise ConfigurationError("Render alignment yaw correspondences are degenerate")
    result = HabitatRenderAlignment(
        scale=abs(coefficient),
        rotation_rad=math.atan2(coefficient.imag, coefficient.real),
        translation_x=translation.real,
        translation_z=translation.imag,
        yaw_offset_rad=math.atan2(yaw_vector.imag, yaw_vector.real),
        simulator_height=float(simulator_height),
        correspondence_count=len(source),
        position_rmse_m=math.sqrt(
            sum(residual * residual for residual in residuals) / len(residuals)
        ),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class HabitatRgbCameraConfig:
    scene: str | Path
    scene_dataset_config: Path | None
    alignment: HabitatRenderAlignment
    width: int = 640
    height: int = 480
    sensor_height: float = 1.0
    hfov_degrees: float = 90.0

    def validate(self) -> None:
        self.alignment.validate()
        values = (self.sensor_height, self.hfov_degrees)
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("Habitat RGB renderer config contains non-finite values")
        if self.width <= 0 or self.height <= 0 or self.sensor_height < 0.0:
            raise ConfigurationError("Habitat RGB renderer dimensions are invalid")
        if not 10.0 <= self.hfov_degrees < 180.0:
            raise ConfigurationError("Habitat RGB renderer field of view is invalid")
        dataset = self.scene_dataset_config
        scene_path = Path(self.scene).expanduser()
        if not scene_path.is_file() and dataset is None:
            raise ConfigurationError(
                "A Habitat scene file or scene dataset config is required for RGB rendering"
            )
        if dataset is not None and not dataset.expanduser().resolve().is_file():
            raise ConfigurationError(f"Habitat scene dataset config not found: {dataset}")


def map_pose_to_simulator(
    config: HabitatRgbCameraConfig,
    x: float,
    y: float,
    yaw: float,
) -> tuple[tuple[float, float, float], float]:
    """Convert a LingBot pose with the camera-only calibrated Sim(2)."""
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ConfigurationError("Habitat RGB render pose contains non-finite values")
    alignment = config.alignment
    cosine = math.cos(alignment.rotation_rad)
    sine = math.sin(alignment.rotation_rad)
    simulator_x = alignment.translation_x + alignment.scale * (cosine * x - sine * y)
    simulator_z = alignment.translation_z + alignment.scale * (sine * x + cosine * y)
    # Habitat yaw and ROS planar yaw use opposite signs.  The separately fitted
    # offset also includes Habitat's -Z camera-forward convention.
    simulator_yaw = _wrap_angle(alignment.yaw_offset_rad - yaw)
    return (
        (simulator_x, alignment.simulator_height, simulator_z),
        simulator_yaw,
    )


class HabitatRgbCamera:
    """Render textured first-person RGB at externally supplied map poses."""

    def __init__(self, config: HabitatRgbCameraConfig) -> None:
        config.validate()
        habitat_sim, np, quaternion, Image = _imports()
        self._np = np
        self._quaternion = quaternion
        self._Image = Image
        self.config = config

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(config.scene)
        if config.scene_dataset_config is not None:
            sim_cfg.scene_dataset_config_file = str(
                config.scene_dataset_config.expanduser().resolve()
            )
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [
            _camera_spec(
                habitat_sim,
                "color_sensor",
                habitat_sim.SensorType.COLOR,
                config,
            )
        ]
        # No navmesh is loaded or queried here.  The simulator is used only to
        # rasterize the configured COLOR sensor from caller-provided poses.
        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self._agent = self._sim.initialize_agent(0)

    def render(self, x: float, y: float, yaw: float):
        position, simulator_yaw = map_pose_to_simulator(self.config, x, y, yaw)
        state = self._agent.get_state()
        state.position = self._np.asarray(position, dtype=self._np.float32)
        state.rotation = self._quaternion.from_rotation_vector(
            self._np.asarray([0.0, simulator_yaw, 0.0], dtype=self._np.float64)
        )
        self._agent.set_state(state, reset_sensors=True)
        rgba = self._np.asarray(
            self._sim.get_sensor_observations()["color_sensor"]
        )
        return self._Image.fromarray(rgba[..., :3].astype(self._np.uint8), mode="RGB")

    def render_to(
        self,
        target: str | Path,
        *,
        x: float,
        y: float,
        yaw: float,
        jpeg_quality: int = 85,
    ) -> tuple[Path, bytes]:
        """Atomically save a PNG snapshot and return the same frame as JPEG."""
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("JPEG quality must be between 1 and 95")
        image = self.render(x, y, yaw)
        path = Path(target).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=jpeg_quality)
        return path, stream.getvalue()

    def close(self) -> None:
        self._sim.close()


__all__ = [
    "HabitatRenderAlignment",
    "HabitatRgbCamera",
    "HabitatRgbCameraConfig",
    "fit_render_alignment",
    "map_pose_to_simulator",
]
