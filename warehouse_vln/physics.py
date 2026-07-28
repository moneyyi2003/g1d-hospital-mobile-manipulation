"""Dependency-light telemetry and acceptance rules for wheel-physics runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from simple_room_vln.core import Pose2D


@dataclass(frozen=True)
class PhysicsLimits:
    """Safety and terminal thresholds for one physical navigation run."""

    max_tilt_rad: float = 0.35
    max_brake_drift_m: float = 0.05
    max_stopped_linear_mps: float = 0.03
    max_stopped_angular_radps: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.max_tilt_rad,
            self.max_brake_drift_m,
            self.max_stopped_linear_mps,
            self.max_stopped_angular_radps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("physics limits must be finite and positive")


def quaternion_wxyz_to_roll_pitch(
    quaternion: Sequence[float],
) -> tuple[float, float]:
    """Return roll and pitch for a normalized-or-near-normalized quaternion."""

    if len(quaternion) != 4:
        raise ValueError("quaternion must contain w, x, y, z")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion norm is invalid")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_argument)
    return roll, pitch


class PhysicsTelemetry:
    """Accumulate physical travel and bounded, JSON-ready samples."""

    def __init__(self, sample_period_frames: int = 30) -> None:
        if sample_period_frames < 1:
            raise ValueError("sample period must be positive")
        self.sample_period_frames = sample_period_frames
        self.distance_m = 0.0
        self.max_abs_roll_rad = 0.0
        self.max_abs_pitch_rad = 0.0
        self._previous_pose: Pose2D | None = None
        self.samples: list[dict] = []

    def observe(
        self,
        *,
        frame: int,
        pose: Pose2D,
        quaternion_wxyz: Sequence[float],
        commanded_twist: Sequence[float],
        wheel_targets_radps: Sequence[float],
        wheel_actual_radps: Sequence[float],
        linear_velocity_mps: Sequence[float],
        angular_velocity_radps: Sequence[float],
        force_sample: bool = False,
    ) -> None:
        vectors = (
            commanded_twist,
            wheel_targets_radps,
            wheel_actual_radps,
            linear_velocity_mps,
            angular_velocity_radps,
        )
        if any(not all(math.isfinite(float(value)) for value in vector) for vector in vectors):
            raise ValueError("physics telemetry contains non-finite values")
        if self._previous_pose is not None:
            self.distance_m += math.dist(
                (self._previous_pose.x, self._previous_pose.y),
                (pose.x, pose.y),
            )
        self._previous_pose = pose
        roll, pitch = quaternion_wxyz_to_roll_pitch(quaternion_wxyz)
        self.max_abs_roll_rad = max(self.max_abs_roll_rad, abs(roll))
        self.max_abs_pitch_rad = max(self.max_abs_pitch_rad, abs(pitch))
        if not force_sample and frame % self.sample_period_frames:
            return
        self.samples.append(
            {
                "frame": int(frame),
                "pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
                "roll_rad": roll,
                "pitch_rad": pitch,
                "commanded_twist": {
                    "linear_mps": float(commanded_twist[0]),
                    "angular_radps": float(commanded_twist[1]),
                },
                "wheel_target_radps": [float(value) for value in wheel_targets_radps],
                "wheel_actual_radps": [float(value) for value in wheel_actual_radps],
                "base_linear_velocity_mps": [
                    float(value) for value in linear_velocity_mps
                ],
                "base_angular_velocity_radps": [
                    float(value) for value in angular_velocity_radps
                ],
            }
        )

    def to_dict(self) -> dict:
        return {
            "sample_period_frames": self.sample_period_frames,
            "physical_distance_m": self.distance_m,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_pitch_rad": self.max_abs_pitch_rad,
            "samples": self.samples,
        }


def evaluate_physics_acceptance(
    *,
    navigation_done: bool,
    position_error_m: float,
    yaw_error_rad: float,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
    max_abs_roll_rad: float,
    max_abs_pitch_rad: float,
    brake_drift_m: float,
    stopped_linear_mps: float,
    stopped_angular_radps: float,
    limits: PhysicsLimits,
) -> tuple[bool, list[str]]:
    """Evaluate navigation, stability, and post-command braking as one result."""

    checks = (
        ("navigation_incomplete", navigation_done),
        ("position_outside_tolerance", position_error_m <= position_tolerance_m),
        ("yaw_outside_tolerance", yaw_error_rad <= yaw_tolerance_rad),
        ("roll_limit_exceeded", max_abs_roll_rad <= limits.max_tilt_rad),
        ("pitch_limit_exceeded", max_abs_pitch_rad <= limits.max_tilt_rad),
        ("brake_drift_exceeded", brake_drift_m <= limits.max_brake_drift_m),
        (
            "linear_stop_speed_exceeded",
            stopped_linear_mps <= limits.max_stopped_linear_mps,
        ),
        (
            "angular_stop_speed_exceeded",
            stopped_angular_radps <= limits.max_stopped_angular_radps,
        ),
    )
    failures = [name for name, passed in checks if not passed]
    return not failures, failures


__all__ = [
    "PhysicsLimits",
    "PhysicsTelemetry",
    "evaluate_physics_acceptance",
    "quaternion_wxyz_to_roll_pitch",
]
