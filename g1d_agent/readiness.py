"""Fail-closed physical readiness gate before handing control to a VLA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping

from .interaction import InteractionProfile


class ReadinessAction(str, Enum):
    START_VLA = "start_vla"
    REACQUIRE_OBJECT = "reacquire_object"
    WAIT_FOR_STABLE_POSE = "wait_for_stable_pose"
    STOP_BASE = "stop_base"
    MOVE_CLOSER = "move_closer"
    MOVE_AWAY = "move_away"
    REALIGN_BASE = "realign_base"
    REPOSITION_FOR_REACHABILITY = "reposition_for_reachability"
    BLOCK_COLLISION = "block_collision"
    BLOCK_CONFIGURATION = "block_configuration"


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    source: str
    frame_id: str
    visible: bool
    detection_confidence: float
    stable_frames: int
    age_sec: float
    distance_m: float
    lateral_error_m: float
    yaw_error_rad: float
    pose_uncertainty_m: float
    camera_names: tuple[str, ...]
    base_linear_velocity_mps: float
    base_angular_velocity_rps: float
    ik_feasible: bool
    collision_free: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectObservation":
        boolean_fields = ("visible", "ik_feasible", "collision_free")
        invalid_booleans = [
            name for name in boolean_fields if not isinstance(value.get(name), bool)
        ]
        if invalid_booleans:
            raise ValueError(
                "object observation needs JSON booleans for "
                + ", ".join(invalid_booleans)
            )
        if not isinstance(value.get("stable_frames"), int) or isinstance(
            value.get("stable_frames"),
            bool,
        ):
            raise ValueError("object observation stable_frames must be an integer")
        camera_names = value.get("camera_names")
        if not isinstance(camera_names, (list, tuple)) or not camera_names:
            raise ValueError("object observation camera_names must be a non-empty list")
        observation = cls(
            object_id=str(value["object_id"]),
            source=str(value["source"]),
            frame_id=str(value["frame_id"]),
            visible=value["visible"],
            detection_confidence=float(value["detection_confidence"]),
            stable_frames=int(value["stable_frames"]),
            age_sec=float(value["age_sec"]),
            distance_m=float(value["distance_m"]),
            lateral_error_m=float(value["lateral_error_m"]),
            yaw_error_rad=float(value["yaw_error_rad"]),
            pose_uncertainty_m=float(value["pose_uncertainty_m"]),
            camera_names=tuple(str(item) for item in value["camera_names"]),
            base_linear_velocity_mps=float(value["base_linear_velocity_mps"]),
            base_angular_velocity_rps=float(value["base_angular_velocity_rps"]),
            ik_feasible=value["ik_feasible"],
            collision_free=value["collision_free"],
        )
        observation.validate()
        return observation

    def validate(self) -> None:
        if not self.object_id or not self.source or not self.frame_id:
            raise ValueError("object observation identity, source and frame cannot be empty")
        values = (
            self.detection_confidence,
            self.age_sec,
            self.distance_m,
            self.lateral_error_m,
            self.yaw_error_rad,
            self.pose_uncertainty_m,
            self.base_linear_velocity_mps,
            self.base_angular_velocity_rps,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("object observation has non-finite values")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("object observation confidence must be within [0, 1]")
        if self.stable_frames < 0:
            raise ValueError("object observation stable_frames cannot be negative")
        if any(
            item < 0.0
            for item in (
                self.age_sec,
                self.distance_m,
                self.pose_uncertainty_m,
            )
        ):
            raise ValueError(
                "object observation ages, distance and uncertainty must be nonnegative"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    actual: Any
    expected: Any
    recovery: ReadinessAction

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["recovery"] = self.recovery.value
        return value


@dataclass(frozen=True)
class ReadinessDecision:
    ready: bool
    action: ReadinessAction
    reason: str
    checks: tuple[ReadinessCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ready": self.ready,
            "action": self.action.value,
            "reason": self.reason,
            "checks": [item.to_dict() for item in self.checks],
        }


class VlaReadinessGate:
    """Evaluate perception, base pose, reachability and safety in priority order."""

    def evaluate(
        self,
        profile: InteractionProfile,
        observation: ObjectObservation,
        *,
        environment: str = "isaac_sim",
    ) -> ReadinessDecision:
        camera_names = set(observation.camera_names)
        required_cameras = set(profile.required_cameras)
        checks = (
            ReadinessCheck(
                "environment_allowed",
                environment in profile.allowed_environments,
                environment,
                list(profile.allowed_environments),
                ReadinessAction.BLOCK_CONFIGURATION,
            ),
            ReadinessCheck(
                "object_identity",
                observation.object_id == profile.object_id,
                observation.object_id,
                profile.object_id,
                ReadinessAction.BLOCK_CONFIGURATION,
            ),
            ReadinessCheck(
                "measurement_frame",
                observation.frame_id == profile.measurement_frame_id,
                observation.frame_id,
                profile.measurement_frame_id,
                ReadinessAction.BLOCK_CONFIGURATION,
            ),
            ReadinessCheck(
                "object_visible",
                observation.visible,
                observation.visible,
                True,
                ReadinessAction.REACQUIRE_OBJECT,
            ),
            ReadinessCheck(
                "detection_confidence",
                observation.detection_confidence
                >= profile.minimum_detection_confidence,
                observation.detection_confidence,
                {"minimum": profile.minimum_detection_confidence},
                ReadinessAction.REACQUIRE_OBJECT,
            ),
            ReadinessCheck(
                "required_cameras",
                required_cameras.issubset(camera_names),
                sorted(camera_names),
                {"contains": sorted(required_cameras)},
                ReadinessAction.REACQUIRE_OBJECT,
            ),
            ReadinessCheck(
                "observation_freshness",
                observation.age_sec <= profile.maximum_observation_age_sec,
                observation.age_sec,
                {"maximum": profile.maximum_observation_age_sec},
                ReadinessAction.REACQUIRE_OBJECT,
            ),
            ReadinessCheck(
                "pose_stability",
                observation.stable_frames >= profile.minimum_stable_frames,
                observation.stable_frames,
                {"minimum": profile.minimum_stable_frames},
                ReadinessAction.WAIT_FOR_STABLE_POSE,
            ),
            ReadinessCheck(
                "pose_uncertainty",
                observation.pose_uncertainty_m
                <= profile.maximum_pose_uncertainty_m,
                observation.pose_uncertainty_m,
                {"maximum": profile.maximum_pose_uncertainty_m},
                ReadinessAction.REACQUIRE_OBJECT,
            ),
            ReadinessCheck(
                "collision_free",
                observation.collision_free,
                observation.collision_free,
                True,
                ReadinessAction.BLOCK_COLLISION,
            ),
            ReadinessCheck(
                "base_stopped_linear",
                abs(observation.base_linear_velocity_mps)
                <= profile.maximum_base_linear_velocity_mps,
                observation.base_linear_velocity_mps,
                {"maximum_abs": profile.maximum_base_linear_velocity_mps},
                ReadinessAction.STOP_BASE,
            ),
            ReadinessCheck(
                "base_stopped_angular",
                abs(observation.base_angular_velocity_rps)
                <= profile.maximum_base_angular_velocity_rps,
                observation.base_angular_velocity_rps,
                {"maximum_abs": profile.maximum_base_angular_velocity_rps},
                ReadinessAction.STOP_BASE,
            ),
            ReadinessCheck(
                "minimum_distance",
                observation.distance_m >= profile.minimum_distance_m,
                observation.distance_m,
                {"minimum": profile.minimum_distance_m},
                ReadinessAction.MOVE_AWAY,
            ),
            ReadinessCheck(
                "maximum_distance",
                observation.distance_m <= profile.maximum_distance_m,
                observation.distance_m,
                {"maximum": profile.maximum_distance_m},
                ReadinessAction.MOVE_CLOSER,
            ),
            ReadinessCheck(
                "lateral_alignment",
                abs(observation.lateral_error_m)
                <= profile.maximum_lateral_error_m,
                observation.lateral_error_m,
                {"maximum_abs": profile.maximum_lateral_error_m},
                ReadinessAction.REALIGN_BASE,
            ),
            ReadinessCheck(
                "yaw_alignment",
                abs(observation.yaw_error_rad)
                <= profile.maximum_yaw_error_rad,
                observation.yaw_error_rad,
                {"maximum_abs": profile.maximum_yaw_error_rad},
                ReadinessAction.REALIGN_BASE,
            ),
            ReadinessCheck(
                "right_arm_ik",
                observation.ik_feasible,
                observation.ik_feasible,
                True,
                ReadinessAction.REPOSITION_FOR_REACHABILITY,
            ),
        )
        failed = next((item for item in checks if not item.passed), None)
        if failed is None:
            return ReadinessDecision(
                True,
                ReadinessAction.START_VLA,
                "物体观测、站位、IK、碰撞和底盘静止条件均满足。",
                checks,
            )
        messages = {
            ReadinessAction.REACQUIRE_OBJECT: "需要重新识别并稳定物体位姿。",
            ReadinessAction.WAIT_FOR_STABLE_POSE: "物体位姿尚未稳定。",
            ReadinessAction.STOP_BASE: "底盘尚未停止。",
            ReadinessAction.MOVE_CLOSER: "机器人离物体过远，需要局部靠近。",
            ReadinessAction.MOVE_AWAY: "机器人离物体过近，需要后退。",
            ReadinessAction.REALIGN_BASE: "机器人相对物体的横向位置或朝向不合适。",
            ReadinessAction.REPOSITION_FOR_REACHABILITY: "右臂 IK 不可达，需要更换站位。",
            ReadinessAction.BLOCK_COLLISION: "碰撞检查未通过，禁止启动 VLA。",
            ReadinessAction.BLOCK_CONFIGURATION: "物体、环境或交互配置不匹配。",
        }
        return ReadinessDecision(
            False,
            failed.recovery,
            messages[failed.recovery],
            checks,
        )


__all__ = [
    "ObjectObservation",
    "ReadinessAction",
    "ReadinessCheck",
    "ReadinessDecision",
    "VlaReadinessGate",
]
