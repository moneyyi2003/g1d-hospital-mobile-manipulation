"""Object-and-skill interaction profiles for manipulation staging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


_CALIBRATION_STATES = {"provisional", "sim_validated", "real_validated"}


@dataclass(frozen=True)
class InteractionProfile:
    profile_id: str
    object_id: str
    name: str
    aliases: tuple[str, ...]
    skill: str
    measurement_frame_id: str
    distance_reference: str
    preferred_distance_m: float
    minimum_distance_m: float
    maximum_distance_m: float
    maximum_yaw_error_rad: float
    maximum_lateral_error_m: float
    minimum_detection_confidence: float
    minimum_stable_frames: int
    maximum_pose_uncertainty_m: float
    maximum_observation_age_sec: float
    maximum_base_linear_velocity_mps: float
    maximum_base_angular_velocity_rps: float
    required_cameras: tuple[str, ...]
    preferred_hand: str
    allowed_environments: tuple[str, ...]
    calibration_status: str
    success: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InteractionProfileDatabase:
    """Strict lookup keyed by object identity and manipulation skill."""

    def __init__(self, profiles: tuple[InteractionProfile, ...], source: Path) -> None:
        if not profiles:
            raise ValueError("interaction profile database is empty")
        keys = [(item.object_id, item.skill) for item in profiles]
        if len(keys) != len(set(keys)):
            raise ValueError("interaction profile database has duplicate object/skill keys")
        self.profiles = profiles
        self.source = source

    @classmethod
    def load(cls, path: Path) -> "InteractionProfileDatabase":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("interaction profile schema_version must be 1")
        profiles = tuple(
            cls._parse_profile(value) for value in payload.get("profiles", [])
        )
        return cls(profiles, path)

    @staticmethod
    def _parse_profile(value: dict[str, Any]) -> InteractionProfile:
        distance = value.get("base_staging", {})
        perception = value.get("perception", {})
        safety = value.get("safety", {})
        profile = InteractionProfile(
            profile_id=str(value["id"]),
            object_id=str(value["object_id"]),
            name=str(value["name"]),
            aliases=tuple(str(item) for item in value.get("aliases", [])),
            skill=str(value["skill"]).casefold(),
            measurement_frame_id=str(distance["frame_id"]),
            distance_reference=str(distance["distance_reference"]),
            preferred_distance_m=float(distance["preferred_distance_m"]),
            minimum_distance_m=float(distance["minimum_distance_m"]),
            maximum_distance_m=float(distance["maximum_distance_m"]),
            maximum_yaw_error_rad=float(distance["maximum_yaw_error_rad"]),
            maximum_lateral_error_m=float(distance["maximum_lateral_error_m"]),
            minimum_detection_confidence=float(
                perception["minimum_detection_confidence"]
            ),
            minimum_stable_frames=int(perception["minimum_stable_frames"]),
            maximum_pose_uncertainty_m=float(
                perception["maximum_pose_uncertainty_m"]
            ),
            maximum_observation_age_sec=float(
                perception["maximum_observation_age_sec"]
            ),
            maximum_base_linear_velocity_mps=float(
                safety["maximum_base_linear_velocity_mps"]
            ),
            maximum_base_angular_velocity_rps=float(
                safety["maximum_base_angular_velocity_rps"]
            ),
            required_cameras=tuple(
                str(item) for item in perception["required_cameras"]
            ),
            preferred_hand=str(value["preferred_hand"]),
            allowed_environments=tuple(
                str(item) for item in value["allowed_environments"]
            ),
            calibration_status=str(value["calibration_status"]),
            success=dict(value.get("success", {})),
        )
        InteractionProfileDatabase._validate(profile)
        return profile

    @staticmethod
    def _validate(profile: InteractionProfile) -> None:
        finite_values = (
            profile.preferred_distance_m,
            profile.minimum_distance_m,
            profile.maximum_distance_m,
            profile.maximum_yaw_error_rad,
            profile.maximum_lateral_error_m,
            profile.minimum_detection_confidence,
            profile.maximum_pose_uncertainty_m,
            profile.maximum_observation_age_sec,
            profile.maximum_base_linear_velocity_mps,
            profile.maximum_base_angular_velocity_rps,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError(
                f"interaction profile {profile.profile_id} has non-finite values"
            )
        if not (
            0.0
            < profile.minimum_distance_m
            <= profile.preferred_distance_m
            <= profile.maximum_distance_m
        ):
            raise ValueError(
                f"interaction profile {profile.profile_id} has invalid distance interval"
            )
        if not profile.measurement_frame_id or not profile.distance_reference:
            raise ValueError(
                f"interaction profile {profile.profile_id} lacks distance frame/reference"
            )
        if not 0.0 <= profile.minimum_detection_confidence <= 1.0:
            raise ValueError(
                f"interaction profile {profile.profile_id} has invalid confidence"
            )
        if profile.minimum_stable_frames < 1:
            raise ValueError(
                f"interaction profile {profile.profile_id} needs stable frames"
            )
        if any(value < 0.0 for value in finite_values[3:]):
            raise ValueError(
                f"interaction profile {profile.profile_id} has negative limits"
            )
        if not profile.required_cameras:
            raise ValueError(
                f"interaction profile {profile.profile_id} has no required cameras"
            )
        if not profile.allowed_environments:
            raise ValueError(
                f"interaction profile {profile.profile_id} has no allowed environment"
            )
        if not profile.preferred_hand or not profile.success:
            raise ValueError(
                f"interaction profile {profile.profile_id} lacks hand/success criteria"
            )
        if profile.calibration_status not in _CALIBRATION_STATES:
            raise ValueError(
                f"interaction profile {profile.profile_id} has invalid calibration status"
            )

    def resolve(self, instruction: str, skill: str) -> InteractionProfile:
        normalized = instruction.casefold().strip()
        normalized_skill = skill.casefold().strip()
        matches = [
            profile
            for profile in self.profiles
            if profile.skill == normalized_skill
            and any(
                alias.casefold() in normalized
                for alias in (profile.object_id, profile.name, *profile.aliases)
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "interaction command must resolve to exactly one "
                f"object/skill profile, got {len(matches)} for skill {normalized_skill!r}"
            )
        return matches[0]


__all__ = ["InteractionProfile", "InteractionProfileDatabase"]
