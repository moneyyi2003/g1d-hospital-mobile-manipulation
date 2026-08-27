"""Versioned mission, skill and failure types for the dual-brain agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class GoalKind(str, Enum):
    NAVIGATE = "navigate"
    INTERACT = "interact"


class SkillKind(str, Enum):
    NAVIGATE = "navigate"
    SEARCH_OBJECT = "search_object"
    APPROACH_ALIGN = "approach_and_align"
    MANIPULATE = "manipulate"
    VERIFY = "verify"


class SkillStatus(str, Enum):
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"


class MissionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"


class FailureCode(str, Enum):
    NONE = "none"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_OCCLUDED = "target_occluded"
    OBJECT_ID_AMBIGUOUS = "object_id_ambiguous"
    OBSERVATION_STALE = "observation_stale"
    TF_UNAVAILABLE = "tf_unavailable"
    PATH_BLOCKED = "path_blocked"
    OUT_OF_REACH = "out_of_reach"
    BAD_VIEWPOINT = "bad_viewpoint"
    BASE_NOT_STOPPED = "base_not_stopped"
    IK_INFEASIBLE = "ik_infeasible"
    COLLISION_RISK = "collision_risk"
    GRASP_FAILED = "grasp_failed"
    OBJECT_SLIPPED = "object_slipped"
    VERIFY_FAILED = "verify_failed"
    VERIFY_TIMEOUT = "verify_timeout"
    VLA_UNAVAILABLE = "vla_unavailable"
    CONTROL_LEASE_LOST = "control_lease_lost"
    UNSUPPORTED_SKILL = "unsupported_skill"
    RETRY_EXHAUSTED = "retry_exhausted"
    ADAPTER_ERROR = "adapter_error"


class ControlResource(str, Enum):
    BASE = "base"
    RIGHT_ARM = "right_arm"
    RIGHT_HAND = "right_hand"


@dataclass(frozen=True)
class TaskGoal:
    goal_id: str
    kind: GoalKind
    instruction: str
    target_id: str = ""
    action: str = ""
    region_hint: str = ""
    payload_object_id: str = ""
    success_condition: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskGoal":
        goal = cls(
            goal_id=str(value["goal_id"]).strip(),
            kind=GoalKind(str(value["kind"])),
            instruction=str(value["instruction"]).strip(),
            target_id=str(value.get("target_id", "")).strip(),
            action=str(value.get("action", "")).strip().casefold(),
            region_hint=str(value.get("region_hint", "")).strip(),
            payload_object_id=str(value.get("payload_object_id", "")).strip(),
            success_condition=str(value.get("success_condition", "")).strip(),
            metadata=dict(value.get("metadata", {})),
        )
        goal.validate()
        return goal

    def validate(self) -> None:
        if not self.goal_id or not self.instruction:
            raise ValueError("goal_id and instruction cannot be empty")
        if self.kind is GoalKind.INTERACT and (not self.target_id or not self.action):
            raise ValueError("interaction goal needs target_id and action")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        return value


@dataclass(frozen=True)
class Mission:
    mission_id: str
    instruction: str
    goals: tuple[TaskGoal, ...]
    maximum_transitions: int = 48
    maximum_attempts_per_skill: int = 2

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Mission":
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("dual-brain mission schema_version must be 1")
        mission = cls(
            mission_id=str(value["mission_id"]).strip(),
            instruction=str(value["instruction"]).strip(),
            goals=tuple(TaskGoal.from_dict(item) for item in value["goals"]),
            maximum_transitions=int(value.get("maximum_transitions", 48)),
            maximum_attempts_per_skill=int(
                value.get("maximum_attempts_per_skill", 2)
            ),
        )
        mission.validate()
        return mission

    def validate(self) -> None:
        if not self.mission_id or not self.instruction or not self.goals:
            raise ValueError("mission id, instruction and goals cannot be empty")
        goal_ids = [goal.goal_id for goal in self.goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("mission has duplicate goal_id values")
        if self.maximum_transitions < 1 or self.maximum_attempts_per_skill < 1:
            raise ValueError("mission transition and retry limits are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mission_id": self.mission_id,
            "instruction": self.instruction,
            "maximum_transitions": self.maximum_transitions,
            "maximum_attempts_per_skill": self.maximum_attempts_per_skill,
            "goals": [goal.to_dict() for goal in self.goals],
        }


@dataclass(frozen=True)
class SkillCommand:
    command_id: str
    mission_id: str
    goal_id: str
    kind: SkillKind
    instruction: str
    target_id: str = ""
    action: str = ""
    payload_object_id: str = ""
    required_resources: tuple[ControlResource, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["required_resources"] = [
            resource.value for resource in self.required_resources
        ]
        return value


@dataclass(frozen=True)
class SkillResult:
    command_id: str
    status: SkillStatus
    message: str
    failure_code: FailureCode = FailureCode.NONE
    details: dict[str, Any] = field(default_factory=dict)
    object_updates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "message": self.message,
            "failure_code": self.failure_code.value,
            "details": self.details,
            "object_updates": list(self.object_updates),
        }


@dataclass(frozen=True)
class MissionResult:
    status: MissionStatus
    message: str
    mission: Mission
    events: tuple[dict[str, Any], ...]
    memory: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status.value,
            "message": self.message,
            "mission": self.mission.to_dict(),
            "events": list(self.events),
            "memory": self.memory,
        }


__all__ = [
    "ControlResource",
    "FailureCode",
    "GoalKind",
    "Mission",
    "MissionResult",
    "MissionStatus",
    "SkillCommand",
    "SkillKind",
    "SkillResult",
    "SkillStatus",
    "TaskGoal",
]
