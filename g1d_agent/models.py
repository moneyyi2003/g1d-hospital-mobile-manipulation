"""Serializable mission types shared by the agent and its adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Capability(str, Enum):
    VLN = "vln"
    VLA = "vla"


class StepKind(str, Enum):
    SEMANTIC_NAVIGATION = "semantic_navigation"
    PREGRASP_DOCKING = "pregrasp_docking"
    MANIPULATION = "manipulation"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


class MissionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class TaskStep:
    step_id: str
    capability: Capability
    kind: StepKind
    instruction: str
    success_condition: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capability"] = self.capability.value
        value["kind"] = self.kind.value
        return value


@dataclass(frozen=True)
class MissionPlan:
    instruction: str
    route: str
    reason: str
    steps: tuple[TaskStep, ...]
    planner: str = "rule_task_planner_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "instruction": self.instruction,
            "route": self.route,
            "reason": self.reason,
            "planner": self.planner,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class StepResult:
    step_id: str
    status: StepStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True)
class MissionResult:
    status: MissionStatus
    message: str
    plan: MissionPlan
    steps: tuple[StepResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status.value,
            "message": self.message,
            "plan": self.plan.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }
