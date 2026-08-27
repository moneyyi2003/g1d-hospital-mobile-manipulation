"""Shared object-level world memory and mission blackboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Mapping


@dataclass
class ObjectRecord:
    object_id: str
    labels: list[str] = field(default_factory=list)
    description: str = ""
    global_pose: dict[str, Any] | None = None
    local_pose: dict[str, Any] | None = None
    room_id: str = ""
    support_surface_id: str = ""
    visible: bool = False
    detection_confidence: float = 0.0
    pose_uncertainty_m: float | None = None
    last_seen_monotonic_sec: float | None = None
    observation_source: str = ""
    map_revision: str = ""
    reachable: bool | None = None
    reachability_context: dict[str, Any] = field(default_factory=dict)
    attachment_state: str = "world"
    parent_frame: str = "map"
    last_action: str = ""
    last_result: str = ""
    last_failure_code: str = "none"
    revision: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectRecord":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value[key] for key in known if key in value})

    def apply(self, update: Mapping[str, Any]) -> None:
        immutable = {"object_id", "revision"}
        known = set(self.__dataclass_fields__)
        unknown = set(update) - known
        if unknown:
            raise ValueError(f"unknown object memory fields: {sorted(unknown)}")
        if "object_id" in update and str(update["object_id"]) != self.object_id:
            raise ValueError("object memory update cannot change object_id")
        for key, value in update.items():
            if key not in immutable:
                setattr(self, key, value)
        self.revision += 1

    def observation_is_fresh(self, maximum_age_sec: float, now: float) -> bool:
        return (
            self.visible
            and self.last_seen_monotonic_sec is not None
            and 0.0 <= now - self.last_seen_monotonic_sec <= maximum_age_sec
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoalProgress:
    goal_id: str
    region_reached: bool = False
    aligned: bool = False
    manipulated: bool = False
    verified: bool = False
    attempts: dict[str, int] = field(default_factory=dict)
    last_failure_code: str = "none"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoalProgress":
        return cls(
            goal_id=str(value["goal_id"]),
            region_reached=bool(value.get("region_reached", False)),
            aligned=bool(value.get("aligned", False)),
            manipulated=bool(value.get("manipulated", False)),
            verified=bool(value.get("verified", False)),
            attempts={
                str(key): int(item)
                for key, item in dict(value.get("attempts", {})).items()
            },
            last_failure_code=str(value.get("last_failure_code", "none")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskBlackboard:
    mission_id: str = ""
    current_goal_index: int = 0
    current_phase: str = "idle"
    active_object_id: str = ""
    carried_object_id: str = ""
    plan_revision: int = 0
    control_owner: str = ""
    goals: dict[str, GoalProgress] = field(default_factory=dict)
    last_failure_code: str = "none"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskBlackboard":
        blackboard = cls(
            mission_id=str(value.get("mission_id", "")),
            current_goal_index=int(value.get("current_goal_index", 0)),
            current_phase=str(value.get("current_phase", "idle")),
            active_object_id=str(value.get("active_object_id", "")),
            carried_object_id=str(value.get("carried_object_id", "")),
            plan_revision=int(value.get("plan_revision", 0)),
            control_owner=str(value.get("control_owner", "")),
            last_failure_code=str(value.get("last_failure_code", "none")),
        )
        blackboard.goals = {
            str(key): GoalProgress.from_dict(item)
            for key, item in dict(value.get("goals", {})).items()
        }
        return blackboard

    def progress_for(self, goal_id: str) -> GoalProgress:
        if goal_id not in self.goals:
            self.goals[goal_id] = GoalProgress(goal_id)
        return self.goals[goal_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "goals": {
                key: progress.to_dict() for key, progress in self.goals.items()
            },
        }


class SharedWorldMemory:
    """Object graph facade plus task blackboard and append-only event history."""

    def __init__(self, output_path: Path | None = None) -> None:
        self.objects: dict[str, ObjectRecord] = {}
        self.blackboard = TaskBlackboard()
        self.events: list[dict[str, Any]] = []
        self.output_path = output_path

    @classmethod
    def load(cls, path: Path) -> "SharedWorldMemory":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("world memory schema_version must be 1")
        memory = cls(path)
        for key, item in dict(payload.get("objects", {})).items():
            object_id = str(key)
            record = ObjectRecord.from_dict(item)
            if record.object_id != object_id:
                raise ValueError(
                    f"world memory object key/id mismatch: {object_id}"
                )
            memory.objects[object_id] = record
        memory.blackboard = TaskBlackboard.from_dict(payload.get("blackboard", {}))
        memory.events = list(payload.get("events", []))
        return memory

    def begin_mission(self, mission_id: str) -> None:
        if self.blackboard.mission_id != mission_id:
            self.blackboard = TaskBlackboard(
                mission_id=mission_id,
                plan_revision=self.blackboard.plan_revision + 1,
            )
            self.record_event("mission_started", {"mission_id": mission_id})

    def get_object(self, object_id: str) -> ObjectRecord | None:
        return self.objects.get(object_id)

    def update_object(self, object_id: str, update: Mapping[str, Any]) -> ObjectRecord:
        if not object_id:
            raise ValueError("object_id cannot be empty")
        record = self.objects.setdefault(object_id, ObjectRecord(object_id))
        record.apply({**dict(update), "object_id": object_id})
        self.record_event(
            "object_updated",
            {"object_id": object_id, "revision": record.revision},
        )
        return record

    def apply_skill_updates(self, updates: tuple[dict[str, Any], ...]) -> None:
        for update in updates:
            object_id = str(update.get("object_id", "")).strip()
            if not object_id:
                raise ValueError("skill object update lacks object_id")
            self.update_object(object_id, update)

    def record_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "monotonic_sec": time.monotonic(),
                "type": event_type,
                "payload": dict(payload),
            }
        )
        self.save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "objects": {
                key: record.to_dict() for key, record in self.objects.items()
            },
            "blackboard": self.blackboard.to_dict(),
            "events": list(self.events),
        }

    def save(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(f".{self.output_path.name}.tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.output_path)


__all__ = [
    "GoalProgress",
    "ObjectRecord",
    "SharedWorldMemory",
    "TaskBlackboard",
]
