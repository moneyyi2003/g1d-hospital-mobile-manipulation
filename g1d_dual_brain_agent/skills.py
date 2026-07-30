"""Skill executor contracts and fail-closed integration adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Protocol

from .memory import SharedWorldMemory
from .models import (
    FailureCode,
    SkillCommand,
    SkillKind,
    SkillResult,
    SkillStatus,
)


class SkillExecutor(Protocol):
    def execute(
        self,
        command: SkillCommand,
        memory: SharedWorldMemory,
    ) -> SkillResult:
        """Execute exactly one structured skill."""


@dataclass
class SkillRegistry:
    executors: dict[SkillKind, SkillExecutor] = field(default_factory=dict)

    def register(self, kind: SkillKind, executor: SkillExecutor) -> None:
        self.executors[kind] = executor

    def resolve(self, kind: SkillKind) -> SkillExecutor:
        return self.executors.get(kind, UnavailableSkillExecutor(kind))


@dataclass(frozen=True)
class UnavailableSkillExecutor:
    kind: SkillKind
    reason: str = ""

    def execute(
        self,
        command: SkillCommand,
        memory: SharedWorldMemory,
    ) -> SkillResult:
        del memory
        reason = self.reason or f"{self.kind.value} backend 尚未接入"
        code = (
            FailureCode.VLA_UNAVAILABLE
            if self.kind is SkillKind.MANIPULATE
            else FailureCode.UNSUPPORTED_SKILL
        )
        return SkillResult(
            command.command_id,
            SkillStatus.BLOCKED,
            f"{reason}；任务不会被误报为成功。",
            code,
            {
                "required_interface": (
                    "g1d_dual_brain_agent.SkillExecutor"
                ),
                "skill": self.kind.value,
            },
        )


@dataclass(frozen=True)
class CallableSkillExecutor:
    """Small adapter useful for tests and in-process integrations."""

    callback: Callable[[SkillCommand, SharedWorldMemory], SkillResult]

    def execute(
        self,
        command: SkillCommand,
        memory: SharedWorldMemory,
    ) -> SkillResult:
        return self.callback(command, memory)


_BACKEND_METHODS = {
    SkillKind.SEARCH_OBJECT: "search_object",
    SkillKind.APPROACH_ALIGN: "approach_and_align",
    SkillKind.VERIFY: "verify_task",
}


@dataclass(frozen=True)
class BackendMethodSkillExecutor:
    """Adapt optional VLA-integration methods without importing its model code."""

    backend: Any
    kind: SkillKind

    def execute(
        self,
        command: SkillCommand,
        memory: SharedWorldMemory,
    ) -> SkillResult:
        method_name = _BACKEND_METHODS.get(self.kind)
        method = getattr(self.backend, method_name or "", None)
        if not callable(method):
            return UnavailableSkillExecutor(self.kind).execute(command, memory)
        request = {
            "schema_version": 1,
            "environment": "isaac_sim",
            "robot": "g1_d",
            "skill": command.to_dict(),
            "world_memory": memory.snapshot(),
        }
        try:
            raw = dict(method(request))
        except Exception as exc:  # integration boundary must return structured failure
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                f"{method_name} backend 异常：{exc}",
                FailureCode.ADAPTER_ERROR,
                {"method": method_name, "exception": type(exc).__name__},
            )
        status_text = str(raw.get("status", "")).casefold()
        success = status_text == "succeeded" and raw.get("success") is True
        status = (
            SkillStatus.SUCCEEDED
            if success
            else SkillStatus.BLOCKED
            if status_text == "blocked"
            else SkillStatus.FAILED
        )
        try:
            failure = (
                FailureCode.NONE
                if success
                else FailureCode(
                    str(raw.get("failure_code", FailureCode.ADAPTER_ERROR.value))
                )
            )
        except ValueError:
            failure = FailureCode.ADAPTER_ERROR
        updates = raw.get("object_updates", [])
        if not isinstance(updates, list) or any(
            not isinstance(item, Mapping) for item in updates
        ):
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                f"{method_name} 返回了无效 object_updates。",
                FailureCode.ADAPTER_ERROR,
                {"result": raw},
            )
        normalized_updates = []
        received_at = time.monotonic()
        for item in updates:
            update = dict(item)
            if (
                self.kind is SkillKind.SEARCH_OBJECT
                and success
                and update.get("visible") is True
                and "last_seen_monotonic_sec" not in update
            ):
                update["last_seen_monotonic_sec"] = received_at
            normalized_updates.append(update)
        return SkillResult(
            command.command_id,
            status,
            str(raw.get("message", f"{method_name} returned {status_text}")),
            failure,
            {"method": method_name, "result": raw},
            tuple(normalized_updates),
        )


__all__ = [
    "BackendMethodSkillExecutor",
    "CallableSkillExecutor",
    "SkillExecutor",
    "SkillRegistry",
    "UnavailableSkillExecutor",
]
