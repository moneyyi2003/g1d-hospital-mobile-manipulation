"""Compatibility bridge to the existing, validated VLN and VLA adapters."""

from __future__ import annotations

from dataclasses import dataclass

from g1d_agent.adapters import StepAdapter, WarehouseVlnAdapter
from g1d_agent.models import (
    Capability,
    StepKind,
    StepStatus,
    TaskStep,
)

from .memory import SharedWorldMemory
from .models import (
    FailureCode,
    SkillCommand,
    SkillKind,
    SkillResult,
    SkillStatus,
)


@dataclass
class FormalWarehouseVlnAdapter(WarehouseVlnAdapter):
    """Warehouse compatibility adapter that forbids the bootstrap entrypoint."""

    def command_for(self, step: TaskStep) -> list[str]:
        if step.kind is not StepKind.SEMANTIC_NAVIGATION:
            raise ValueError(
                "Warehouse 当前只有正式区域语义导航；"
                "物体停靠需 live 对齐 backend"
            )
        command = [
            str(self.workspace / "mobilemanibench.sh"),
            "warehouse-vln-formal",
            "--command",
            step.instruction,
        ]
        if self.headless:
            command.append("--headless")
        if self.test:
            command.append("--test")
        if self.no_camera:
            command.append("--no-camera")
        return command


def _failure_from_result(message: str, details: dict) -> FailureCode:
    explicit = str(details.get("failure_code", "")).casefold()
    if explicit:
        try:
            return FailureCode(explicit)
        except ValueError:
            pass
    readiness = details.get("readiness", {})
    action = (
        str(readiness.get("action", "")).casefold()
        if isinstance(readiness, dict)
        else ""
    )
    readiness_failures = {
        "reacquire_object": FailureCode.TARGET_NOT_FOUND,
        "wait_for_stable_pose": FailureCode.OBSERVATION_STALE,
        "stop_base": FailureCode.BASE_NOT_STOPPED,
        "move_closer": FailureCode.OUT_OF_REACH,
        "move_away": FailureCode.OUT_OF_REACH,
        "realign_base": FailureCode.BAD_VIEWPOINT,
        "reposition_for_reachability": FailureCode.IK_INFEASIBLE,
        "block_collision": FailureCode.COLLISION_RISK,
    }
    if action in readiness_failures:
        return readiness_failures[action]
    text = message.casefold()
    mappings = (
        (("collision", "碰撞"), FailureCode.COLLISION_RISK),
        (("不可达", "out_of_reach"), FailureCode.OUT_OF_REACH),
        (("不可见", "缺少实时", "reacquire"), FailureCode.TARGET_NOT_FOUND),
        (("path", "路径"), FailureCode.PATH_BLOCKED),
        (("尚未", "未接入", "unavailable"), FailureCode.VLA_UNAVAILABLE),
    )
    for needles, code in mappings:
        if any(item in text for item in needles):
            return code
    return FailureCode.ADAPTER_ERROR


@dataclass(frozen=True)
class LegacyVlnSkillExecutor:
    """Reuse existing semantic-map navigation and Hospital precision docking."""

    adapter: StepAdapter

    def execute(
        self,
        command: SkillCommand,
        memory: SharedWorldMemory,
    ) -> SkillResult:
        if command.kind is SkillKind.NAVIGATE:
            old_kind = StepKind.SEMANTIC_NAVIGATION
        elif command.kind is SkillKind.APPROACH_ALIGN:
            old_kind = StepKind.PREGRASP_DOCKING
        else:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                f"现有 VLN 不支持 {command.kind.value}。",
                FailureCode.UNSUPPORTED_SKILL,
            )
        step = TaskStep(
            step_id=command.command_id,
            capability=Capability.VLN,
            kind=old_kind,
            instruction=command.instruction,
            success_condition=(
                "现有语义地点导航报告到达"
                if old_kind is StepKind.SEMANTIC_NAVIGATION
                else "现有物体相对停靠报告到达"
            ),
            metadata={
                "skill": command.action,
                "target_id": command.target_id,
                "dual_brain_command": command.to_dict(),
            },
        )
        result = self.adapter.execute(
            step,
            {
                "world_memory": memory.snapshot(),
                "control_lease": command.context.get("control_lease"),
            },
        )
        status = {
            StepStatus.SUCCEEDED: SkillStatus.SUCCEEDED,
            StepStatus.BLOCKED: SkillStatus.BLOCKED,
        }.get(result.status, SkillStatus.FAILED)
        failure = (
            FailureCode.NONE
            if status is SkillStatus.SUCCEEDED
            else _failure_from_result(result.message, result.details)
        )
        return SkillResult(
            command.command_id,
            status,
            result.message,
            failure,
            {"legacy_step": step.to_dict(), "legacy_result": result.to_dict()},
        )


@dataclass(frozen=True)
class LegacyVlaSkillExecutor:
    """Keep the current VLA plugin/readiness contract as the manipulation slot."""

    adapter: StepAdapter

    def execute(
        self,
        command: SkillCommand,
        memory: SharedWorldMemory,
    ) -> SkillResult:
        if command.kind is not SkillKind.MANIPULATE:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                f"VLA adapter 不支持 {command.kind.value}。",
                FailureCode.UNSUPPORTED_SKILL,
            )
        step = TaskStep(
            step_id=command.command_id,
            capability=Capability.VLA,
            kind=StepKind.MANIPULATION,
            instruction=command.instruction,
            success_condition="VLA 和独立验证器均报告成功",
            metadata={
                "skill": command.action,
                "target_id": command.target_id,
                "payload_object_id": command.payload_object_id,
                "dual_brain_command": command.to_dict(),
            },
        )
        result = self.adapter.execute(
            step,
            {
                "world_memory": memory.snapshot(),
                "control_lease": command.context.get("control_lease"),
            },
        )
        status = {
            StepStatus.SUCCEEDED: SkillStatus.SUCCEEDED,
            StepStatus.BLOCKED: SkillStatus.BLOCKED,
        }.get(result.status, SkillStatus.FAILED)
        failure = (
            FailureCode.NONE
            if status is SkillStatus.SUCCEEDED
            else _failure_from_result(result.message, result.details)
        )
        return SkillResult(
            command.command_id,
            status,
            result.message,
            failure,
            {"legacy_step": step.to_dict(), "legacy_result": result.to_dict()},
        )


__all__ = [
    "FormalWarehouseVlnAdapter",
    "LegacyVlaSkillExecutor",
    "LegacyVlnSkillExecutor",
]
