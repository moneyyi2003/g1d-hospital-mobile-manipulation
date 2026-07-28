"""Sequential, fail-closed execution of VLN/VLA mission plans."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import StepAdapter
from .models import (
    Capability,
    MissionPlan,
    MissionResult,
    MissionStatus,
    StepResult,
    StepStatus,
)
from .router import RuleTaskPlanner


@dataclass
class G1DTaskAgent:
    planner: RuleTaskPlanner
    vln: StepAdapter
    vla: StepAdapter

    def plan(self, instruction: str) -> MissionPlan:
        return self.planner.plan(instruction)

    def execute(self, plan: MissionPlan) -> MissionResult:
        results: list[StepResult] = []
        for index, step in enumerate(plan.steps):
            adapter = self.vln if step.capability is Capability.VLN else self.vla
            context = {
                "plan": plan.to_dict(),
                "previous_steps": [item.to_dict() for item in results],
            }
            try:
                result = adapter.execute(step, context)
            except Exception as exc:
                result = StepResult(
                    step.step_id,
                    StepStatus.FAILED,
                    f"{step.capability.value.upper()} adapter 异常：{exc}",
                    {"error_type": type(exc).__name__},
                )
            results.append(result)
            if result.status is StepStatus.SUCCEEDED:
                continue
            for skipped in plan.steps[index + 1 :]:
                results.append(
                    StepResult(
                        skipped.step_id,
                        StepStatus.SKIPPED,
                        f"前置步骤 {step.step_id} 未成功，未执行本步骤。",
                    )
                )
            if result.status is StepStatus.BLOCKED:
                return MissionResult(
                    MissionStatus.BLOCKED,
                    result.message,
                    plan,
                    tuple(results),
                )
            return MissionResult(
                MissionStatus.FAILED,
                result.message,
                plan,
                tuple(results),
            )
        return MissionResult(
            MissionStatus.SUCCEEDED,
            "全部 VLN/VLA 步骤均通过各自成功判据。",
            plan,
            tuple(results),
        )


__all__ = ["G1DTaskAgent"]
