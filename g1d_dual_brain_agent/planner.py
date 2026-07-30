"""Mission loading and conservative compatibility compilation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from g1d_agent.router import RuleTaskPlanner

from .models import GoalKind, Mission, TaskGoal


def load_mission(path: Path) -> Mission:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Mission.from_dict(payload)


def compile_command(
    instruction: str,
    *,
    object_id: str = "",
    region_hint: str = "",
    mission_id: str = "",
) -> Mission:
    """Reuse the old auditable language router, then emit v2 task goals.

    Object identity is deliberately not guessed from free text. Interaction
    commands need an object ID from the reviewed scene memory/catalog.
    """

    old_plan = RuleTaskPlanner().plan(instruction)
    if not mission_id:
        digest = hashlib.sha256(
            f"{instruction}\0{object_id}\0{region_hint}".encode("utf-8")
        ).hexdigest()[:12]
        mission_id = f"command-{digest}"
    if old_plan.route == "vln":
        goals = (
            TaskGoal(
                goal_id="navigate-1",
                kind=GoalKind.NAVIGATE,
                instruction=instruction.strip(),
                success_condition=old_plan.steps[0].success_condition,
                metadata={"compiled_from": old_plan.to_dict()},
            ),
        )
    else:
        if not object_id.strip():
            raise ValueError(
                "操作任务必须用 --object-id 或 mission JSON "
                "指定审核对象 ID；Agent 不会从自由文本猜坐标或对象身份"
            )
        manipulation = old_plan.steps[-1]
        goals = (
            TaskGoal(
                goal_id="interact-1",
                kind=GoalKind.INTERACT,
                instruction=instruction.strip(),
                target_id=object_id.strip(),
                action=str(manipulation.metadata.get("skill", "manipulation")),
                region_hint=region_hint.strip(),
                success_condition=manipulation.success_condition,
                metadata={"compiled_from": old_plan.to_dict()},
            ),
        )
    return Mission(mission_id, instruction.strip(), goals)


__all__ = ["compile_command", "load_mission"]
