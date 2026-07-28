"""Conservative task decomposition into VLN and VLA capabilities."""

from __future__ import annotations

from .models import Capability, MissionPlan, StepKind, TaskStep


_MANIPULATION_TERMS = (
    "抓",
    "拿",
    "取",
    "捡",
    "拾",
    "放",
    "递",
    "倒",
    "打开",
    "关闭",
    "开门",
    "关门",
    "按",
    "推",
    "拉",
    "抬",
    "送",
    "给我",
    "pick",
    "grasp",
    "grab",
    "place",
    "put",
    "hand over",
    "open",
    "close",
    "push",
    "pull",
    "lift",
    "bring",
    "pour",
    "press",
)

_NAVIGATION_TERMS = (
    "带我",
    "带你",
    "前往",
    "导航",
    "移动到",
    "走到",
    "去",
    "到",
    "靠近",
    "停到",
    "停在",
    "navigate",
    "go to",
    "move to",
    "drive to",
    "take me",
)

_LOCAL_TERMS = (
    "眼前",
    "面前",
    "手边",
    "旁边这个",
    "当前这个",
    "已经在",
    "within reach",
    "in front of me",
    "nearby object",
    "this object",
)

_SKILL_TERMS = (
    (
        "pick",
        ("抓", "拿", "取", "捡", "拾", "抬", "pick", "grasp", "grab", "lift"),
    ),
    ("place", ("放", "place", "put")),
    ("hand_over", ("递", "hand over")),
    ("pour", ("倒", "pour")),
    ("open", ("打开", "开门", "open")),
    ("close", ("关闭", "关门", "close")),
    ("push", ("推", "push")),
    ("pull", ("拉", "pull")),
    ("press", ("按", "press")),
    ("deliver", ("送", "给我", "bring")),
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _infer_skill(text: str) -> str:
    for skill, terms in _SKILL_TERMS:
        if _contains_any(text, terms):
            return skill
    return "manipulation"


class RuleTaskPlanner:
    """Produce an auditable capability plan without inventing coordinates.

    The planner only decides which subsystem owns each phase. Exact place IDs,
    object poses, paths, observations, and joint commands stay inside the
    existing VLN and future VLA adapters.
    """

    name = "rule_task_planner_v1"

    def plan(self, instruction: str) -> MissionPlan:
        command = instruction.strip()
        if not command:
            raise ValueError("任务指令不能为空")
        normalized = command.casefold()
        manipulation = _contains_any(normalized, _MANIPULATION_TERMS)
        navigation = _contains_any(normalized, _NAVIGATION_TERMS)
        explicitly_local = _contains_any(normalized, _LOCAL_TERMS)
        skill = _infer_skill(normalized)

        if manipulation and explicitly_local and not navigation:
            return MissionPlan(
                instruction=command,
                route="vla",
                reason="指令要求操作已在机器人可达范围内的物体。",
                planner=self.name,
                steps=(self._manipulation_step(command, "1", skill),),
            )

        if manipulation:
            return MissionPlan(
                instruction=command,
                route="vln_then_vla",
                reason=(
                    "移动操作任务需要先由现有 VLN 到达物体预抓取停靠位，"
                    "再由 VLA 执行近距离操作。"
                ),
                planner=self.name,
                steps=(
                    TaskStep(
                        step_id="1",
                        capability=Capability.VLN,
                        kind=StepKind.PREGRASP_DOCKING,
                        instruction=command,
                        success_condition=(
                            "底盘到达经 occupancy/footprint 验证的预抓取停靠位并停止"
                        ),
                        metadata={
                            "vln_runner": "hospital-object-docking",
                            "skill": skill,
                        },
                    ),
                    self._manipulation_step(command, "2", skill),
                ),
            )

        if navigation:
            return MissionPlan(
                instruction=command,
                route="vln",
                reason="指令只要求改变机器人所在地点，不包含物体操作。",
                planner=self.name,
                steps=(
                    TaskStep(
                        step_id="1",
                        capability=Capability.VLN,
                        kind=StepKind.SEMANTIC_NAVIGATION,
                        instruction=command,
                        success_condition=(
                            "现有 Hospital VLN 从审核地点库选出目标并报告到达"
                        ),
                        metadata={"vln_runner": "hospital-vln"},
                    ),
                ),
            )

        raise ValueError(
            "无法安全判断任务需要 VLN 还是 VLA；请明确说明要前往哪里或操作什么物体"
        )

    @staticmethod
    def _manipulation_step(command: str, step_id: str, skill: str) -> TaskStep:
        return TaskStep(
            step_id=step_id,
            capability=Capability.VLA,
            kind=StepKind.MANIPULATION,
            instruction=command,
            success_condition=(
                "VLA backend 返回结构化成功，且 Isaac/真机安全层验证任务判据"
            ),
            metadata={
                "vla_backend": "external_plugin",
                "skill": skill,
            },
        )


__all__ = ["RuleTaskPlanner"]
