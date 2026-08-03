"""Event-driven executive that routes between VLN, alignment and VLA skills."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from .control import ControlArbiter, ControlLease
from .memory import GoalProgress, SharedWorldMemory
from .models import (
    ControlResource,
    FailureCode,
    GoalKind,
    Mission,
    MissionResult,
    MissionStatus,
    SkillCommand,
    SkillKind,
    SkillResult,
    SkillStatus,
    TaskGoal,
)
from .skills import SkillRegistry


_SKILL_RESOURCES = {
    SkillKind.NAVIGATE: (ControlResource.BASE,),
    SkillKind.SEARCH_OBJECT: (ControlResource.BASE,),
    SkillKind.APPROACH_ALIGN: (ControlResource.BASE,),
    SkillKind.MANIPULATE: (
        ControlResource.BASE,
        ControlResource.RIGHT_ARM,
        ControlResource.RIGHT_HAND,
    ),
    SkillKind.VERIFY: (),
}

_RETRYABLE_FAILURES = {
    FailureCode.TARGET_NOT_FOUND,
    FailureCode.TARGET_OCCLUDED,
    FailureCode.OBSERVATION_STALE,
    FailureCode.PATH_BLOCKED,
    FailureCode.OUT_OF_REACH,
    FailureCode.BAD_VIEWPOINT,
    FailureCode.BASE_NOT_STOPPED,
    FailureCode.IK_INFEASIBLE,
    FailureCode.GRASP_FAILED,
    FailureCode.OBJECT_SLIPPED,
    FailureCode.VERIFY_FAILED,
    FailureCode.VERIFY_TIMEOUT,
}

_SAFETY_BLOCKS = {
    FailureCode.COLLISION_RISK,
    FailureCode.TF_UNAVAILABLE,
    FailureCode.OBJECT_ID_AMBIGUOUS,
    FailureCode.VLA_UNAVAILABLE,
    FailureCode.CONTROL_LEASE_LOST,
    FailureCode.UNSUPPORTED_SKILL,
}


class DualBrainExecutive:
    """Choose the next skill from task progress and shared object memory."""

    def __init__(
        self,
        skills: SkillRegistry,
        memory: SharedWorldMemory,
        *,
        controls: ControlArbiter | None = None,
        maximum_object_observation_age_sec: float = 2.0,
    ) -> None:
        if maximum_object_observation_age_sec <= 0.0:
            raise ValueError("maximum object observation age must be positive")
        self.skills = skills
        self.memory = memory
        self.controls = controls or ControlArbiter()
        self.maximum_object_observation_age_sec = (
            maximum_object_observation_age_sec
        )
        self._command_sequence = 0

    def execute(self, mission: Mission) -> MissionResult:
        mission.validate()
        self.memory.begin_mission(mission.mission_id)
        board = self.memory.blackboard
        board.current_phase = MissionStatus.RUNNING.value
        self.memory.record_event(
            "mission_running",
            {"mission_id": mission.mission_id, "instruction": mission.instruction},
        )

        for transition in range(mission.maximum_transitions):
            if board.current_goal_index >= len(mission.goals):
                board.current_phase = MissionStatus.SUCCEEDED.value
                self.memory.record_event(
                    "mission_succeeded",
                    {"mission_id": mission.mission_id, "transitions": transition},
                )
                return self._result(
                    MissionStatus.SUCCEEDED,
                    "所有任务目标均已通过执行与验证。",
                    mission,
                )

            goal = mission.goals[board.current_goal_index]
            progress = board.progress_for(goal.goal_id)
            required_carried_object = str(
                goal.metadata.get("requires_carried_object_id", "")
            ).strip()
            if (
                goal.kind is GoalKind.NAVIGATE
                and required_carried_object
                and board.carried_object_id != required_carried_object
            ):
                return self._terminal(
                    mission,
                    MissionStatus.BLOCKED,
                    (
                        f"{goal.goal_id} 要求携带 {required_carried_object}，"
                        f"但共享记忆当前携带对象为 "
                        f"{board.carried_object_id or '空'}；拒绝执行返回导航。"
                    ),
                    FailureCode.OBJECT_SLIPPED,
                )
            board.active_object_id = (
                goal.target_id if goal.kind is GoalKind.INTERACT else ""
            )
            kind = self._select_skill(goal, progress)
            if kind is None:
                self._complete_goal(goal, progress)
                continue
            attempts = progress.attempts.get(kind.value, 0)
            if attempts >= mission.maximum_attempts_per_skill:
                return self._terminal(
                    mission,
                    MissionStatus.BLOCKED,
                    (
                        f"{goal.goal_id} 的 {kind.value} 已达到 "
                        f"{mission.maximum_attempts_per_skill} 次执行上限。"
                    ),
                    FailureCode.RETRY_EXHAUSTED,
                )

            resources = _SKILL_RESOURCES[kind]
            command = self._make_command(mission, goal, kind, resources)
            lease: ControlLease | None = None
            if resources:
                try:
                    lease = self.controls.acquire(command.command_id, resources)
                except RuntimeError as exc:
                    return self._terminal(
                        mission,
                        MissionStatus.BLOCKED,
                        f"无法取得 {kind.value} 控制租约：{exc}",
                        FailureCode.CONTROL_LEASE_LOST,
                    )
            try:
                if lease is not None:
                    command = replace(
                        command,
                        context={
                            **command.context,
                            "control_lease": {
                                "owner": lease.owner,
                                "generation": lease.generation,
                                "resources": [
                                    item.value for item in lease.resources
                                ],
                            },
                        },
                    )
                board.control_owner = command.command_id
                board.current_phase = kind.value
                progress.attempts[kind.value] = attempts + 1
                self.memory.record_event(
                    "skill_started",
                    {
                        "transition": transition,
                        "command": command.to_dict(),
                    },
                )
                result = self._execute_skill(command)
                if lease is not None and not self.controls.validate(lease):
                    result = SkillResult(
                        command.command_id,
                        SkillStatus.BLOCKED,
                        "技能执行期间控制租约丢失。",
                        FailureCode.CONTROL_LEASE_LOST,
                        {"original_result": result.to_dict()},
                    )
                try:
                    self.memory.apply_skill_updates(result.object_updates)
                except (TypeError, ValueError) as exc:
                    result = SkillResult(
                        command.command_id,
                        SkillStatus.FAILED,
                        f"技能返回的对象记忆更新无效：{exc}",
                        FailureCode.ADAPTER_ERROR,
                        {"original_result": result.to_dict()},
                    )
                if (
                    kind is SkillKind.SEARCH_OBJECT
                    and result.status is SkillStatus.SUCCEEDED
                ):
                    observed = self.memory.get_object(goal.target_id)
                    if (
                        observed is None
                        or not observed.observation_is_fresh(
                            self.maximum_object_observation_age_sec,
                            time.monotonic(),
                        )
                    ):
                        result = SkillResult(
                            command.command_id,
                            SkillStatus.FAILED,
                            (
                                "搜索 backend 报告成功，但没有写入"
                                "目标对象的新鲜可见观测。"
                            ),
                            FailureCode.TARGET_NOT_FOUND,
                            {"original_result": result.to_dict()},
                        )
            finally:
                if lease is not None:
                    self.controls.release(lease)
                board.control_owner = ""

            self.memory.record_event(
                "skill_finished",
                {"result": result.to_dict()},
            )
            if result.status is SkillStatus.SUCCEEDED:
                self._accept_success(goal, progress, kind)
                continue
            terminal = self._handle_failure(goal, progress, kind, result)
            if terminal is not None:
                status, message, failure = terminal
                return self._terminal(mission, status, message, failure)

        return self._terminal(
            mission,
            MissionStatus.FAILED,
            f"超过 maximum_transitions={mission.maximum_transitions}。",
            FailureCode.RETRY_EXHAUSTED,
        )

    def _select_skill(
        self,
        goal: TaskGoal,
        progress: GoalProgress,
    ) -> SkillKind | None:
        if goal.kind is GoalKind.NAVIGATE:
            return None if progress.verified else SkillKind.NAVIGATE
        if goal.region_hint and not progress.region_reached:
            return SkillKind.NAVIGATE
        record = self.memory.get_object(goal.target_id)
        if (
            record is None
            or not record.observation_is_fresh(
                self.maximum_object_observation_age_sec,
                time.monotonic(),
            )
        ):
            return SkillKind.SEARCH_OBJECT
        if not progress.aligned:
            return SkillKind.APPROACH_ALIGN
        if not progress.manipulated:
            return SkillKind.MANIPULATE
        if not progress.verified:
            return SkillKind.VERIFY
        return None

    def _make_command(
        self,
        mission: Mission,
        goal: TaskGoal,
        kind: SkillKind,
        resources: tuple[ControlResource, ...],
    ) -> SkillCommand:
        self._command_sequence += 1
        if kind is SkillKind.NAVIGATE and goal.region_hint:
            instruction = goal.region_hint
        elif kind is SkillKind.SEARCH_OBJECT:
            instruction = f"搜索并确认目标 {goal.target_id}：{goal.instruction}"
        elif kind is SkillKind.APPROACH_ALIGN:
            instruction = f"对准目标 {goal.target_id}：{goal.instruction}"
        elif kind is SkillKind.VERIFY and goal.success_condition:
            instruction = goal.success_condition
        else:
            instruction = goal.instruction
        return SkillCommand(
            command_id=f"{mission.mission_id}:{self._command_sequence:04d}",
            mission_id=mission.mission_id,
            goal_id=goal.goal_id,
            kind=kind,
            instruction=instruction,
            target_id=goal.target_id,
            action=goal.action,
            payload_object_id=goal.payload_object_id,
            required_resources=resources,
            context={
                "mission_instruction": mission.instruction,
                "goal_metadata": goal.metadata,
                "plan_revision": self.memory.blackboard.plan_revision,
            },
        )

    def _execute_skill(self, command: SkillCommand) -> SkillResult:
        try:
            result = self.skills.resolve(command.kind).execute(
                command,
                self.memory,
            )
        except Exception as exc:  # keep an integration exception out of the executive
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                f"{command.kind.value} adapter 异常：{exc}",
                FailureCode.ADAPTER_ERROR,
                {"exception": type(exc).__name__},
            )
        if result.command_id != command.command_id:
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                "技能结果 command_id 与请求不一致。",
                FailureCode.ADAPTER_ERROR,
                {"returned_command_id": result.command_id},
            )
        if (
            result.status is not SkillStatus.SUCCEEDED
            and result.failure_code is FailureCode.NONE
        ):
            return SkillResult(
                result.command_id,
                result.status,
                result.message,
                FailureCode.ADAPTER_ERROR,
                result.details,
                result.object_updates,
            )
        return result

    def _accept_success(
        self,
        goal: TaskGoal,
        progress: GoalProgress,
        kind: SkillKind,
    ) -> None:
        if goal.kind is GoalKind.NAVIGATE:
            progress.region_reached = True
            progress.verified = True
        elif kind is SkillKind.NAVIGATE:
            progress.region_reached = True
        elif kind is SkillKind.APPROACH_ALIGN:
            progress.aligned = True
            self.memory.update_object(
                goal.target_id,
                {
                    "reachable": True,
                    "last_action": kind.value,
                    "last_result": SkillStatus.SUCCEEDED.value,
                    "last_failure_code": FailureCode.NONE.value,
                },
            )
        elif kind is SkillKind.MANIPULATE:
            progress.manipulated = True
            if goal.action == "pick":
                self.memory.update_object(
                    goal.target_id,
                    {
                        "attachment_state": "candidate_held",
                        "parent_frame": "right_hand",
                        "last_action": goal.action,
                        "last_result": SkillStatus.SUCCEEDED.value,
                        "last_failure_code": FailureCode.NONE.value,
                    },
                )
            elif goal.action == "place" and goal.payload_object_id:
                self.memory.update_object(
                    goal.payload_object_id,
                    {
                        "attachment_state": "candidate_placed",
                        "last_action": goal.action,
                        "last_result": SkillStatus.SUCCEEDED.value,
                        "last_failure_code": FailureCode.NONE.value,
                    },
                )
        elif kind is SkillKind.VERIFY:
            progress.verified = True
            if goal.action == "pick":
                self.memory.blackboard.carried_object_id = goal.target_id
                self.memory.update_object(
                    goal.target_id,
                    {
                        "attachment_state": "held",
                        "parent_frame": "right_hand",
                        "last_action": "verify_pick",
                        "last_result": SkillStatus.SUCCEEDED.value,
                    },
                )
            elif goal.action == "place" and goal.payload_object_id:
                self.memory.blackboard.carried_object_id = ""
                self.memory.update_object(
                    goal.payload_object_id,
                    {
                        "attachment_state": "world",
                        "parent_frame": "map",
                        "last_action": "verify_place",
                        "last_result": SkillStatus.SUCCEEDED.value,
                    },
                )
        progress.last_failure_code = FailureCode.NONE.value
        self.memory.record_event(
            "skill_success_applied",
            {"goal_id": goal.goal_id, "skill": kind.value},
        )

    def _handle_failure(
        self,
        goal: TaskGoal,
        progress: GoalProgress,
        kind: SkillKind,
        result: SkillResult,
    ) -> tuple[MissionStatus, str, FailureCode] | None:
        failure = result.failure_code
        progress.last_failure_code = failure.value
        self.memory.blackboard.last_failure_code = failure.value
        record = self.memory.get_object(goal.target_id) if goal.target_id else None
        if record is not None:
            self.memory.update_object(
                goal.target_id,
                {
                    "last_action": kind.value,
                    "last_result": result.status.value,
                    "last_failure_code": failure.value,
                },
            )

        if failure in {
            FailureCode.TARGET_NOT_FOUND,
            FailureCode.TARGET_OCCLUDED,
            FailureCode.OBSERVATION_STALE,
        }:
            if record is not None:
                self.memory.update_object(
                    goal.target_id,
                    {"visible": False, "reachable": None},
                )
            progress.aligned = False
        elif failure is FailureCode.PATH_BLOCKED:
            progress.region_reached = False
        elif failure in {
            FailureCode.OUT_OF_REACH,
            FailureCode.BAD_VIEWPOINT,
            FailureCode.BASE_NOT_STOPPED,
            FailureCode.IK_INFEASIBLE,
        }:
            progress.aligned = False
            if record is not None:
                self.memory.update_object(goal.target_id, {"reachable": False})
        elif failure is FailureCode.OBJECT_SLIPPED:
            progress.aligned = False
            progress.manipulated = False
            self.memory.blackboard.carried_object_id = ""
            slipped_id = goal.payload_object_id or goal.target_id
            if slipped_id:
                self.memory.update_object(
                    slipped_id,
                    {
                        "attachment_state": "world",
                        "parent_frame": "map",
                        "reachable": False,
                    },
                )
        elif failure in {
            FailureCode.GRASP_FAILED,
            FailureCode.VERIFY_FAILED,
            FailureCode.VERIFY_TIMEOUT,
        }:
            progress.manipulated = failure is not FailureCode.GRASP_FAILED

        if failure in _RETRYABLE_FAILURES:
            self.memory.blackboard.plan_revision += 1
            self.memory.record_event(
                "dynamic_replan",
                {
                    "goal_id": goal.goal_id,
                    "failed_skill": kind.value,
                    "failure_code": failure.value,
                    "next_skill": self._select_skill(goal, progress).value,
                    "plan_revision": self.memory.blackboard.plan_revision,
                },
            )
            return None
        status = (
            MissionStatus.BLOCKED
            if result.status is SkillStatus.BLOCKED or failure in _SAFETY_BLOCKS
            else MissionStatus.FAILED
        )
        return status, result.message, failure

    def _complete_goal(
        self,
        goal: TaskGoal,
        progress: GoalProgress,
    ) -> None:
        progress.verified = True
        board = self.memory.blackboard
        self.memory.record_event(
            "goal_completed",
            {"goal_id": goal.goal_id, "goal_index": board.current_goal_index},
        )
        board.current_goal_index += 1
        board.active_object_id = ""

    def _terminal(
        self,
        mission: Mission,
        status: MissionStatus,
        message: str,
        failure: FailureCode,
    ) -> MissionResult:
        board = self.memory.blackboard
        board.current_phase = status.value
        board.last_failure_code = failure.value
        self.memory.record_event(
            f"mission_{status.value}",
            {
                "mission_id": mission.mission_id,
                "message": message,
                "failure_code": failure.value,
            },
        )
        return self._result(status, message, mission)

    def _result(
        self,
        status: MissionStatus,
        message: str,
        mission: Mission,
    ) -> MissionResult:
        return MissionResult(
            status,
            message,
            mission,
            tuple(self.memory.events),
            self.memory.snapshot(),
        )


__all__ = ["DualBrainExecutive"]
