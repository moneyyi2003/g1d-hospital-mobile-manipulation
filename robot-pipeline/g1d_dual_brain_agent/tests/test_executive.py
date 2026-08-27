from __future__ import annotations

from collections import defaultdict
import time
import unittest

from g1d_dual_brain_agent.executive import DualBrainExecutive
from g1d_dual_brain_agent.memory import SharedWorldMemory
from g1d_dual_brain_agent.models import (
    FailureCode,
    GoalKind,
    Mission,
    MissionStatus,
    SkillKind,
    SkillResult,
    SkillStatus,
    TaskGoal,
)
from g1d_dual_brain_agent.skills import CallableSkillExecutor, SkillRegistry


class ScriptedSkills:
    def __init__(self) -> None:
        self.calls = []
        self.counts = defaultdict(int)
        self.first_manipulation_out_of_reach = False
        self.first_physical_manipulation_failure = False
        self.align_collision = False

    def executor(self, kind):
        return CallableSkillExecutor(
            lambda command, memory: self.execute(kind, command, memory)
        )

    def execute(self, kind, command, memory):
        self.calls.append(command)
        self.counts[kind] += 1
        if kind is SkillKind.SEARCH_OBJECT:
            return SkillResult(
                command.command_id,
                SkillStatus.SUCCEEDED,
                "target visible",
                object_updates=(
                    {
                        "object_id": command.target_id,
                        "visible": True,
                        "detection_confidence": 0.93,
                        "last_seen_monotonic_sec": time.monotonic(),
                        "observation_source": "fake_rgb_tracker",
                    },
                ),
            )
        if kind is SkillKind.APPROACH_ALIGN and self.align_collision:
            return SkillResult(
                command.command_id,
                SkillStatus.BLOCKED,
                "collision envelope occupied",
                FailureCode.COLLISION_RISK,
            )
        if (
            kind is SkillKind.MANIPULATE
            and self.first_manipulation_out_of_reach
            and self.counts[kind] == 1
        ):
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                "object left workspace",
                FailureCode.OUT_OF_REACH,
            )
        if (
            kind is SkillKind.MANIPULATE
            and self.first_physical_manipulation_failure
        ):
            return SkillResult(
                command.command_id,
                SkillStatus.FAILED,
                "physical grasp failed",
                FailureCode.GRASP_FAILED,
                {"manipulation": {"physical_execution": True}},
            )
        return SkillResult(
            command.command_id,
            SkillStatus.SUCCEEDED,
            f"{kind.value} succeeded",
        )


def make_registry(scripted: ScriptedSkills) -> SkillRegistry:
    registry = SkillRegistry()
    for kind in SkillKind:
        registry.register(kind, scripted.executor(kind))
    return registry


def interaction_mission(maximum_attempts: int = 3) -> Mission:
    return Mission(
        "mission-pick",
        "去厨房拿起红色杯子",
        (
            TaskGoal(
                "pick-cup",
                GoalKind.INTERACT,
                "拿起红色杯子",
                target_id="cup-03",
                action="pick",
                region_hint="请导航到已审核的厨房操作区",
                success_condition="杯子抬起并稳定保持",
            ),
        ),
        maximum_attempts_per_skill=maximum_attempts,
    )


class DualBrainExecutiveTest(unittest.TestCase):
    def test_full_event_driven_chain(self) -> None:
        scripted = ScriptedSkills()
        memory = SharedWorldMemory()

        result = DualBrainExecutive(
            make_registry(scripted),
            memory,
        ).execute(interaction_mission())

        self.assertEqual(result.status, MissionStatus.SUCCEEDED)
        self.assertEqual(
            [call.kind for call in scripted.calls],
            [
                SkillKind.NAVIGATE,
                SkillKind.SEARCH_OBJECT,
                SkillKind.APPROACH_ALIGN,
                SkillKind.MANIPULATE,
                SkillKind.VERIFY,
            ],
        )
        self.assertEqual(memory.blackboard.carried_object_id, "cup-03")
        self.assertEqual(memory.get_object("cup-03").attachment_state, "held")
        self.assertEqual(memory.blackboard.control_owner, "")

    def test_out_of_reach_replans_to_alignment(self) -> None:
        scripted = ScriptedSkills()
        scripted.first_manipulation_out_of_reach = True

        result = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        ).execute(interaction_mission())

        self.assertEqual(result.status, MissionStatus.SUCCEEDED)
        self.assertEqual(
            [call.kind for call in scripted.calls],
            [
                SkillKind.NAVIGATE,
                SkillKind.SEARCH_OBJECT,
                SkillKind.APPROACH_ALIGN,
                SkillKind.MANIPULATE,
                SkillKind.APPROACH_ALIGN,
                SkillKind.MANIPULATE,
                SkillKind.VERIFY,
            ],
        )
        replans = [
            item for item in result.events if item["type"] == "dynamic_replan"
        ]
        self.assertEqual(replans[-1]["payload"]["next_skill"], "approach_and_align")

    def test_collision_blocks_without_retrying(self) -> None:
        scripted = ScriptedSkills()
        scripted.align_collision = True

        result = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        ).execute(interaction_mission())

        self.assertEqual(result.status, MissionStatus.BLOCKED)
        self.assertEqual(
            result.memory["blackboard"]["last_failure_code"],
            "collision_risk",
        )
        self.assertEqual(scripted.counts[SkillKind.APPROACH_ALIGN], 1)
        self.assertEqual(scripted.counts[SkillKind.MANIPULATE], 0)

    def test_physical_manipulation_failure_does_not_teleport_retry(self) -> None:
        scripted = ScriptedSkills()
        scripted.first_physical_manipulation_failure = True

        result = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        ).execute(interaction_mission())

        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertEqual(scripted.counts[SkillKind.MANIPULATE], 1)
        self.assertFalse(
            any(event["type"] == "dynamic_replan" for event in result.events)
        )

    def test_navigation_goal_does_not_require_object_or_vla(self) -> None:
        scripted = ScriptedSkills()
        mission = Mission(
            "mission-nav",
            "请带我到客厅沙发旁",
            (
                TaskGoal(
                    "nav-sofa",
                    GoalKind.NAVIGATE,
                    "请带我到客厅沙发旁",
                ),
            ),
        )

        result = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        ).execute(mission)

        self.assertEqual(result.status, MissionStatus.SUCCEEDED)
        self.assertEqual(
            [call.kind for call in scripted.calls],
            [SkillKind.NAVIGATE],
        )

    def test_missing_search_backend_fails_closed(self) -> None:
        scripted = ScriptedSkills()
        registry = make_registry(scripted)
        registry.executors.pop(SkillKind.SEARCH_OBJECT)

        result = DualBrainExecutive(
            registry,
            SharedWorldMemory(),
        ).execute(interaction_mission())

        self.assertEqual(result.status, MissionStatus.BLOCKED)
        self.assertEqual(
            result.memory["blackboard"]["last_failure_code"],
            FailureCode.UNSUPPORTED_SKILL.value,
        )
        self.assertEqual(scripted.counts[SkillKind.MANIPULATE], 0)

    def test_control_acquisition_failure_is_structured_block(self) -> None:
        scripted = ScriptedSkills()
        executive = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        )
        executive.controls.emergency_stop()
        mission = Mission(
            "mission-estop",
            "请带我到客厅沙发旁",
            (
                TaskGoal(
                    "nav-sofa",
                    GoalKind.NAVIGATE,
                    "请带我到客厅沙发旁",
                ),
            ),
        )

        result = executive.execute(mission)

        self.assertEqual(result.status, MissionStatus.BLOCKED)
        self.assertEqual(
            result.memory["blackboard"]["last_failure_code"],
            FailureCode.CONTROL_LEASE_LOST.value,
        )
        self.assertEqual(scripted.calls, [])

    def test_return_navigation_requires_verified_carried_object(self) -> None:
        scripted = ScriptedSkills()
        mission = Mission(
            "mission-return-guard",
            "回到客厅",
            (
                TaskGoal(
                    "return-home",
                    GoalKind.NAVIGATE,
                    "living_room_sofa",
                    payload_object_id="cup-03",
                    metadata={"requires_carried_object_id": "cup-03"},
                ),
            ),
        )

        result = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        ).execute(mission)

        self.assertEqual(result.status, MissionStatus.BLOCKED)
        self.assertEqual(
            result.memory["blackboard"]["last_failure_code"],
            FailureCode.OBJECT_SLIPPED.value,
        )
        self.assertEqual(scripted.calls, [])

    def test_pick_verification_unlocks_return_navigation(self) -> None:
        scripted = ScriptedSkills()
        mission = Mission(
            "mission-pick-return",
            "拿杯子再回到客厅",
            (
                TaskGoal(
                    "pick-cup",
                    GoalKind.INTERACT,
                    "拿杯子",
                    target_id="cup-03",
                    action="pick",
                    success_condition="杯子已抬升并保持",
                ),
                TaskGoal(
                    "return-home",
                    GoalKind.NAVIGATE,
                    "living_room_sofa",
                    payload_object_id="cup-03",
                    metadata={"requires_carried_object_id": "cup-03"},
                ),
            ),
        )

        result = DualBrainExecutive(
            make_registry(scripted),
            SharedWorldMemory(),
        ).execute(mission)

        self.assertEqual(result.status, MissionStatus.SUCCEEDED)
        self.assertEqual(
            [call.kind for call in scripted.calls],
            [
                SkillKind.SEARCH_OBJECT,
                SkillKind.APPROACH_ALIGN,
                SkillKind.MANIPULATE,
                SkillKind.VERIFY,
                SkillKind.NAVIGATE,
            ],
        )


if __name__ == "__main__":
    unittest.main()
