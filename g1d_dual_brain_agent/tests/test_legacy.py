from __future__ import annotations

import unittest

from g1d_agent.models import StepKind, StepResult, StepStatus
from g1d_dual_brain_agent.legacy import (
    FormalWarehouseVlnAdapter,
    LegacyVlnSkillExecutor,
)
from g1d_dual_brain_agent.memory import SharedWorldMemory
from g1d_dual_brain_agent.models import (
    SkillCommand,
    SkillKind,
    SkillStatus,
)


class FakeOldAdapter:
    def __init__(self) -> None:
        self.steps = []

    def execute(self, step, context=None):
        self.steps.append(step)
        return StepResult(step.step_id, StepStatus.SUCCEEDED, "arrived")


class LegacyBridgeTest(unittest.TestCase):
    def test_v2_warehouse_adapter_uses_formal_scan_derived_map(self) -> None:
        adapter = FormalWarehouseVlnAdapter()
        step = type(
            "Step",
            (),
            {
                "kind": StepKind.SEMANTIC_NAVIGATION,
                "instruction": "请带我到东侧货架通道",
            },
        )()

        self.assertEqual(
            adapter.command_for(step)[1],
            "warehouse-vln-formal",
        )

    def test_navigation_maps_to_existing_semantic_navigation(self) -> None:
        old = FakeOldAdapter()
        bridge = LegacyVlnSkillExecutor(old)
        command = SkillCommand(
            "cmd-1",
            "mission-1",
            "goal-1",
            SkillKind.NAVIGATE,
            "请带我到客厅沙发旁",
        )

        result = bridge.execute(command, SharedWorldMemory())

        self.assertEqual(result.status, SkillStatus.SUCCEEDED)
        self.assertEqual(old.steps[0].kind, StepKind.SEMANTIC_NAVIGATION)

    def test_alignment_maps_to_existing_object_docking(self) -> None:
        old = FakeOldAdapter()
        bridge = LegacyVlnSkillExecutor(old)
        command = SkillCommand(
            "cmd-2",
            "mission-1",
            "goal-1",
            SkillKind.APPROACH_ALIGN,
            "对准红色方块",
            target_id="red_cube_demo",
            action="pick",
        )

        result = bridge.execute(command, SharedWorldMemory())

        self.assertEqual(result.status, SkillStatus.SUCCEEDED)
        self.assertEqual(old.steps[0].kind, StepKind.PREGRASP_DOCKING)
        self.assertEqual(old.steps[0].metadata["target_id"], "red_cube_demo")


if __name__ == "__main__":
    unittest.main()
