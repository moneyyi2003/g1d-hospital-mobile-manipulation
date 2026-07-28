from __future__ import annotations

import unittest
from pathlib import Path

from g1d_agent.adapters import HospitalVlnAdapter
from g1d_agent.agent import G1DTaskAgent
from g1d_agent.models import (
    Capability,
    MissionStatus,
    StepKind,
    StepResult,
    StepStatus,
)
from g1d_agent.router import RuleTaskPlanner
from scripts.run_g1d_agent import ROOT, _project_path


class FakeAdapter:
    def __init__(self, status: StepStatus) -> None:
        self.status = status
        self.calls = []
        self.contexts = []

    def execute(self, step, context=None):
        self.calls.append(step)
        self.contexts.append(context)
        return StepResult(step.step_id, self.status, self.status.value)


class RaisingAdapter:
    def execute(self, step, context=None):
        raise RuntimeError("backend disconnected")


class RuleTaskPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RuleTaskPlanner()

    def test_navigation_only_uses_existing_vln(self) -> None:
        plan = self.planner.plan("带我去找个能坐着等医生的地方")

        self.assertEqual(plan.route, "vln")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability, Capability.VLN)
        self.assertEqual(plan.steps[0].kind, StepKind.SEMANTIC_NAVIGATION)

    def test_mobile_manipulation_routes_vln_then_vla(self) -> None:
        plan = self.planner.plan("去桌边拿起红色方块")

        self.assertEqual(plan.route, "vln_then_vla")
        self.assertEqual(
            [step.capability for step in plan.steps],
            [Capability.VLN, Capability.VLA],
        )
        self.assertEqual(plan.steps[0].kind, StepKind.PREGRASP_DOCKING)
        self.assertEqual(plan.steps[1].kind, StepKind.MANIPULATION)

    def test_local_manipulation_uses_vla_only(self) -> None:
        plan = self.planner.plan("抓起眼前的杯子")

        self.assertEqual(plan.route, "vla")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability, Capability.VLA)

    def test_ambiguous_instruction_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "无法安全判断"):
            self.planner.plan("帮我一下")


class HospitalVlnAdapterTest(unittest.TestCase):
    def test_steps_delegate_to_existing_hospital_commands(self) -> None:
        planner = RuleTaskPlanner()
        adapter = HospitalVlnAdapter(workspace=Path("/workspace"))
        navigation = planner.plan("带我去候诊区").steps[0]
        docking = planner.plan("去桌边拿起红色方块").steps[0]

        self.assertEqual(
            adapter.command_for(navigation)[:2],
            ["/workspace/mobilemanibench.sh", "hospital-vln"],
        )
        self.assertEqual(
            adapter.command_for(docking)[:2],
            ["/workspace/mobilemanibench.sh", "hospital-object-docking"],
        )

    def test_cli_relative_paths_are_root_relative(self) -> None:
        self.assertEqual(
            _project_path(Path("g1d_agent/vla_backend.example.json")),
            ROOT / "g1d_agent/vla_backend.example.json",
        )


class G1DTaskAgentTest(unittest.TestCase):
    def test_vln_then_unavailable_vla_is_blocked(self) -> None:
        vln = FakeAdapter(StepStatus.SUCCEEDED)
        vla = FakeAdapter(StepStatus.BLOCKED)
        agent = G1DTaskAgent(RuleTaskPlanner(), vln, vla)

        result = agent.execute(agent.plan("去桌边拿起红色方块"))

        self.assertEqual(result.status, MissionStatus.BLOCKED)
        self.assertEqual(len(vln.calls), 1)
        self.assertEqual(len(vla.calls), 1)
        self.assertEqual(
            vla.contexts[0]["previous_steps"][0]["status"],
            StepStatus.SUCCEEDED.value,
        )

    def test_failed_navigation_prevents_vla_execution(self) -> None:
        vln = FakeAdapter(StepStatus.FAILED)
        vla = FakeAdapter(StepStatus.SUCCEEDED)
        agent = G1DTaskAgent(RuleTaskPlanner(), vln, vla)

        result = agent.execute(agent.plan("去桌边拿起红色方块"))

        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertEqual(len(vla.calls), 0)
        self.assertEqual(result.steps[1].status, StepStatus.SKIPPED)

    def test_adapter_exception_is_a_structured_failure(self) -> None:
        vla = FakeAdapter(StepStatus.SUCCEEDED)
        agent = G1DTaskAgent(RuleTaskPlanner(), RaisingAdapter(), vla)

        result = agent.execute(agent.plan("去桌边拿起红色方块"))

        self.assertEqual(result.status, MissionStatus.FAILED)
        self.assertEqual(result.steps[0].details["error_type"], "RuntimeError")
        self.assertEqual(len(vla.calls), 0)


if __name__ == "__main__":
    unittest.main()
