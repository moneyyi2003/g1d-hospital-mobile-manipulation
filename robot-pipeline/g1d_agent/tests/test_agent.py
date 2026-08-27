from __future__ import annotations

import unittest
from pathlib import Path

from g1d_agent.adapters import (
    FamilyHomeVlnAdapter,
    HospitalVlnAdapter,
    PluginVlaAdapter,
    WarehouseVlnAdapter,
)
from g1d_agent.agent import G1DTaskAgent
from g1d_agent.interaction import InteractionProfileDatabase
from g1d_agent.models import (
    Capability,
    MissionStatus,
    StepKind,
    StepResult,
    StepStatus,
)
from g1d_agent.router import RuleTaskPlanner
from scripts.run_g1d_agent import (
    ROOT,
    _project_path,
    _validate_static_observation_mode,
)
from scripts.run_hospital_object_docking_demo import select_standoff


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


class ReadyBackend:
    def ready(self):
        return True

    def execute(self, request):
        return {"status": "succeeded", "success": True}


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
        self.assertEqual(plan.steps[0].metadata["skill"], "pick")
        self.assertEqual(plan.steps[1].metadata["skill"], "pick")

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
        profiles = InteractionProfileDatabase.load(
            Path(__file__).resolve().parents[1] / "interaction_profiles.json"
        )
        adapter = HospitalVlnAdapter(
            workspace=Path("/workspace"),
            profiles=profiles,
        )
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
        self.assertEqual(
            adapter.command_for(docking)[-2:],
            ["--standoff", "0.8"],
        )

    def test_cli_relative_paths_are_root_relative(self) -> None:
        self.assertEqual(
            _project_path(Path("g1d_agent/vla_backend.example.json")),
            ROOT / "g1d_agent/vla_backend.example.json",
        )

    def test_agent_standoff_overrides_free_form_command_distance(self) -> None:
        self.assertEqual(select_standoff("停到方块前1.5米", 0.8), 0.8)
        self.assertEqual(select_standoff("停到方块前0.7米", None), 0.7)

    def test_missing_interaction_profile_blocks_before_subprocess(self) -> None:
        profiles = InteractionProfileDatabase.load(
            Path(__file__).resolve().parents[1] / "interaction_profiles.json"
        )
        adapter = HospitalVlnAdapter(
            workspace=Path("/workspace"),
            profiles=profiles,
        )
        step = RuleTaskPlanner().plan("去拿杯子").steps[0]

        result = adapter.execute(step)

        self.assertEqual(result.status, StepStatus.BLOCKED)
        self.assertEqual(result.details["agent_phase"], "resolve_interaction_profile")

    def test_static_observation_cannot_enable_real_vla(self) -> None:
        adapter = PluginVlaAdapter(
            backend=ReadyBackend(),
            config_path=Path("/config.json"),
            backend_name="test:create",
        )

        with self.assertRaisesRegex(ValueError, "只允许 contract 测试"):
            _validate_static_observation_mode(
                adapter,
                Path("object_observation.json"),
            )

    def test_warehouse_adapter_only_accepts_semantic_navigation(self) -> None:
        adapter = WarehouseVlnAdapter(workspace=Path("/workspace"))
        navigation = RuleTaskPlanner().plan(
            "请带我到东侧货架通道"
        ).steps[0]
        docking = RuleTaskPlanner().plan(
            "去桌边拿起红色方块"
        ).steps[0]

        self.assertEqual(
            adapter.command_for(navigation)[:2],
            ["/workspace/mobilemanibench.sh", "warehouse-vln"],
        )
        result = adapter.execute(docking)
        self.assertEqual(result.status, StepStatus.BLOCKED)
        self.assertEqual(
            result.details["agent_phase"],
            "warehouse_capability_check",
        )

    def test_family_home_adapter_uses_home_runner_and_blocks_pregrasp(self) -> None:
        adapter = FamilyHomeVlnAdapter(workspace=Path("/workspace"))
        navigation = RuleTaskPlanner().plan(
            "我困了，请带我到卧室床边"
        ).steps[0]
        docking = RuleTaskPlanner().plan(
            "去桌边拿起红色方块"
        ).steps[0]

        self.assertEqual(
            adapter.command_for(navigation)[:2],
            ["/workspace/mobilemanibench.sh", "home-vln-formal"],
        )
        result = adapter.execute(docking)
        self.assertEqual(result.status, StepStatus.BLOCKED)
        self.assertEqual(
            result.details["agent_phase"],
            "family_home_capability_check",
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
