from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from g1d_agent.interaction import InteractionProfileDatabase
from g1d_agent.models import StepResult, StepStatus
from g1d_agent.readiness import (
    ObjectObservation,
    ReadinessAction,
    VlaReadinessGate,
)
from g1d_agent.router import RuleTaskPlanner
from g1d_agent.supervisor import (
    BackendObservationProvider,
    BackendRecoveryController,
    JsonObservationProvider,
    ReadinessVlaAdapter,
    UnavailableObservationProvider,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "g1d_agent/interaction_profiles.json"
OBSERVATION = ROOT / "g1d_agent/object_observation.example.json"


class FakeVla:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, step, context=None):
        self.calls.append((step, context))
        return StepResult(step.step_id, StepStatus.SUCCEEDED, "fake vla succeeded")


class SequenceObservationProvider:
    def __init__(self, observations) -> None:
        self.observations = list(observations)
        self.index = 0

    def observe(self, step, context, profile):
        value = self.observations[min(self.index, len(self.observations) - 1)]
        self.index += 1
        return value


class FakeRecovery:
    def __init__(self) -> None:
        self.calls = []

    def recover(self, decision, observation, profile):
        self.calls.append((decision, observation, profile))
        return {"status": "succeeded", "message": decision.action.value}


class FakeIntegrationBackend:
    def __init__(self, observation) -> None:
        self.observation = observation
        self.observation_requests = []
        self.recovery_requests = []

    def observe_readiness(self, request):
        self.observation_requests.append(request)
        return self.observation.to_dict()

    def recover_readiness(self, request):
        self.recovery_requests.append(request)
        return {"status": "succeeded"}


class InteractionProfileTest(unittest.TestCase):
    def test_resolves_object_and_skill_with_strict_distance_interval(self) -> None:
        database = InteractionProfileDatabase.load(PROFILES)

        profile = database.resolve("请拿起红色方块", "pick")

        self.assertEqual(profile.object_id, "red_cube_demo")
        self.assertEqual(profile.measurement_frame_id, "base_link")
        self.assertLessEqual(
            profile.minimum_distance_m,
            profile.preferred_distance_m,
        )
        self.assertLessEqual(
            profile.preferred_distance_m,
            profile.maximum_distance_m,
        )
        with self.assertRaisesRegex(ValueError, "got 0"):
            database.resolve("请打开红色方块", "open")


class VlaReadinessGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = InteractionProfileDatabase.load(PROFILES).resolve(
            "拿起红色方块",
            "pick",
        )
        self.observation = JsonObservationProvider.load(OBSERVATION).observation
        self.gate = VlaReadinessGate()

    def test_ready_requires_all_checks(self) -> None:
        decision = self.gate.evaluate(self.profile, self.observation)

        self.assertTrue(decision.ready)
        self.assertEqual(decision.action, ReadinessAction.START_VLA)
        self.assertTrue(all(item.passed for item in decision.checks))

    def test_distance_selects_directional_recovery(self) -> None:
        too_far = self.gate.evaluate(
            self.profile,
            replace(self.observation, distance_m=1.2),
        )
        too_close = self.gate.evaluate(
            self.profile,
            replace(self.observation, distance_m=0.5),
        )

        self.assertEqual(too_far.action, ReadinessAction.MOVE_CLOSER)
        self.assertEqual(too_close.action, ReadinessAction.MOVE_AWAY)

    def test_perception_failure_precedes_motion(self) -> None:
        decision = self.gate.evaluate(
            self.profile,
            replace(
                self.observation,
                camera_names=("head_rgb",),
                distance_m=1.2,
            ),
        )

        self.assertEqual(decision.action, ReadinessAction.REACQUIRE_OBJECT)

    def test_base_must_stop_before_vla(self) -> None:
        decision = self.gate.evaluate(
            self.profile,
            replace(self.observation, base_linear_velocity_mps=0.08),
        )

        self.assertEqual(decision.action, ReadinessAction.STOP_BASE)

    def test_alignment_ik_and_collision_have_distinct_recovery(self) -> None:
        realign = self.gate.evaluate(
            self.profile,
            replace(self.observation, yaw_error_rad=0.4),
        )
        reposition = self.gate.evaluate(
            self.profile,
            replace(self.observation, ik_feasible=False),
        )
        collision = self.gate.evaluate(
            self.profile,
            replace(self.observation, collision_free=False),
        )

        self.assertEqual(realign.action, ReadinessAction.REALIGN_BASE)
        self.assertEqual(
            reposition.action,
            ReadinessAction.REPOSITION_FOR_REACHABILITY,
        )
        self.assertEqual(collision.action, ReadinessAction.BLOCK_COLLISION)

    def test_wrong_frame_is_configuration_block(self) -> None:
        decision = self.gate.evaluate(
            self.profile,
            replace(self.observation, frame_id="map"),
        )

        self.assertEqual(decision.action, ReadinessAction.BLOCK_CONFIGURATION)

    def test_observation_schema_rejects_string_boolean(self) -> None:
        value = self.observation.to_dict()
        value["visible"] = "false"

        with self.assertRaisesRegex(ValueError, "JSON booleans"):
            ObjectObservation.from_dict(value)


class ReadinessVlaAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = InteractionProfileDatabase.load(PROFILES)
        self.step = RuleTaskPlanner().plan("抓起眼前的红色方块").steps[0]

    def test_missing_live_observation_blocks_before_vla(self) -> None:
        backend = FakeVla()
        adapter = ReadinessVlaAdapter(
            backend,
            self.profiles,
            UnavailableObservationProvider(),
            VlaReadinessGate(),
        )

        result = adapter.execute(self.step, {})

        self.assertEqual(result.status, StepStatus.BLOCKED)
        self.assertEqual(result.details["agent_phase"], "object_search")
        self.assertEqual(len(backend.calls), 0)

    def test_optional_backend_hooks_adapt_to_live_interfaces(self) -> None:
        observation = JsonObservationProvider.load(OBSERVATION).observation
        backend = FakeIntegrationBackend(observation)
        profile = self.profiles.resolve("抓红色方块", "pick")
        provider = BackendObservationProvider(backend)
        controller = BackendRecoveryController(backend)

        observed = provider.observe(self.step, {}, profile)
        decision = VlaReadinessGate().evaluate(profile, observed)
        recovery = controller.recover(decision, observed, profile)

        self.assertEqual(observed.object_id, "red_cube_demo")
        self.assertEqual(recovery["status"], "succeeded")
        self.assertEqual(len(backend.observation_requests), 1)
        self.assertEqual(len(backend.recovery_requests), 1)

    def test_failed_gate_returns_recovery_without_calling_vla(self) -> None:
        backend = FakeVla()
        source = JsonObservationProvider.load(OBSERVATION)
        source = JsonObservationProvider(
            replace(source.observation, distance_m=1.1)
        )
        adapter = ReadinessVlaAdapter(
            backend,
            self.profiles,
            source,
            VlaReadinessGate(),
        )

        result = adapter.execute(self.step, {})

        self.assertEqual(result.status, StepStatus.BLOCKED)
        self.assertEqual(result.details["readiness"]["action"], "move_closer")
        self.assertEqual(len(backend.calls), 0)

    def test_ready_gate_passes_versioned_handoff_to_vla(self) -> None:
        backend = FakeVla()
        adapter = ReadinessVlaAdapter(
            backend,
            self.profiles,
            JsonObservationProvider.load(OBSERVATION),
            VlaReadinessGate(),
        )

        result = adapter.execute(self.step, {"previous_steps": []})

        self.assertEqual(result.status, StepStatus.SUCCEEDED)
        self.assertTrue(result.details["readiness"]["ready"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(
            backend.calls[0][1]["vla_handoff"]["readiness"]["action"],
            "start_vla",
        )

    def test_recovery_rechecks_live_observation_before_vla(self) -> None:
        backend = FakeVla()
        ready = JsonObservationProvider.load(OBSERVATION).observation
        source = SequenceObservationProvider(
            [replace(ready, distance_m=1.1), ready]
        )
        recovery = FakeRecovery()
        adapter = ReadinessVlaAdapter(
            backend,
            self.profiles,
            source,
            VlaReadinessGate(),
            recovery=recovery,
        )

        result = adapter.execute(self.step, {})

        self.assertEqual(result.status, StepStatus.SUCCEEDED)
        self.assertEqual(len(recovery.calls), 1)
        self.assertEqual(
            recovery.calls[0][0].action,
            ReadinessAction.MOVE_CLOSER,
        )
        self.assertEqual(len(result.details["readiness_history"]), 2)
        self.assertEqual(len(backend.calls), 1)

    def test_collision_block_never_calls_recovery(self) -> None:
        backend = FakeVla()
        ready = JsonObservationProvider.load(OBSERVATION).observation
        recovery = FakeRecovery()
        adapter = ReadinessVlaAdapter(
            backend,
            self.profiles,
            JsonObservationProvider(replace(ready, collision_free=False)),
            VlaReadinessGate(),
            recovery=recovery,
        )

        result = adapter.execute(self.step, {})

        self.assertEqual(result.status, StepStatus.BLOCKED)
        self.assertEqual(len(recovery.calls), 0)
        self.assertEqual(len(backend.calls), 0)


if __name__ == "__main__":
    unittest.main()
