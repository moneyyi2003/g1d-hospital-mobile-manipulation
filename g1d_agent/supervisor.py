"""VLA handoff supervisor driven by live object and robot observations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from .adapters import StepAdapter
from .interaction import InteractionProfile, InteractionProfileDatabase
from .models import StepResult, StepStatus, TaskStep
from .readiness import (
    ObjectObservation,
    ReadinessAction,
    ReadinessDecision,
    VlaReadinessGate,
)


class ObjectObservationProvider(Protocol):
    def observe(
        self,
        step: TaskStep,
        context: Mapping[str, Any],
        profile: InteractionProfile,
    ) -> ObjectObservation | None:
        """Return the latest metric object/base observation or None."""


class ReadinessRecoveryController(Protocol):
    def recover(
        self,
        decision: ReadinessDecision,
        observation: ObjectObservation,
        profile: InteractionProfile,
    ) -> Mapping[str, Any]:
        """Apply one bounded recovery (scan/wait/stop/redock) and report its result."""


class UnavailableObservationProvider:
    def observe(
        self,
        step: TaskStep,
        context: Mapping[str, Any],
        profile: InteractionProfile,
    ) -> ObjectObservation | None:
        return None


@dataclass(frozen=True)
class BackendObservationProvider:
    """Adapt an integration backend's optional live-readiness method."""

    backend: Any

    def observe(
        self,
        step: TaskStep,
        context: Mapping[str, Any],
        profile: InteractionProfile,
    ) -> ObjectObservation | None:
        raw = self.backend.observe_readiness(
            {
                "schema_version": 1,
                "step": step.to_dict(),
                "mission_context": dict(context),
                "interaction_profile": profile.to_dict(),
            }
        )
        if raw is None:
            return None
        return ObjectObservation.from_dict(raw)


@dataclass(frozen=True)
class BackendRecoveryController:
    """Adapt an integration backend's optional bounded recovery method."""

    backend: Any

    def recover(
        self,
        decision: ReadinessDecision,
        observation: ObjectObservation,
        profile: InteractionProfile,
    ) -> Mapping[str, Any]:
        ready = getattr(self.backend, "ready", None)
        if callable(ready) and not ready():
            return {
                "status": "blocked",
                "message": "VLA integration backend 未就绪，不执行局部恢复动作。",
            }
        return self.backend.recover_readiness(
            {
                "schema_version": 1,
                "decision": decision.to_dict(),
                "observation": observation.to_dict(),
                "interaction_profile": profile.to_dict(),
            }
        )


@dataclass(frozen=True)
class JsonObservationProvider:
    """Contract-test provider; not a substitute for live perception."""

    observation: ObjectObservation

    @classmethod
    def load(cls, path: Path) -> "JsonObservationProvider":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("object observation schema_version must be 1")
        if payload.get("activation") != "contract_test_only":
            raise ValueError("static object observation must be contract_test_only")
        return cls(ObjectObservation.from_dict(payload["observation"]))

    def observe(
        self,
        step: TaskStep,
        context: Mapping[str, Any],
        profile: InteractionProfile,
    ) -> ObjectObservation:
        return self.observation


@dataclass
class ReadinessVlaAdapter:
    """Gate VLA execution and expose a concrete recovery action when blocked."""

    delegate: StepAdapter
    profiles: InteractionProfileDatabase
    observations: ObjectObservationProvider
    gate: VlaReadinessGate
    environment: str = "isaac_sim"
    recovery: ReadinessRecoveryController | None = None
    maximum_recovery_attempts: int = 3

    def execute(
        self,
        step: TaskStep,
        context: Mapping[str, Any] | None = None,
    ) -> StepResult:
        mission_context = dict(context or {})
        skill = str(step.metadata.get("skill", ""))
        try:
            profile = self.profiles.resolve(step.instruction, skill)
        except ValueError as exc:
            return StepResult(
                step.step_id,
                StepStatus.BLOCKED,
                f"没有唯一的物体—技能交互配置：{exc}",
                {
                    "agent_phase": "resolve_interaction_profile",
                    "profiles": str(self.profiles.source),
                },
            )
        if self.maximum_recovery_attempts < 0:
            return StepResult(
                step.step_id,
                StepStatus.FAILED,
                "maximum_recovery_attempts 不能为负数。",
            )
        history: list[dict[str, Any]] = []
        blocking_actions = {
            ReadinessAction.BLOCK_COLLISION,
            ReadinessAction.BLOCK_CONFIGURATION,
        }
        for attempt in range(self.maximum_recovery_attempts + 1):
            observation = self.observations.observe(
                step,
                mission_context,
                profile,
            )
            if observation is None:
                return StepResult(
                    step.step_id,
                    StepStatus.BLOCKED,
                    "缺少实时物体/底盘观测，不能判断是否允许启动 VLA。",
                    {
                        "agent_phase": "object_search",
                        "next_action": "reacquire_object",
                        "required_interface": "g1d_agent.ObjectObservationProvider",
                        "interaction_profile": profile.to_dict(),
                        "readiness_history": history,
                    },
                )
            decision = self.gate.evaluate(
                profile,
                observation,
                environment=self.environment,
            )
            record = {
                "attempt": attempt,
                "observation": observation.to_dict(),
                "decision": decision.to_dict(),
            }
            history.append(record)
            readiness_payload = {
                "agent_phase": "vla_readiness_check",
                "interaction_profile": profile.to_dict(),
                "observation": observation.to_dict(),
                "readiness": decision.to_dict(),
                "readiness_history": history,
            }
            if decision.ready:
                vla_context = {
                    **mission_context,
                    "vla_handoff": readiness_payload,
                }
                result = self.delegate.execute(step, vla_context)
                return StepResult(
                    result.step_id,
                    result.status,
                    result.message,
                    {**result.details, **readiness_payload},
                )
            if (
                decision.action in blocking_actions
                or self.recovery is None
                or attempt >= self.maximum_recovery_attempts
            ):
                return StepResult(
                    step.step_id,
                    StepStatus.BLOCKED,
                    decision.reason,
                    readiness_payload,
                )
            recovery_result = dict(
                self.recovery.recover(decision, observation, profile)
            )
            record["recovery"] = recovery_result
            if recovery_result.get("status") != "succeeded":
                return StepResult(
                    step.step_id,
                    StepStatus.BLOCKED,
                    str(
                        recovery_result.get(
                            "message",
                            f"恢复动作 {decision.action.value} 未成功。",
                        )
                    ),
                    readiness_payload,
                )
        raise AssertionError("unreachable readiness loop")


__all__ = [
    "BackendObservationProvider",
    "BackendRecoveryController",
    "ReadinessRecoveryController",
    "JsonObservationProvider",
    "ObjectObservationProvider",
    "ReadinessVlaAdapter",
    "UnavailableObservationProvider",
]
