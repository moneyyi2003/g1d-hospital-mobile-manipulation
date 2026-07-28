"""Execution adapters for the existing Hospital VLN and future VLA backend."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Protocol

from .models import StepKind, StepResult, StepStatus, TaskStep


ROOT = Path(__file__).resolve().parents[1]


class StepAdapter(Protocol):
    def execute(
        self,
        step: TaskStep,
        context: Mapping[str, Any] | None = None,
    ) -> StepResult:
        """Execute one planned step and return a structured result."""


@dataclass
class HospitalVlnAdapter:
    """Delegate navigation to the already validated Hospital runners."""

    headless: bool = True
    test: bool = True
    no_camera: bool = True
    workspace: Path = ROOT

    def command_for(self, step: TaskStep) -> list[str]:
        if step.kind is StepKind.SEMANTIC_NAVIGATION:
            command = [
                str(self.workspace / "mobilemanibench.sh"),
                "hospital-vln",
                "--command",
                step.instruction,
            ]
        elif step.kind is StepKind.PREGRASP_DOCKING:
            command = [
                str(self.workspace / "mobilemanibench.sh"),
                "hospital-object-docking",
                "--command",
                step.instruction,
            ]
        else:
            raise ValueError(f"VLN adapter does not support step kind {step.kind.value}")
        if self.headless:
            command.append("--headless")
        if self.test:
            command.append("--test")
        if self.no_camera:
            command.append("--no-camera")
        return command

    def execute(
        self,
        step: TaskStep,
        context: Mapping[str, Any] | None = None,
    ) -> StepResult:
        argv = self.command_for(step)
        completed = subprocess.run(
            argv,
            cwd=self.workspace,
            check=False,
        )
        if completed.returncode == 0:
            if step.kind is StepKind.PREGRASP_DOCKING:
                artifacts = {
                    "docking_plan": str(
                        self.workspace
                        / "outputs/hospital_object_docking/docking_plan.json"
                    ),
                    "run_summary": str(
                        self.workspace
                        / "outputs/hospital_object_docking/run_summary.json"
                    ),
                }
            else:
                artifacts = {
                    "run_summary": str(
                        self.workspace / "outputs/hospital_vln/run_summary.json"
                    )
                }
            return StepResult(
                step.step_id,
                StepStatus.SUCCEEDED,
                "现有 Hospital VLN runner 已报告到达。",
                {
                    "argv": argv,
                    "returncode": completed.returncode,
                    "handoff_artifacts": artifacts,
                },
            )
        return StepResult(
            step.step_id,
            StepStatus.FAILED,
            f"Hospital VLN runner 失败，返回码 {completed.returncode}。",
            {"argv": argv, "returncode": completed.returncode},
        )


class UnavailableVlaAdapter:
    """Explicit placeholder used until the trained VLA is delivered."""

    reason = "VLA 权重/backend 尚未接入"

    def execute(
        self,
        step: TaskStep,
        context: Mapping[str, Any] | None = None,
    ) -> StepResult:
        return StepResult(
            step.step_id,
            StepStatus.BLOCKED,
            f"{self.reason}；导航结果不会被误报为操作成功。",
            {
                "required_interface": "g1d_agent.vla_backend_v1",
                "config_template": "g1d_agent/vla_backend.example.json",
            },
        )


class VlaBackend(Protocol):
    """Minimal contract implemented by the VLA owner's integration package."""

    def ready(self) -> bool:
        """Return true only after weights, cameras, action mapping and safety are ready."""

    def execute(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run the closed-loop manipulation phase and return a structured result."""


@dataclass
class PluginVlaAdapter:
    """Load an external VLA backend without coupling it to the agent package."""

    backend: VlaBackend
    config_path: Path
    backend_name: str

    @classmethod
    def from_config(cls, path: Path) -> "PluginVlaAdapter | UnavailableVlaAdapter":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("VLA config schema_version must be 1")
        if not payload.get("enabled", False):
            adapter = UnavailableVlaAdapter()
            adapter.reason = str(payload.get("disabled_reason", adapter.reason))
            return adapter
        factory_ref = str(payload.get("backend", {}).get("factory", ""))
        module_name, separator, factory_name = factory_ref.partition(":")
        if not separator or not module_name or not factory_name:
            raise ValueError("VLA backend factory must use 'python.module:create_backend'")
        factory = getattr(importlib.import_module(module_name), factory_name)
        backend = factory(payload)
        return cls(backend=backend, config_path=path, backend_name=factory_ref)

    def execute(
        self,
        step: TaskStep,
        context: Mapping[str, Any] | None = None,
    ) -> StepResult:
        if not self.backend.ready():
            return StepResult(
                step.step_id,
                StepStatus.BLOCKED,
                f"VLA backend {self.backend_name} 尚未就绪。",
                {"config": str(self.config_path)},
            )
        request = {
            "schema_version": 1,
            "environment": "isaac_sim",
            "robot": "g1_d",
            "instruction": step.instruction,
            "step": step.to_dict(),
            "mission_context": dict(context or {}),
        }
        raw = dict(self.backend.execute(request))
        status_text = str(raw.get("status", "")).casefold()
        if status_text == "succeeded" and raw.get("success") is True:
            status = StepStatus.SUCCEEDED
        elif status_text == "blocked":
            status = StepStatus.BLOCKED
        else:
            status = StepStatus.FAILED
        return StepResult(
            step.step_id,
            status,
            str(raw.get("message", f"VLA backend returned {status_text or 'invalid status'}")),
            {"backend": self.backend_name, "result": raw},
        )


__all__ = [
    "HospitalVlnAdapter",
    "PluginVlaAdapter",
    "StepAdapter",
    "UnavailableVlaAdapter",
    "VlaBackend",
]
