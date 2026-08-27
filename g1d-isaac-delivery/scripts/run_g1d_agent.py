#!/usr/bin/env python3
"""Plan or execute one G1-D task through the existing VLN and future VLA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1d_agent.adapters import (  # noqa: E402
    FamilyHomeVlnAdapter,
    HospitalVlnAdapter,
    PluginVlaAdapter,
    UnavailableVlaAdapter,
    WarehouseVlnAdapter,
)
from g1d_agent.agent import G1DTaskAgent  # noqa: E402
from g1d_agent.interaction import InteractionProfileDatabase  # noqa: E402
from g1d_agent.models import MissionStatus  # noqa: E402
from g1d_agent.readiness import VlaReadinessGate  # noqa: E402
from g1d_agent.router import RuleTaskPlanner  # noqa: E402
from g1d_agent.supervisor import (  # noqa: E402
    BackendObservationProvider,
    BackendRecoveryController,
    JsonObservationProvider,
    ReadinessVlaAdapter,
    UnavailableObservationProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        help="Chinese or English task instruction",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the generated plan; the default only prints the plan",
    )
    parser.add_argument(
        "--navigation-scene",
        choices=("hospital", "warehouse", "home"),
        default="hospital",
        help="Select the existing scene-specific VLN adapter",
    )
    parser.add_argument(
        "--vla-config",
        type=Path,
        help="VLA backend config; omitted means the explicit unavailable placeholder",
    )
    parser.add_argument(
        "--interaction-profiles",
        type=Path,
        default=ROOT / "g1d_agent/interaction_profiles.json",
        help="Object-and-skill staging/readiness profile database",
    )
    parser.add_argument(
        "--readiness-observation",
        type=Path,
        help=(
            "Contract-test object observation JSON; real execution must replace "
            "this with a live Isaac/robot observation provider"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/g1d_agent/mission.json",
    )
    parser.add_argument("--with-camera", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser.parse_args()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _project_path(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def _validate_static_observation_mode(
    raw_vla,
    observation_path: Path | None,
) -> None:
    if observation_path is not None and isinstance(raw_vla, PluginVlaAdapter):
        raise ValueError(
            "--readiness-observation 只允许 contract 测试，不能与已启用的 VLA backend 同用"
        )


def main() -> int:
    args = parse_args()
    profiles = InteractionProfileDatabase.load(
        _project_path(args.interaction_profiles)
    )
    planner = RuleTaskPlanner()
    plan = planner.plan(args.command)
    if not args.execute:
        payload = {
            "schema_version": 1,
            "status": MissionStatus.PLANNED.value,
            "message": "仅生成计划；加 --execute 才会启动 Isaac/VLA。",
            "navigation_scene": args.navigation_scene,
            "plan": plan.to_dict(),
            "steps": [],
        }
        _write(_project_path(args.output), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    raw_vla = (
        PluginVlaAdapter.from_config(_project_path(args.vla_config))
        if args.vla_config
        else UnavailableVlaAdapter()
    )
    _validate_static_observation_mode(raw_vla, args.readiness_observation)
    integration_backend = getattr(raw_vla, "backend", None)
    if args.readiness_observation:
        observations = JsonObservationProvider.load(
            _project_path(args.readiness_observation)
        )
    elif integration_backend is not None and callable(
        getattr(integration_backend, "observe_readiness", None)
    ):
        observations = BackendObservationProvider(integration_backend)
    else:
        observations = UnavailableObservationProvider()
    recovery = (
        BackendRecoveryController(integration_backend)
        if integration_backend is not None
        and callable(getattr(integration_backend, "recover_readiness", None))
        else None
    )
    vla = ReadinessVlaAdapter(
        delegate=raw_vla,
        profiles=profiles,
        observations=observations,
        gate=VlaReadinessGate(),
        recovery=recovery,
    )
    if args.navigation_scene == "warehouse":
        vln = WarehouseVlnAdapter(
            test=not args.no_test,
            no_camera=not args.with_camera,
        )
    elif args.navigation_scene == "home":
        vln = FamilyHomeVlnAdapter(
            test=not args.no_test,
            no_camera=not args.with_camera,
        )
    else:
        vln = HospitalVlnAdapter(
            test=not args.no_test,
            no_camera=not args.with_camera,
            profiles=profiles,
        )
    agent = G1DTaskAgent(
        planner=planner,
        vln=vln,
        vla=vla,
    )

    result = agent.execute(plan)
    payload = result.to_dict()
    payload["navigation_scene"] = args.navigation_scene
    _write(_project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.status is MissionStatus.SUCCEEDED:
        return 0
    if result.status is MissionStatus.BLOCKED:
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
