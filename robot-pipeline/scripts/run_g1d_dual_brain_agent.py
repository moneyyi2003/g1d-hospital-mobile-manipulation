#!/usr/bin/env python3
"""Plan or execute a dynamic VLN-align-VLA mission for G1-D."""

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
)
from g1d_agent.interaction import InteractionProfileDatabase  # noqa: E402
from g1d_agent.readiness import VlaReadinessGate  # noqa: E402
from g1d_agent.supervisor import (  # noqa: E402
    BackendObservationProvider,
    BackendRecoveryController,
    JsonObservationProvider,
    ReadinessVlaAdapter,
    UnavailableObservationProvider,
)
from g1d_dual_brain_agent.executive import DualBrainExecutive  # noqa: E402
from g1d_dual_brain_agent.legacy import (  # noqa: E402
    FormalWarehouseVlnAdapter,
    LegacyVlaSkillExecutor,
    LegacyVlnSkillExecutor,
)
from g1d_dual_brain_agent.memory import SharedWorldMemory  # noqa: E402
from g1d_dual_brain_agent.models import (  # noqa: E402
    MissionStatus,
    SkillKind,
)
from g1d_dual_brain_agent.planner import (  # noqa: E402
    compile_command,
    load_mission,
)
from g1d_dual_brain_agent.skills import (  # noqa: E402
    BackendMethodSkillExecutor,
    SkillRegistry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mission", type=Path, help="Version-1 mission JSON")
    source.add_argument("--command", help="Chinese or English task instruction")
    parser.add_argument(
        "--object-id",
        default="",
        help="Reviewed object ID required when --command asks for manipulation",
    )
    parser.add_argument(
        "--region-hint",
        default="",
        help="Reviewed coarse place description visited before object search",
    )
    parser.add_argument("--mission-id", default="")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute skills; default only validates and writes the mission",
    )
    parser.add_argument(
        "--navigation-scene",
        choices=("hospital", "warehouse", "home"),
        default="home",
    )
    parser.add_argument(
        "--vla-config",
        type=Path,
        help="Existing external VLA plugin config; omitted means unavailable",
    )
    parser.add_argument(
        "--interaction-profiles",
        type=Path,
        default=ROOT / "g1d_agent/interaction_profiles.json",
    )
    parser.add_argument(
        "--readiness-observation",
        type=Path,
        help="Contract test only; cannot be combined with an enabled VLA",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=ROOT / "outputs/g1d_dual_brain_agent/world_memory.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/g1d_dual_brain_agent/mission_result.json",
    )
    parser.add_argument("--with-camera", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser.parse_args()


def _project_path(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_or_create_memory(path: Path) -> SharedWorldMemory:
    return (
        SharedWorldMemory.load(path)
        if path.exists()
        else SharedWorldMemory(path)
    )


def _build_vln(args: argparse.Namespace, profiles):
    common = {
        "test": not args.no_test,
        "no_camera": not args.with_camera,
    }
    if args.navigation_scene == "warehouse":
        return FormalWarehouseVlnAdapter(**common)
    if args.navigation_scene == "hospital":
        return HospitalVlnAdapter(profiles=profiles, **common)
    return FamilyHomeVlnAdapter(**common)


def _build_raw_vla(args: argparse.Namespace):
    return (
        PluginVlaAdapter.from_config(_project_path(args.vla_config))
        if args.vla_config
        else UnavailableVlaAdapter()
    )


def _build_vla(args: argparse.Namespace, raw_vla, profiles, backend):
    if args.readiness_observation and isinstance(raw_vla, PluginVlaAdapter):
        raise ValueError(
            "--readiness-observation 只允许 contract 测试，"
            "不能与已启用 VLA 同用"
        )
    if args.readiness_observation:
        observations = JsonObservationProvider.load(
            _project_path(args.readiness_observation)
        )
    elif backend is not None and callable(
        getattr(backend, "observe_readiness", None)
    ):
        observations = BackendObservationProvider(backend)
    else:
        observations = UnavailableObservationProvider()
    recovery = (
        BackendRecoveryController(backend)
        if backend is not None
        and callable(getattr(backend, "recover_readiness", None))
        else None
    )
    return ReadinessVlaAdapter(
        delegate=raw_vla,
        profiles=profiles,
        observations=observations,
        gate=VlaReadinessGate(),
        recovery=recovery,
    )


def _mission_from_args(args: argparse.Namespace):
    if args.mission:
        return load_mission(_project_path(args.mission))
    return compile_command(
        args.command,
        object_id=args.object_id,
        region_hint=args.region_hint,
        mission_id=args.mission_id,
    )


def main() -> int:
    args = parse_args()
    mission = _mission_from_args(args)
    output_path = _project_path(args.output)
    if not args.execute:
        payload = {
            "schema_version": 1,
            "status": MissionStatus.PLANNED.value,
            "message": "任务合同已验证；加 --execute 才会启动技能。",
            "navigation_scene": args.navigation_scene,
            "mission": mission.to_dict(),
        }
        _write(output_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    profiles = InteractionProfileDatabase.load(
        _project_path(args.interaction_profiles)
    )
    raw_vla = _build_raw_vla(args)
    integration_backend = getattr(raw_vla, "backend", None)
    vla = _build_vla(
        args,
        raw_vla,
        profiles,
        integration_backend,
    )
    legacy_vln = LegacyVlnSkillExecutor(_build_vln(args, profiles))
    registry = SkillRegistry()
    registry.register(SkillKind.NAVIGATE, legacy_vln)
    if integration_backend is not None and callable(
        getattr(integration_backend, "approach_and_align", None)
    ):
        registry.register(
            SkillKind.APPROACH_ALIGN,
            BackendMethodSkillExecutor(
                integration_backend,
                SkillKind.APPROACH_ALIGN,
            ),
        )
    else:
        registry.register(SkillKind.APPROACH_ALIGN, legacy_vln)
    registry.register(
        SkillKind.MANIPULATE,
        LegacyVlaSkillExecutor(vla),
    )
    if integration_backend is not None:
        registry.register(
            SkillKind.SEARCH_OBJECT,
            BackendMethodSkillExecutor(
                integration_backend,
                SkillKind.SEARCH_OBJECT,
            ),
        )
        registry.register(
            SkillKind.VERIFY,
            BackendMethodSkillExecutor(
                integration_backend,
                SkillKind.VERIFY,
            ),
        )

    memory_path = _project_path(args.memory)
    memory = _load_or_create_memory(memory_path)
    result = DualBrainExecutive(registry, memory).execute(mission)
    payload = result.to_dict()
    payload["navigation_scene"] = args.navigation_scene
    payload["memory_path"] = str(memory_path)
    _write(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.status is MissionStatus.SUCCEEDED:
        return 0
    if result.status is MissionStatus.BLOCKED:
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
