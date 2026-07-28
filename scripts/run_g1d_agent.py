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
    HospitalVlnAdapter,
    PluginVlaAdapter,
    UnavailableVlaAdapter,
)
from g1d_agent.agent import G1DTaskAgent  # noqa: E402
from g1d_agent.models import MissionStatus  # noqa: E402
from g1d_agent.router import RuleTaskPlanner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Chinese or English task instruction")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the generated plan; the default only prints the plan",
    )
    parser.add_argument(
        "--vla-config",
        type=Path,
        help="VLA backend config; omitted means the explicit unavailable placeholder",
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


def main() -> int:
    args = parse_args()
    vla = (
        PluginVlaAdapter.from_config(_project_path(args.vla_config))
        if args.vla_config
        else UnavailableVlaAdapter()
    )
    agent = G1DTaskAgent(
        planner=RuleTaskPlanner(),
        vln=HospitalVlnAdapter(
            test=not args.no_test,
            no_camera=not args.with_camera,
        ),
        vla=vla,
    )
    plan = agent.plan(args.command)
    if not args.execute:
        payload = {
            "schema_version": 1,
            "status": MissionStatus.PLANNED.value,
            "message": "仅生成计划；加 --execute 才会启动 Isaac/VLA。",
            "plan": plan.to_dict(),
            "steps": [],
        }
        _write(_project_path(args.output), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    result = agent.execute(plan)
    payload = result.to_dict()
    _write(_project_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.status is MissionStatus.SUCCEEDED:
        return 0
    if result.status is MissionStatus.BLOCKED:
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
