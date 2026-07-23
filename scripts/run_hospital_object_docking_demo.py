#!/usr/bin/env python3
"""Run an isolated Hospital object-level precision docking demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hospital_vln.object_docking import (  # noqa: E402
    build_object_docking_plan,
    load_object_targets,
    parse_standoff,
    resolve_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="请停到红色方块前0.8米")
    parser.add_argument(
        "--objects",
        type=Path,
        default=ROOT / "hospital_vln/object_targets_demo.json",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=ROOT / "outputs/hospital_vln/lingbot_map/map.yaml",
    )
    parser.add_argument(
        "--places",
        type=Path,
        default=ROOT / "outputs/hospital_vln/places_formal.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/hospital_object_docking",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--arrival-hold-seconds", type=float, default=0.0)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_object_targets(args.objects.resolve())
    target = resolve_object(args.command, targets)
    standoff_m = parse_standoff(args.command)
    plan = build_object_docking_plan(args.map.resolve(), target, standoff_m)

    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "docking_plan.json"
    plan_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pose = plan.docking_pose
    print(f"Object: {target.name} ({target.object_id})")
    print(
        f"Constraint: in front {standoff_m:.3f} m -> "
        f"dock=({pose.x:.3f}, {pose.y:.3f}, {pose.yaw:.3f})"
    )
    print(
        f"Validated: path={plan.path_length_m:.3f} m, "
        f"object_distance={plan.object_distance_m:.3f} m, "
        f"facing_error={plan.facing_error_rad:.3f} rad"
    )
    print(f"Isolated plan: {plan_path}")
    if args.plan_only:
        return 0

    simulator = ROOT / "isaacsim/python.sh"
    argv = [
        str(simulator),
        str(ROOT / "run_g1d_hospital_vln.py"),
        "--command",
        args.command,
        "--target-id",
        "waiting_area",
        f"--dynamic-docking-pose={pose.x},{pose.y},{pose.yaw}",
        f"--demo-object-pose={target.x},{target.y},{target.z}",
        "--demo-object-size",
        str(target.size_m),
        "--map",
        str(args.map.resolve()),
        "--places",
        str(args.places.resolve()),
        "--output-dir",
        str(args.output.resolve()),
        "--live-dir",
        str((args.output / "live").resolve()),
        "--position-tolerance",
        "0.03",
        "--yaw-tolerance",
        "0.05",
        "--arrival-hold-seconds",
        str(args.arrival_hold_seconds),
    ]
    if args.headless or args.test:
        argv.append("--headless")
    if args.test:
        argv.append("--test")
    if args.no_camera:
        argv.append("--no-camera")
    if args.record_gif:
        argv.extend(
            [
                "--record-gif",
                str((args.output / "object_docking.gif").resolve()),
            ]
        )
    print("Launching isolated Isaac demo; existing 6006 dashboard is not modified.")
    return subprocess.run(argv, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
