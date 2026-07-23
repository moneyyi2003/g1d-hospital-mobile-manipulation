#!/usr/bin/env python3
"""Build isolated experimental docking candidates without modifying the stable demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hospital_vln.docking import build_waiting_area_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=ROOT / "outputs/hospital_vln/lingbot_map/map.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/hospital_docking/waiting_area_candidates.json",
    )
    parser.add_argument(
        "--blocked-candidate",
        action="append",
        default=[],
        help="Reject a candidate as dynamically occupied; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_waiting_area_artifact(
        args.map,
        args.output,
        blocked_candidate_ids=args.blocked_candidate,
    )
    selected = payload["selected_candidate_id"]
    pose = payload["selected_pose"]
    eligible = sum(item["eligible"] for item in payload["candidates"])
    print(f"Hospital docking artifact: {args.output.resolve()}")
    print(f"Eligible candidates: {eligible}/{len(payload['candidates'])}")
    print(
        f"Selected: {selected} "
        f"({pose['x']:.3f}, {pose['y']:.3f}, {pose['yaw']:.3f})"
    )
    print("Stable places_formal.json and Hospital map were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
