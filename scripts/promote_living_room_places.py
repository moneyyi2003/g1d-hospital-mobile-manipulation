#!/usr/bin/env python3
"""Promote map-reachable SAM3 candidates into the reviewed VLN place schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=ROOT / "outputs/living_room_vln/places_candidates.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/living_room_vln/places_formal.json")
    args = parser.parse_args()
    source = json.loads(args.candidates.read_text(encoding="utf-8"))
    approved = []
    for candidate in source.get("places", []):
        docks = [item for item in candidate.get("docking_candidates", []) if item.get("checks", {}).get("reachable") and item.get("checks", {}).get("occupancy_status") == "free"]
        if not docks:
            continue
        selected = docks[0]
        approved.append({
            "id": candidate["id"],
            "name": candidate["name"],
            "aliases": candidate.get("aliases", [candidate["name"]]),
            "target": candidate["target"],
            "status": "approved",
            "entrance_pose": selected["pose"],
            "docking_candidates": docks,
            "selected_docking_candidate": selected["id"],
            "metadata": candidate.get("metadata", {}),
            "review": {
                "status": "approved",
                "reviewer": "reachable_sam3_lingbot_promotion",
                "source": "g1d_rgb_survey+lingbot_map+official_sam3.1",
                "reason": "SAM3 instance has a free, reachable docking pose in the LingBot occupancy map",
            },
        })
    payload = {
        "schema_version": 2,
        "map": {**source["map"], "source": "g1d_rgb_survey+official_lingbot_map+official_sam3.1"},
        "semantic_evidence": {"artifact": str(ROOT / "outputs/living_room_vln/semantic/sam3_observations.json"), "source": "official_sam3.1_text_video_tracking"},
        "places": approved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"promoted {len(approved)} reachable places -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
