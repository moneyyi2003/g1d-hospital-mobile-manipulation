#!/usr/bin/env python3
"""Build the CGS scanned-office place catalog (candidates + approved).

Merges the per-prompt SAM3 observations dumped by the lingbot-sam3 pipeline
into one semantic artifact, runs the standard candidate builder against the
ROS map, then promotes reachable/free candidates into the formal VLN place
schema with Chinese aliases so `resolve_place` matches direct Chinese
commands without an LLM fallback.

Outputs (all under outputs/cgs_office_vln/):

    semantic/sam3_observations.json   merged observations (map frame)
    places_candidates.json            build_candidate_catalog result
    places_formal.json                approved places with zh aliases

Run from the repo root with the vln env (has lingbot_nav on PYTHONPATH):

    PYTHONPATH=lingbot_semantic_nav/src \
        .conda/envs/vln/bin/python scripts/build_cgs_office_places.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingbot_nav.models import Pose2D
from lingbot_nav.place_catalog_builder import build_candidate_catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = ROOT / "outputs/CGS/forma_8181_pipeline/maps"
DEFAULT_OUT = ROOT / "outputs/cgs_office_vln"

DEFAULT_MAP_ID = "isaac-cgs-office-lingbot-sam3-v2"
# 0.3 m footprint: the new 8181 scan has wide rooms (14.7k safe cells at this
# radius), comfortably above the G1-D chassis half-width (0.26 m).
FOOTPRINT_RADIUS_M = 0.3

# prompt -> Chinese aliases (first alias doubles as the display name).
ZH_ALIASES: dict[str, list[str]] = {
    "office chair": ["办公椅", "椅子"],
    "office desk": ["办公桌", "桌子"],
    "trash can": ["垃圾桶"],
    "potted plant": ["绿植", "盆栽", "植物"],
    "glass door": ["玻璃门", "门"],
    "water jug": ["水壶"],
}

# Configured per run.
MAP_DIR = OUT = None
MAP_ID = ""
START = None


def configure(map_dir: Path, out: Path, map_id: str, start: Pose2D) -> None:
    global MAP_DIR, OUT, MAP_ID, START
    MAP_DIR, OUT, MAP_ID, START = map_dir, out, map_id, start


def pick_start(map_dir: Path, footprint_radius_m: float) -> Pose2D:
    """Auto-select a start pose inside the largest safe component.

    Safe = free cells not within `footprint_radius_m` of any occupied cell.
    Returns the safe grid cell nearest the component bbox center.
    """
    import math
    import re

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    yaml_text = (map_dir / "map.yaml").read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in yaml_text.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip("'\"")
    resolution = float(fields["resolution"])
    origin_values = [float(item) for item in re.findall(r"[-+0-9.eE]+", fields["origin"])]
    origin_x, origin_y = origin_values[0], origin_values[1]
    with open(map_dir / "map.pgm", "rb") as stream:
        tokens: list[bytes] = []
        while len(tokens) < 4:
            line = stream.readline()
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        width, height = int(tokens[1]), int(tokens[2])
        pixels = np.frombuffer(stream.read(width * height), dtype=np.uint8).reshape(height, width)
    pixels = pixels[::-1]
    free = pixels == 254
    # Match TraversabilityGrid.footprint_is_free: every cell within
    # footprint_radius_m + res*sqrt(2)/2 of the center must be strictly free
    # (254); unknown (205) counts as blocked. That is a circular erosion.
    radius_m = footprint_radius_m + resolution * math.sqrt(2) / 2
    r = int(math.ceil(radius_m / resolution))
    rr, cc = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (rr * rr + cc * cc) * (resolution * resolution) <= radius_m * radius_m
    safe = ndimage.binary_erosion(free, structure=disk, border_value=0)
    labels, _ = ndimage.label(safe)
    sizes = ndimage.sum(labels > 0, labels, index=np.arange(1, labels.max() + 1))
    largest = int(np.argmax(sizes) + 1)
    ys, xs = np.nonzero(labels == largest)
    cy = (ys.min() + ys.max()) / 2.0
    cx = (xs.min() + xs.max()) / 2.0
    best = min(
        zip(xs.tolist(), ys.tolist()),
        key=lambda cell: (cell[0] - cx) ** 2 + (cell[1] - cy) ** 2,
    )
    x = origin_x + (best[0] + 0.5) * resolution
    y = origin_y + (best[1] + 0.5) * resolution
    print(f"[cgs-places] auto start: largest safe comp has {int(sizes[largest - 1])} cells, "
          f"start=({x:.2f},{y:.2f})")
    return Pose2D(x=x, y=y, yaw=0.0)


def merge_observations() -> list[str]:
    """Concatenate the per-prompt observation files into one artifact."""
    merged = {
        "schema_version": 1,
        "frame_id": "map",
        "mask_source": "official_sam3.1_text_video_tracking",
        "survey_pose_used_for_model_inference": False,
        "sources": [],
        "observations": [],
    }
    sources = sorted(MAP_DIR.glob("observations_*.json"))
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        merged["sources"].append(str(source))
        merged["observations"].extend(payload.get("observations", []))
    merged["sources"] = [
        str(Path(item).resolve().relative_to(ROOT))
        for item in merged["sources"]
    ]
    target = OUT / "semantic/sam3_observations.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[cgs-places] merged {len(merged['observations'])} observations from "
          f"{len(sources)} files -> {target.resolve()}")
    return sources


def promote(candidates_path: Path, output: Path) -> None:
    """Promote reachable/free candidates and inject Chinese aliases."""
    source = json.loads(candidates_path.read_text(encoding="utf-8"))
    approved = []
    for candidate in source.get("places", []):
        docks = [
            item for item in candidate.get("docking_candidates", [])
            if item.get("checks", {}).get("reachable")
            and item.get("checks", {}).get("occupancy_status") == "free"
        ]
        if not docks:
            continue
        prompt = candidate["name"]
        zh = ZH_ALIASES.get(prompt, [prompt])
        aliases = [*zh, prompt, candidate["id"]]
        selected = docks[0]
        approved.append({
            "id": candidate["id"],
            "name": zh[0],
            "aliases": aliases,
            "target": candidate["target"],
            "status": "approved",
            "entrance_pose": selected["pose"],
            "docking_candidates": docks,
            "selected_docking_candidate": selected["id"],
            "metadata": {**candidate.get("metadata", {}), "prompt": prompt},
            "review": {
                "status": "approved",
                "reviewer": "reachable_sam3_lingbot_promotion",
                "source": "lingbot_map+official_sam3.1",
                "reason": "SAM3 instance has a free, reachable docking pose in the LingBot occupancy map",
            },
        })
    payload = {
        "schema_version": 2,
        "map": {**source["map"], "source": "lingbot_map+official_sam3.1"},
        "semantic_evidence": {
            "artifact": str((OUT / "semantic/sam3_observations.json").resolve().relative_to(ROOT)),
            "source": "official_sam3.1_text_video_tracking",
        },
        "places": approved,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[cgs-places] promoted {len(approved)} approved places -> {output.resolve()}")
    for place in approved:
        print(f"  - {place['name']} ({place['id']}) @ "
              f"({place['entrance_pose']['x']:.2f},{place['entrance_pose']['y']:.2f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_MAP_DIR,
                        help="maps dir with observations_*.json and map.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="where to write semantic/ + places (default: outputs/cgs_office_vln)")
    parser.add_argument("--footprint-radius", type=float, default=FOOTPRINT_RADIUS_M,
                        help="robot footprint radius in meters")
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID,
                        help="catalog map identity string")
    parser.add_argument("--start-x", type=float, help="start pose x (auto if omitted)")
    parser.add_argument("--start-y", type=float, help="start pose y (auto if omitted)")
    args = parser.parse_args(argv)
    if args.start_x is not None and args.start_y is not None:
        start = Pose2D(x=args.start_x, y=args.start_y, yaw=0.0)
    else:
        start = pick_start(args.input_dir, args.footprint_radius)
    configure(args.input_dir, args.output_dir, args.map_id, start)
    merge_observations()
    candidates = OUT / "places_candidates.json"
    build_candidate_catalog(
        OUT / "semantic/sam3_observations.json",
        MAP_DIR / "map.yaml",
        candidates,
        map_id=MAP_ID,
        reachability_start=START,
        footprint_radius_m=args.footprint_radius,
    )
    raw = json.loads(candidates.read_text(encoding="utf-8"))
    print(f"[cgs-places] candidates: {len(raw['places'])} places, "
          f"sha256={raw['map']['sha256'][:16]}...")
    promote(candidates, OUT / "places_formal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
