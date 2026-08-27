#!/usr/bin/env python3
"""Build a scene-specific LingBot-Map + SAM3 candidate map bundle.

This deliberately produces *candidate* places.  A candidate is promoted to an
approved navigation goal only after its RGB/occupancy evidence is reviewed;
no USD prim name or simulator coordinate is used as semantic truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingbot_nav.errors import ConfigurationError
from lingbot_nav.mapping.alignment import align_lingbot_to_survey, build_pose_anchored_alignment
from lingbot_nav.mapping.lingbot_backend import LingBotInferenceConfig, run_lingbot_map
from lingbot_nav.mapping.mask_projection import build_track_observations
from lingbot_nav.mapping.metric_map import build_metric_occupancy_map
from lingbot_nav.perception.sam3_backend import Sam3TrackConfig, run_sam3_tracking
from lingbot_nav.place_catalog_builder import build_candidate_catalog
from lingbot_nav.models import Pose2D
from lingbot_nav.sim.map_views import render_mapping_views


# These are category prompts for RGB-based SAM3 tracking, not USD prim names.
# They cover the visible fixed furnishings used to propose navigation regions.
DEFAULT_PROMPTS = ("bed", "counter", "cabinet", "door", "table", "chair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/living_room_vln")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/lingbot-map-long.pt")
    parser.add_argument("--sam3-checkpoint", type=Path, default=ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt")
    parser.add_argument("--stage", choices=("all", "infer", "align", "map", "sam3", "project", "candidates", "render"), default="all")
    parser.add_argument("--mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--inlier-threshold", type=float, default=0.45)
    parser.add_argument("--sam3-prompt", action="append", default=[])
    parser.add_argument("--sam3-prompt-frame", type=int, default=-1)
    parser.add_argument("--sam3-threshold", type=float, default=0.50)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--force-sam3", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combine_observations(paths: list[Path], target: Path) -> dict:
    observations = []
    sources = []
    for path in paths:
        payload = _read_json(path)
        if payload.get("frame_id") != "map":
            raise ValueError(f"semantic observation not in map frame: {path}")
        observations.extend(payload.get("observations", []))
        sources.append(str(path))
    result = {
        "schema_version": 1,
        "frame_id": "map",
        "mask_source": "official_sam3.1_text_video_tracking",
        "survey_pose_used_for_model_inference": False,
        "sources": sources,
        "observations": observations,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    survey_manifest = output / "survey/capture_manifest.json"
    survey_rgb = output / "survey/rgb"
    if not survey_manifest.is_file() or not survey_rgb.is_dir():
        raise FileNotFoundError("need a completed G1-D survey at outputs/living_room_vln/survey/")
    survey = _read_json(survey_manifest)
    frames = survey.get("frames", [])
    if len(frames) < 12 or len(list(survey_rgb.glob("*.png"))) != len(frames):
        raise ValueError("incomplete RGB survey; rebuild the scene survey before LingBot-Map")

    lingbot = output / "lingbot"
    predictions = lingbot / "predictions"
    preprocessed = lingbot / "preprocessed_rgb"
    alignment_file = output / "lingbot_to_living_room.json"
    map_dir = output / "lingbot_map"
    sam3_dir = output / "sam3"
    semantic_dir = output / "semantic"
    observations_file = semantic_dir / "sam3_observations.json"

    if args.stage in {"all", "infer"}:
        manifest = lingbot / "lingbot_manifest.json"
        if args.force_inference or not manifest.is_file():
            result = run_lingbot_map(survey_rgb, args.checkpoint, lingbot, LingBotInferenceConfig(mode=args.mode))
            print(f"[living-room] LingBot-Map frames={result['outputs']['frame_count']}")
        if args.stage == "infer":
            return 0

    alignment = None
    if args.stage in {"all", "align"}:
        try:
            alignment = align_lingbot_to_survey(predictions, survey_manifest, alignment_file, inlier_threshold_m=args.inlier_threshold)
        except ConfigurationError:
            alignment = build_pose_anchored_alignment(predictions, survey_manifest, alignment_file, floor_z_m=0.001)
        print(f"[living-room] alignment={alignment['artifact_type']}")
        if args.stage == "align":
            return 0
    if alignment is None and args.stage in {"all", "map", "project"}:
        alignment = _read_json(alignment_file)

    if args.stage in {"all", "map"}:
        result = build_metric_occupancy_map(
            predictions, alignment_file, map_dir,
            scale_m_per_unit=float(alignment["scale_m_per_unit"]),
            resolution_m=args.resolution, ground_z_m=0.001,
            survey_manifest=survey_manifest, traversed_footprint_clearance_m=0.55,
        )
        print(f"[living-room] occupancy={result['map']['yaml']}")
        if args.stage == "map":
            return 0

    if args.stage in {"all", "sam3"}:
        prompts = tuple(args.sam3_prompt) or DEFAULT_PROMPTS
        prompt_frame = args.sam3_prompt_frame if args.sam3_prompt_frame >= 0 else max(0, len(frames) // 2)
        if prompt_frame >= len(frames):
            raise ValueError("SAM3 prompt frame exceeds survey length")
        manifest = sam3_dir / "sam3_manifest.json"
        if args.force_sam3 or not manifest.is_file():
            run_sam3_tracking(preprocessed, prompts, sam3_dir, checkpoint=args.sam3_checkpoint,
                              config=Sam3TrackConfig(prompt_frame=prompt_frame, probability_threshold=args.sam3_threshold, propagation_direction="both"))
        if args.stage == "sam3":
            return 0

    if args.stage in {"all", "project"}:
        manifest = _read_json(sam3_dir / "sam3_manifest.json")
        paths = []
        for item in manifest["prompts"]:
            source = Path(item["artifact_directory"])
            target = semantic_dir / "observations" / f"{source.name}.json"
            build_track_observations(predictions, source, alignment_file, target, prompt=str(item["prompt"]), scale_m_per_unit=float(alignment["scale_m_per_unit"]))
            paths.append(target)
        _combine_observations(paths, observations_file)
        if args.stage == "project":
            return 0

    if args.stage in {"all", "candidates"}:
        # Use the actual first robot pose from the completed survey as the
        # reachability seed.  The scene has no hand-authored navigation start.
        first_pose = frames[0].get("robot_pose", {})
        reachability_start = Pose2D(
            float(first_pose.get("x", 0.0)),
            float(first_pose.get("y", 0.0)),
            float(first_pose.get("yaw", 0.0)),
        )
        candidates = build_candidate_catalog(
            observations_file, map_dir / "map.yaml", output / "places_candidates.json",
            map_id="isaac-living-room-lingbot-sam3-v1",
            reachability_start=reachability_start, footprint_radius_m=0.40,
        )
        print(f"[living-room] candidate semantic places={len(candidates['places'])}")
        if args.stage == "candidates":
            return 0

    if args.stage in {"all", "render"}:
        metadata, assets = render_mapping_views(map_dir / "map.yaml", output / "map_preview", pointcloud_path=map_dir / "lingbot_map_metric.ply")
        (output / "mapping_summary.json").write_text(json.dumps({
            "schema_version": 1,
            "scene": "home_lab scanned living room",
            "source": "g1d_rgb_survey+official_lingbot_map+official_sam3.1",
            "map": metadata,
            "assets": {key: str(value) for key, value in assets.items()},
            "places_candidates": str(output / "places_candidates.json"),
            "approval_required_before_navigation": True,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
