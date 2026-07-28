"""Build the formal Warehouse LingBot RGB-only + SAM3.1 map bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingbot_nav.errors import ConfigurationError
from lingbot_nav.mapping.alignment import (
    align_lingbot_to_survey,
    build_pose_anchored_alignment,
)
from lingbot_nav.mapping.lingbot_backend import LingBotInferenceConfig, run_lingbot_map
from lingbot_nav.mapping.mask_projection import build_track_observations
from lingbot_nav.mapping.metric_map import build_metric_occupancy_map
from lingbot_nav.perception.sam3_backend import Sam3TrackConfig, run_sam3_tracking
from lingbot_nav.sim.map_views import render_mapping_views
from warehouse_vln.formal_places import build_formal_place_catalog


DEFAULT_PROMPTS = ("warehouse shelf", "pallet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/warehouse_vln")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/lingbot-map.pt")
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "infer", "align", "map", "sam3", "project", "places", "render"),
        default="all",
    )
    parser.add_argument("--mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--inlier-threshold", type=float, default=0.45)
    parser.add_argument(
        "--alignment-mode",
        choices=("auto", "sim3", "pose-anchored"),
        default="auto",
    )
    parser.add_argument("--sam3-prompt", action="append", default=[])
    parser.add_argument(
        "--sam3-prompt-frame",
        type=int,
        default=-1,
        help="-1 selects the last preprocessed RGB frame",
    )
    parser.add_argument("--sam3-threshold", type=float, default=0.50)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--force-sam3", action="store_true")
    return parser.parse_args()


def _combine_observations(paths: list[Path], target: Path) -> dict:
    observations = []
    sources = []
    geometry_sources = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("frame_id") != "map":
            raise ValueError(f"semantic observation is not in map frame: {path}")
        observations.extend(payload.get("observations", []))
        geometry_sources.add(str(payload.get("geometry_source", "")))
        sources.append(str(path))
    result = {
        "schema_version": 1,
        "frame_id": "map",
        "geometry_source": sorted(item for item in geometry_sources if item),
        "mask_source": "official_sam3.1_text_video_tracking",
        "survey_pose_used_for_model_inference": False,
        "sources": sources,
        "observations": observations,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    survey_manifest = output / "survey/capture_manifest.json"
    survey_rgb = output / "survey/rgb"
    lingbot_output = output / "lingbot"
    predictions = lingbot_output / "predictions"
    preprocessed_rgb = lingbot_output / "preprocessed_rgb"
    alignment_path = output / "lingbot_to_warehouse.json"
    map_output = output / "lingbot_map"
    sam3_output = output / "sam3"
    semantic_output = output / "semantic"
    combined_observations = semantic_output / "sam3_observations.json"
    preview_output = output / "map_preview"

    if args.stage in {"all", "infer"}:
        manifest_path = lingbot_output / "lingbot_manifest.json"
        if args.force_inference or not manifest_path.is_file():
            result = run_lingbot_map(
                survey_rgb,
                args.checkpoint,
                lingbot_output,
                LingBotInferenceConfig(mode=args.mode),
            )
            print(f"[Warehouse map] LingBot frames: {result['outputs']['frame_count']}")
        else:
            print(f"[Warehouse map] reusing LingBot inference: {manifest_path}")
        if args.stage == "infer":
            return 0

    alignment = None
    if args.stage in {"all", "align"}:
        if args.alignment_mode in {"auto", "sim3"}:
            try:
                alignment = align_lingbot_to_survey(
                    predictions,
                    survey_manifest,
                    alignment_path,
                    inlier_threshold_m=args.inlier_threshold,
                )
            except ConfigurationError as exc:
                if args.alignment_mode == "sim3":
                    raise
                print(f"[Warehouse map] global Sim(3) rejected: {exc}")
        if alignment is None:
            alignment = build_pose_anchored_alignment(
                predictions,
                survey_manifest,
                alignment_path,
                floor_z_m=0.001,
            )
        print(
            f"[Warehouse map] alignment={alignment['artifact_type']} "
            f"scale={alignment['scale_m_per_unit']:.6f}"
        )
        if args.stage == "align":
            return 0

    if alignment is None and args.stage in {"all", "map", "project"}:
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))

    if args.stage in {"all", "map"}:
        map_result = build_metric_occupancy_map(
            predictions,
            alignment_path,
            map_output,
            scale_m_per_unit=float(alignment["scale_m_per_unit"]),
            resolution_m=args.resolution,
            ground_z_m=0.001,
            survey_manifest=survey_manifest,
            traversed_footprint_clearance_m=0.55,
        )
        print(
            "[Warehouse map] occupancy "
            f"{map_result['map']['width']}x{map_result['map']['height']} -> "
            f"{map_result['map']['yaml']}"
        )
        if args.stage == "map":
            return 0

    if args.stage in {"all", "sam3"}:
        sam3_manifest = sam3_output / "sam3_manifest.json"
        if args.force_sam3 or not sam3_manifest.is_file():
            frames = sorted(preprocessed_rgb.glob("*.png"))
            if not frames:
                raise FileNotFoundError(f"no LingBot preprocessed RGB in {preprocessed_rgb}")
            prompt_frame = args.sam3_prompt_frame
            if prompt_frame < 0:
                prompt_frame = len(frames) - 1
            if prompt_frame >= len(frames):
                raise ValueError(
                    f"SAM3 prompt frame {prompt_frame} exceeds {len(frames)} RGB frames"
                )
            prompts = tuple(args.sam3_prompt) or DEFAULT_PROMPTS
            direction = "backward" if prompt_frame == len(frames) - 1 else "both"
            result = run_sam3_tracking(
                preprocessed_rgb,
                prompts,
                sam3_output,
                checkpoint=args.sam3_checkpoint,
                config=Sam3TrackConfig(
                    prompt_frame=prompt_frame,
                    probability_threshold=args.sam3_threshold,
                    propagation_direction=direction,
                ),
            )
            print(
                f"[Warehouse map] SAM3 prompts={len(result['prompts'])} "
                f"prompt_frame={prompt_frame}"
            )
        else:
            print(f"[Warehouse map] reusing SAM3 tracking: {sam3_manifest}")
        if args.stage == "sam3":
            return 0

    if args.stage in {"all", "project"}:
        sam_manifest = json.loads(
            (sam3_output / "sam3_manifest.json").read_text(encoding="utf-8")
        )
        observation_paths = []
        for prompt_record in sam_manifest["prompts"]:
            artifact_directory = Path(prompt_record["artifact_directory"])
            target = (
                semantic_output
                / "observations"
                / f"{artifact_directory.name}.json"
            )
            observations = build_track_observations(
                predictions,
                artifact_directory,
                alignment_path,
                target,
                prompt=str(prompt_record["prompt"]),
                scale_m_per_unit=float(alignment["scale_m_per_unit"]),
            )
            observation_paths.append(target)
            print(
                f"[Warehouse map] projected {len(observations)} observations "
                f"for {prompt_record['prompt']!r}"
            )
        combined = _combine_observations(observation_paths, combined_observations)
        print(
            f"[Warehouse map] combined semantic observations: "
            f"{len(combined['observations'])}"
        )
        if args.stage == "project":
            return 0

    if args.stage in {"all", "places"}:
        places = build_formal_place_catalog(
            map_output / "map.yaml",
            combined_observations,
            alignment_path,
            output / "places_formal.json",
        )
        approved = sum(
            item.get("status") == "approved" for item in places["places"]
        )
        rejected = len(places["places"]) - approved
        print(
            f"[Warehouse map] places approved={approved} rejected={rejected}"
        )
        if args.stage == "places":
            return 0

    if args.stage in {"all", "render"}:
        metadata, assets = render_mapping_views(
            map_output / "map.yaml",
            preview_output,
            pointcloud_path=map_output / "lingbot_map_metric.ply",
        )
        summary = {
            "schema_version": 1,
            "scene": "MobileManiBench warehouse_multiple_shelves",
            "map": metadata,
            "assets": {name: str(path) for name, path in assets.items()},
            "inputs": {
                "survey_manifest": str(survey_manifest),
                "predictions": str(predictions),
                "alignment": str(alignment_path),
                "sam3": str(sam3_output / "sam3_manifest.json"),
                "semantic_observations": str(combined_observations),
                "places": str(output / "places_formal.json"),
            },
        }
        summary_path = output / "mapping_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Warehouse map] previews: {preview_output}")
        print(f"[Warehouse map] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
