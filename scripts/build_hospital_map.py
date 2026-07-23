"""Build and render the metric Hospital map from a G1-D RGB survey."""

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
from lingbot_nav.mapping.metric_map import build_metric_occupancy_map
from lingbot_nav.sim.map_views import render_mapping_views
from hospital_vln.formal_places import build_formal_place_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/hospital_vln")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/lingbot-map.pt")
    parser.add_argument(
        "--stage", choices=("all", "infer", "align", "map", "render"), default="all"
    )
    parser.add_argument("--mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--inlier-threshold", type=float, default=0.45)
    parser.add_argument(
        "--alignment-mode",
        choices=("auto", "sim3", "pose-anchored"),
        default="auto",
        help="Use global Sim(3), offline survey-pose depth fusion, or validated fallback",
    )
    parser.add_argument("--force-inference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    survey_manifest = output / "survey/capture_manifest.json"
    survey_rgb = output / "survey/rgb"
    lingbot_output = output / "lingbot"
    predictions = lingbot_output / "predictions"
    alignment_path = output / "lingbot_to_hospital.json"
    map_output = output / "lingbot_map"
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
            print(f"[Hospital map] LingBot frames: {result['outputs']['frame_count']}")
        else:
            print(f"[Hospital map] Reusing inference: {manifest_path}")
        if args.stage == "infer":
            return 0

    alignment = None
    if args.stage in {"all", "align"}:
        alignment = None
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
                print(f"[Hospital map] global Sim(3) rejected: {exc}")
        if alignment is None:
            alignment = build_pose_anchored_alignment(
                predictions,
                survey_manifest,
                alignment_path,
                floor_z_m=0.001,
            )
        if alignment["artifact_type"] == "lingbot_to_metric_survey_sim3":
            print(
                "[Hospital map] global Sim(3) "
                f"scale={alignment['scale_m_per_unit']:.6f} "
                f"inliers={alignment['inliers']}/{alignment['correspondences']} "
                f"rmse={alignment['rmse_m']:.3f}m"
            )
        else:
            print(
                "[Hospital map] pose-anchored depth fusion "
                f"scale={alignment['scale_m_per_unit']:.6f} "
                f"floor_frames={alignment['inliers']}/{alignment['frames_considered']} "
                f"p10-p90={alignment['scale_p10']:.3f}-{alignment['scale_p90']:.3f}"
            )
        if args.stage == "align":
            return 0

    if alignment is None and args.stage in {"all", "map"}:
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
        )
        print(
            "[Hospital map] occupancy "
            f"{map_result['map']['width']}x{map_result['map']['height']} -> "
            f"{map_result['map']['yaml']}"
        )
        places = build_formal_place_catalog(
            map_output / "map.yaml", output / "places_formal.json"
        )
        print(f"[Hospital map] approved places: {len(places['places'])}")
        if args.stage == "map":
            return 0

    if args.stage in {"all", "render"}:
        metadata, assets = render_mapping_views(
            map_output / "map.yaml",
            preview_output,
            pointcloud_path=map_output / "lingbot_map_metric.ply",
        )
        summary = {
            "schema_version": 1,
            "scene": "IsaacSim Hospital reception survey",
            "map": metadata,
            "assets": {name: str(path) for name, path in assets.items()},
            "inputs": {
                "survey_manifest": str(survey_manifest),
                "predictions": str(predictions),
                "alignment": str(alignment_path),
            },
        }
        summary_path = output / "mapping_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[Hospital map] point-cloud/occupancy previews: {preview_output}")
        print(f"[Hospital map] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
