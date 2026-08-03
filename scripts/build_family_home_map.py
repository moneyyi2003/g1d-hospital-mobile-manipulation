#!/usr/bin/env python3
"""Build the formal family-home map only from G1-D RGB survey products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from family_home_vln.formal_mapping import (  # noqa: E402
    build_formal_object_catalog,
    build_formal_place_catalog,
    build_scan_semantic_layers,
)
from family_home_vln.discovery import (  # noqa: E402
    DISCOVERY_PIPELINE_VERSION,
    run_object_discovery,
    survey_signature,
    validate_survey,
)
from family_home_vln.household_objects import OBJECT_SET_SIGNATURE  # noqa: E402
from family_home_vln.rgb_triangulation import (  # noqa: E402
    triangulate_reviewed_sam3_track,
)
from lingbot_nav.errors import ConfigurationError  # noqa: E402
from lingbot_nav.mapping.alignment import (  # noqa: E402
    align_lingbot_to_survey,
    build_pose_anchored_alignment,
)
from lingbot_nav.mapping.lingbot_backend import (  # noqa: E402
    LingBotInferenceConfig,
    run_lingbot_map,
)
from lingbot_nav.mapping.mask_projection import build_track_observations  # noqa: E402
from lingbot_nav.mapping.metric_map import build_metric_occupancy_map  # noqa: E402
from lingbot_nav.perception.sam3_backend import (  # noqa: E402
    Sam3TrackConfig,
    run_sam3_tracking,
)
from lingbot_nav.sim.map_views import render_mapping_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/family_home_vln")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/lingbot-map.pt")
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "discover",
            "infer",
            "align",
            "map",
            "sam3",
            "project",
            "layers",
            "objects",
            "places",
            "render",
        ),
        default="all",
    )
    parser.add_argument("--mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--lingbot-num-scale-frames", type=int, default=8)
    parser.add_argument("--lingbot-keyframe-interval", type=int, default=1)
    parser.add_argument("--lingbot-kv-cache-window", type=int, default=64)
    parser.add_argument("--lingbot-window-size", type=int, default=64)
    parser.add_argument("--lingbot-overlap-size", type=int, default=16)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--inlier-threshold", type=float, default=0.45)
    parser.add_argument(
        "--alignment-mode", choices=("auto", "sim3", "pose-anchored"), default="auto"
    )
    parser.add_argument("--sam3-prompt", action="append", default=[])
    parser.add_argument(
        "--allow-manual-prompts",
        action="store_true",
        help="explicit diagnostic override; normal formal mapping uses autonomous discovery",
    )
    parser.add_argument(
        "--sam3-prompt-frame",
        action="append",
        default=[],
        metavar="PROMPT=INDEX",
        help=(
            "operator override for one discovered label; autonomous discovery supplies "
            "the default label and frame"
        ),
    )
    parser.add_argument(
        "--discovery-model",
        type=Path,
        default=ROOT / "checkpoints/florence-2-base-ft",
    )
    parser.add_argument("--discovery-maximum-frames", type=int, default=80)
    parser.add_argument("--discovery-min-frame-occurrences", type=int, default=2)
    parser.add_argument("--discovery-max-objects", type=int, default=16)
    parser.add_argument("--sam3-threshold", type=float, default=0.50)
    parser.add_argument(
        "--sam3-projection-window",
        type=int,
        default=36,
        help="project prompt frame plus this many forward frames to reject late tracker drift",
    )
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--force-discovery", action="store_true")
    parser.add_argument("--force-sam3", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combine_observations(paths: list[Path], target: Path) -> dict:
    observations = []
    geometry_sources = set()
    for path in paths:
        payload = _read_json(path)
        if payload.get("frame_id") != "map":
            raise ValueError(f"semantic observation is not in map frame: {path}")
        observations.extend(payload.get("observations", []))
        geometry_sources.add(str(payload.get("geometry_source", "")))
    result = {
        "schema_version": 1,
        "frame_id": "map",
        "geometry_source": sorted(item for item in geometry_sources if item),
        "mask_source": "official_sam3.1_text_video_tracking",
        "survey_pose_used_for_model_inference": False,
        "observations": observations,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _build_reviewed_metric_object_anchors(
    survey_manifest: Path,
    sam3_manifest: Path,
    review_file: Path,
    output_file: Path,
) -> dict:
    """Triangulate explicitly reviewed portable-object mask windows."""

    review = _read_json(review_file)
    sam3 = _read_json(sam3_manifest)
    sessions = {
        str(item.get("prompt", "")).casefold(): Path(
            item["artifact_directory"]
        )
        for item in sam3.get("prompts", [])
    }
    objects = {}
    for label, policy in review.get("labels", {}).items():
        localization = policy.get("metric_localization")
        if (
            policy.get("status") != "approved"
            or policy.get("manipulation_ready") is not True
            or not isinstance(localization, dict)
        ):
            continue
        prompt = str(localization["sam3_prompt"]).casefold()
        if prompt not in sessions:
            raise ValueError(
                f"metric localization for {label!r} needs SAM3 prompt {prompt!r}"
            )
        result = triangulate_reviewed_sam3_track(
            survey_manifest,
            sessions[prompt],
            start_frame=int(localization["start_frame"]),
            end_frame_exclusive=int(localization["end_frame_exclusive"]),
            downward_pitch_deg=float(
                localization.get("downward_pitch_deg", 25.0)
            ),
            minimum_camera_baseline_m=float(
                localization.get("minimum_camera_baseline_m", 0.08)
            ),
            maximum_median_ray_error_m=float(
                localization.get("maximum_median_ray_error_m", 0.05)
            ),
        )
        result["sam3_prompt"] = prompt
        result["reviewed_object_label"] = label
        objects[label.casefold()] = result
    payload = {
        "schema_version": 1,
        "artifact_type": "reviewed_rgb_sam3_metric_object_anchors",
        "survey_manifest": str(survey_manifest),
        "sam3_manifest": str(sam3_manifest),
        "objects": objects,
        "isaac_scene_truth_used": False,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _limit_observation_window(path: Path, start: int, frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError("--sam3-projection-window must be positive")
    payload = _read_json(path)
    original = payload.get("observations", [])
    payload["observations"] = [
        item
        for item in original
        if start <= int(item["frame_index"]) < start + frame_count
    ]
    payload["temporal_quality_gate"] = {
        "start_frame": start,
        "end_frame_exclusive": start + frame_count,
        "reason": "reject late SAM3 tracker drift after the prompted object leaves view",
        "raw_observation_count": len(original),
        "accepted_observation_count": len(payload["observations"]),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(payload["observations"])


def _explicit_prompt_frames(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        prompt, separator, raw_index = value.rpartition("=")
        if not separator or not prompt.strip():
            raise ValueError("--sam3-prompt-frame must use PROMPT=INDEX")
        index = int(raw_index)
        if index < 0:
            raise ValueError("SAM3 prompt frame must be non-negative")
        result[prompt.strip().casefold()] = index
    return result


def _run_prompt_sessions(
    preprocessed_rgb: Path,
    prompts: tuple[str, ...],
    prompt_frames: dict[str, int],
    output: Path,
    checkpoint: Path,
    threshold: float,
    *,
    force: bool,
    prompt_source: str,
    survey_signature_value: str,
) -> dict:
    frame_count = len(tuple(preprocessed_rgb.glob("*.png")))
    if frame_count == 0:
        raise FileNotFoundError(f"no LingBot preprocessed RGB in {preprocessed_rgb}")
    records = []
    sessions = output / "prompt_sessions"

    def write_incremental_manifest() -> dict:
        manifest = {
            "schema_version": 1,
            "pipeline": "official_sam3.1_per_autonomously_discovered_label",
            "video_resource": str(preprocessed_rgb),
            "survey_signature": survey_signature_value,
            "prompt_source": prompt_source,
            "prompt_frame_selection": (
                "Florence-2 RGB detection frame or explicit diagnostic override"
            ),
            "category_list_supplied_to_discovery": False,
            "usd_semantics_read": False,
            "scene_object_coordinates_read": False,
            "complete": len(records) == len(prompts),
            "completed_prompt_count": len(records),
            "expected_prompt_count": len(prompts),
            "prompts": records,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "sam3_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    for prompt_index, prompt in enumerate(prompts):
        prompt_frame = prompt_frames[prompt.casefold()]
        if prompt_frame >= frame_count:
            raise ValueError(f"SAM3 prompt frame {prompt_frame} exceeds {frame_count} RGB frames")
        prompt_key = hashlib.sha256(prompt.casefold().encode("utf-8")).hexdigest()[:12]
        session_root = sessions / prompt_key
        session_manifest = session_root / "sam3_manifest.json"
        if force and session_root.exists():
            shutil.rmtree(session_root)
        config = Sam3TrackConfig(
            prompt_frame=prompt_frame,
            probability_threshold=threshold,
            # The upstream multiplex predictor can lose text-seeded tracker
            # points when it switches from forward to reverse propagation.
            # Survey order gives each target a clear first view.
            propagation_direction="forward",
        )
        result = _read_json(session_manifest) if session_manifest.is_file() else None
        cached_config = result.get("config", {}) if result is not None else {}
        cache_matches = (
            result is not None
            and str(result["prompts"][0]["prompt"]).casefold() == prompt.casefold()
            and int(cached_config.get("prompt_frame", -1)) == prompt_frame
            and float(cached_config.get("probability_threshold", -1.0)) == threshold
            and cached_config.get("propagation_direction") == "forward"
        )
        if not cache_matches:
            if session_root.exists():
                shutil.rmtree(session_root)
            result = run_sam3_tracking(
                preprocessed_rgb,
                (prompt,),
                session_root,
                checkpoint=checkpoint,
                config=config,
            )
        record = dict(result["prompts"][0])
        artifact_name = Path(record["artifact_directory"]).name
        actual_artifacts = session_root / artifact_name
        if not actual_artifacts.is_dir():
            raise FileNotFoundError(
                f"SAM3 session manifest has no artifact directory: {actual_artifacts}"
            )
        record["artifact_directory"] = str(actual_artifacts.resolve())
        record["prompt_frame"] = prompt_frame
        records.append(record)
        # Each prompt is an independent official SAM3 session. Persist it now
        # so an unrelated GPU workload cannot force already completed labels
        # to be recomputed after an OOM or process restart.
        write_incremental_manifest()
        print(
            f"[Family map] SAM3 {prompt!r} frame={prompt_frame} "
            f"detections={record['detections']}"
        )
    return write_incremental_manifest()


def main() -> int:
    args = parse_args()
    for name in (
        "lingbot_num_scale_frames",
        "lingbot_keyframe_interval",
        "lingbot_kv_cache_window",
        "lingbot_window_size",
        "lingbot_overlap_size",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.lingbot_overlap_size >= args.lingbot_window_size:
        raise ValueError("--lingbot-overlap-size must be smaller than window size")
    output = args.output_dir.expanduser().resolve()
    survey_manifest = output / "survey/capture_manifest.json"
    survey_rgb = output / "survey/rgb"
    lingbot_output = output / "lingbot"
    predictions = lingbot_output / "predictions"
    preprocessed_rgb = lingbot_output / "preprocessed_rgb"
    alignment_path = output / "lingbot_to_family_home.json"
    map_output = output / "lingbot_map"
    sam3_output = output / "sam3"
    semantic_output = output / "semantic"
    combined_observations = semantic_output / "sam3_observations.json"
    preview_output = output / "map_preview"
    discovery_path = output / "discovery/object_discovery.json"

    survey, _rgb_files = validate_survey(survey_manifest, survey_rgb)
    if survey.get("household_object_set_signature") != OBJECT_SET_SIGNATURE:
        raise ValueError(
            "RGB 巡检不是当前家庭物品版本；请重新运行 home-survey 后再建图。"
        )
    current_survey_signature = survey_signature(survey_manifest, survey_rgb)

    if args.stage in {"all", "discover"}:
        cached_discovery = _read_json(discovery_path) if discovery_path.is_file() else {}
        cached_input = cached_discovery.get("input", {})
        cached_gates = cached_discovery.get("quality_gates", {})
        cached_model = cached_discovery.get("model", {})
        discovery_cache_matches = (
            cached_discovery.get("pipeline_version") == DISCOVERY_PIPELINE_VERSION
            and cached_input.get("survey_signature") == current_survey_signature
            and cached_model.get("path") == str(args.discovery_model.resolve())
            and cached_gates.get("min_frame_occurrences")
            == args.discovery_min_frame_occurrences
            and cached_gates.get("max_objects") == args.discovery_max_objects
            and len(cached_input.get("sampled_frame_indices", []))
            == min(len(_rgb_files), args.discovery_maximum_frames)
        )
        if args.force_discovery or not discovery_cache_matches:
            discovery = run_object_discovery(
                survey_manifest,
                survey_rgb,
                discovery_path,
                model_path=args.discovery_model,
                maximum_frames=args.discovery_maximum_frames,
                min_frame_occurrences=args.discovery_min_frame_occurrences,
                max_objects=args.discovery_max_objects,
            )
        else:
            discovery = cached_discovery
            print(f"[Family map] reusing autonomous discovery: {discovery_path}")
        if not discovery.get("objects"):
            raise ValueError("自主 RGB 发现没有形成跨帧物体候选，停止 SAM3/正式地图构建")
        if args.stage == "discover":
            return 0

    if args.stage in {"all", "infer"}:
        manifest_path = lingbot_output / "lingbot_manifest.json"
        signature_path = lingbot_output / "survey_signature.txt"
        requested_lingbot_config = LingBotInferenceConfig(
            mode=args.mode,
            num_scale_frames=args.lingbot_num_scale_frames,
            keyframe_interval=args.lingbot_keyframe_interval,
            kv_cache_sliding_window=args.lingbot_kv_cache_window,
            window_size=args.lingbot_window_size,
            overlap_size=args.lingbot_overlap_size,
        )
        cached_manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        cache_matches = (
            manifest_path.is_file()
            and signature_path.is_file()
            and signature_path.read_text(encoding="utf-8").strip()
            == current_survey_signature
            and cached_manifest.get("config")
            == {
                key: value
                for key, value in vars(requested_lingbot_config).items()
            }
        )
        if args.force_inference or not cache_matches:
            result = run_lingbot_map(
                survey_rgb,
                args.checkpoint,
                lingbot_output,
                requested_lingbot_config,
            )
            signature_path.write_text(current_survey_signature + "\n", encoding="utf-8")
            print(f"[Family map] LingBot frames: {result['outputs']['frame_count']}")
        else:
            print(f"[Family map] reusing LingBot inference: {manifest_path}")
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
                print(f"[Family map] global Sim(3) rejected: {exc}")
        if alignment is None:
            alignment = build_pose_anchored_alignment(
                predictions, survey_manifest, alignment_path, floor_z_m=0.001
            )
        print(
            f"[Family map] alignment={alignment['artifact_type']} "
            f"scale={alignment['scale_m_per_unit']:.6f}"
        )
        if args.stage == "align":
            return 0

    if alignment is None and args.stage in {"all", "map", "project"}:
        alignment = _read_json(alignment_path)

    if args.stage in {"all", "map"}:
        result = build_metric_occupancy_map(
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
            f"[Family map] occupancy {result['map']['width']}x{result['map']['height']} "
            f"-> {result['map']['yaml']}"
        )
        if args.stage == "map":
            return 0

    if args.stage in {"all", "sam3"}:
        discovery = _read_json(discovery_path)
        if args.sam3_prompt and not args.allow_manual_prompts:
            raise ValueError(
                "--sam3-prompt 是人工类别注入；诊断时必须同时显式传 "
                "--allow-manual-prompts，正式流程不要使用。"
            )
        discovered = {
            str(item["label"]).casefold(): item for item in discovery.get("objects", [])
        }
        prompts = (
            tuple(args.sam3_prompt)
            if args.sam3_prompt
            else tuple(str(item["sam3_prompt"]) for item in discovery.get("objects", []))
        )
        if not prompts:
            raise ValueError("object_discovery.json 没有可交给 SAM3 的自主发现标签")
        overrides = _explicit_prompt_frames(args.sam3_prompt_frame)
        prompt_frames = {}
        for prompt in prompts:
            key = prompt.casefold()
            if key in overrides:
                prompt_frames[key] = overrides[key]
            elif key in discovered:
                prompt_frames[key] = int(discovered[key]["prompt_frame"])
            else:
                raise ValueError(
                    f"manual prompt {prompt!r} has no frame; pass "
                    f"--sam3-prompt-frame '{prompt}=INDEX'"
                )
        _run_prompt_sessions(
            preprocessed_rgb,
            prompts,
            prompt_frames,
            sam3_output,
            args.sam3_checkpoint,
            args.sam3_threshold,
            force=(
                args.force_sam3
                or not (sam3_output / "sam3_manifest.json").is_file()
                or _read_json(sam3_output / "sam3_manifest.json").get("survey_signature")
                != current_survey_signature
            ),
            prompt_source=(
                "explicit_manual_diagnostic_override"
                if args.sam3_prompt
                else "florence2_category_free_rgb_discovery"
            ),
            survey_signature_value=current_survey_signature,
        )
        if args.stage == "sam3":
            return 0

    if args.stage in {"all", "project"}:
        manifest = _read_json(sam3_output / "sam3_manifest.json")
        if manifest.get("complete") is not True:
            raise ValueError(
                "SAM3 标签会话尚未全部完成；请重新运行 home-map。"
                f"当前 {manifest.get('completed_prompt_count', len(manifest.get('prompts', [])))}/"
                f"{manifest.get('expected_prompt_count', 'unknown')}，"
                "已完成标签会从 prompt_sessions 断点复用。"
            )
        paths = []
        for record in manifest["prompts"]:
            artifact_directory = Path(record["artifact_directory"])
            target = semantic_output / "observations" / f"{artifact_directory.name}.json"
            observations = build_track_observations(
                predictions,
                artifact_directory,
                alignment_path,
                target,
                prompt=str(record["prompt"]),
                scale_m_per_unit=float(alignment["scale_m_per_unit"]),
            )
            accepted = _limit_observation_window(
                target,
                int(record["prompt_frame"]),
                args.sam3_projection_window,
            )
            paths.append(target)
            print(
                f"[Family map] projected {len(observations)} raw / {accepted} gated "
                f"observations for {record['prompt']!r}"
            )
        combined = _combine_observations(paths, combined_observations)
        print(f"[Family map] combined semantic observations: {len(combined['observations'])}")
        if args.stage == "project":
            return 0

    if args.stage in {"all", "layers"}:
        metadata = build_scan_semantic_layers(
            map_output / "map.yaml", combined_observations, semantic_output
        )
        print(
            f"[Family map] semantic anchors={len(metadata['anchors'])} "
            f"regions={len(metadata['region_labels'])}"
        )
        if args.stage == "layers":
            return 0

    if args.stage in {"all", "objects"}:
        review_file = ROOT / "family_home_vln/object_review.json"
        triangulated_anchors_file = (
            semantic_output / "metric_object_anchors.json"
        )
        metric_anchors = _build_reviewed_metric_object_anchors(
            survey_manifest,
            sam3_output / "sam3_manifest.json",
            review_file,
            triangulated_anchors_file,
        )
        print(
            "[Family map] reviewed metric object anchors="
            f"{len(metric_anchors['objects'])}"
        )
        objects = build_formal_object_catalog(
            map_output / "map.yaml",
            combined_observations,
            discovery_path,
            review_file,
            output / "objects_formal.json",
            household_object_set_signature=OBJECT_SET_SIGNATURE,
            triangulated_anchors_file=triangulated_anchors_file,
        )
        approved_objects = sum(
            item.get("status") == "approved" for item in objects["objects"]
        )
        print(
            f"[Family map] objects approved={approved_objects} "
            f"rejected={len(objects['objects']) - approved_objects}"
        )
        if args.stage == "objects":
            return 0

    if args.stage in {"all", "places"}:
        places = build_formal_place_catalog(
            map_output / "map.yaml",
            combined_observations,
            alignment_path,
            semantic_output / "region_map.npy",
            output / "places_formal.json",
            household_object_set_signature=OBJECT_SET_SIGNATURE,
        )
        approved = sum(item.get("status") == "approved" for item in places["places"])
        print(f"[Family map] places approved={approved} rejected={len(places['places']) - approved}")
        if args.stage == "places":
            return 0

    if args.stage in {"all", "render"}:
        metadata, assets = render_mapping_views(
            map_output / "map.yaml",
            preview_output,
            pointcloud_path=map_output / "lingbot_map_metric.ply",
            semantic_map_path=semantic_output / "semantic_map.npy",
            region_map_path=semantic_output / "region_map.npy",
        )
        required = {"rgb_pointcloud", "semantic", "occupancy", "region"}
        if set(assets) != required:
            raise RuntimeError(f"formal four-layer render is incomplete: {sorted(assets)}")
        descriptions = {
            "rgb_pointcloud": "G1-D RGB 巡检经 LingBot-Map 生成的 RGB 点云俯视投影",
            "semantic": "SAM3.1 mask 经 LingBot RGB-only 几何投影并通过时间窗漂移过滤",
            "occupancy": "LingBot RGB-only 深度点云构建的 ROS occupancy",
            "region": "正式 occupancy 可通行空间按 SAM3.1 语义锚点做测地划分",
        }
        for layer in metadata["layers"]:
            layer["description"] = descriptions[layer["id"]]
        summary = {
            "schema_version": 1,
            "scene": "G1-D scanned multi-zone family home",
            "source": "g1d_rgb_survey+florence2_category_free+lingbot_rgb_only+sam3.1",
            "map": metadata,
            "assets": {name: str(path) for name, path in assets.items()},
            "inputs": {
                "survey_manifest": str(survey_manifest),
                "discovery": str(discovery_path),
                "predictions": str(predictions),
                "alignment": str(alignment_path),
                "sam3": str(sam3_output / "sam3_manifest.json"),
                "semantic_observations": str(combined_observations),
                "places": str(output / "places_formal.json"),
                "objects": str(output / "objects_formal.json"),
            },
        }
        (output / "mapping_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[Family map] formal four-layer summary: {output / 'mapping_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
