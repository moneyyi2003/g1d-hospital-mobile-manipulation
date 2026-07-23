"""Command-line entry points for the integration adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .errors import SemanticNavError
from .intent import create_intent_parser
from .map_validation import validate_places_on_map
from .mapping.alignment import align_lingbot_to_survey
from .mapping.lingbot_backend import LingBotInferenceConfig, run_lingbot_map
from .mapping.mask_projection import build_track_observations
from .mapping.metric_map import build_metric_occupancy_map
from .sim.map_views import render_mapping_views
from .mission import MissionResolver
from .perception.sam3_backend import Sam3TrackConfig, run_sam3_tracking
from .place_db import PlaceDatabase
from .place_catalog_builder import approve_place, build_candidate_catalog
from .models import Pose2D
from .upstreams import load_upstreams


def _places(path: str, allow_legacy: bool = False) -> PlaceDatabase:
    return PlaceDatabase.load(path, allow_legacy=allow_legacy)


def _cmd_verify_upstreams(_args) -> dict:
    result = {}
    for name, repository in load_upstreams().items():
        if repository.group != "models" and not (repository.checkout / ".git").is_dir():
            result[name] = {
                "status": "locked_external_package",
                "commit": repository.commit,
                "group": repository.group,
            }
            continue
        try:
            repository.verify()
            result[name] = {"status": "ok", "commit": repository.commit}
        except SemanticNavError as exc:
            result[name] = {"status": "missing_or_mismatch", "error": str(exc)}
    return result


def _cmd_parse(args) -> dict:
    places = _places(args.places, args.allow_legacy)
    parser = create_intent_parser(
        args.provider, places, allow_rule_fallback=args.allow_rule_fallback
    )
    return MissionResolver(parser, places).resolve(args.instruction).to_dict()


def _cmd_validate_places(args) -> dict:
    places = _places(args.places, args.allow_legacy)
    checks = validate_places_on_map(places, args.map, robot_radius=args.robot_radius)
    return {
        "valid": all(item.status == "free" for item in checks),
        "checks": [item.to_dict() for item in checks],
    }


def _cmd_lingbot(args) -> dict:
    return run_lingbot_map(
        args.rgb,
        args.checkpoint,
        args.output,
        LingBotInferenceConfig(mode=args.mode),
    )


def _cmd_align_survey(args) -> dict:
    return align_lingbot_to_survey(
        args.predictions,
        args.survey_manifest,
        args.output,
        inlier_threshold_m=args.inlier_threshold,
    )


def _cmd_sam3(args) -> dict:
    prompts = [item.strip() for item in Path(args.prompts).read_text(encoding="utf-8").splitlines()]
    return run_sam3_tracking(
        args.video,
        prompts,
        args.output,
        checkpoint=args.checkpoint,
        config=Sam3TrackConfig(probability_threshold=args.threshold),
    )


def _cmd_build_map(args) -> dict:
    return build_metric_occupancy_map(
        args.predictions,
        args.alignment,
        args.output,
        scale_m_per_unit=args.scale,
        resolution_m=args.resolution,
        ground_z_m=args.ground_z,
    )


def _cmd_render_map(args) -> dict:
    metadata, assets = render_mapping_views(
        Path(args.map),
        Path(args.output),
        pointcloud_path=Path(args.pointcloud) if args.pointcloud else None,
    )
    return {
        "metadata": metadata,
        "assets": {name: str(path) for name, path in assets.items()},
    }


def _cmd_project(args) -> dict:
    observations = build_track_observations(
        args.predictions,
        args.sam3,
        args.alignment,
        args.output,
        prompt=args.prompt,
        scale_m_per_unit=args.scale,
    )
    return {"observations": len(observations), "output": args.output}


def _cmd_build_places(args) -> dict:
    result = build_candidate_catalog(
        args.observations,
        args.map,
        args.output,
        map_id=args.map_id,
        reachability_start=Pose2D(args.start_x, args.start_y, args.start_yaw, "map"),
        footprint_radius_m=args.robot_radius,
    )
    return {"candidate_places": len(result["places"]), "output": args.output}


def _cmd_approve_place(args) -> dict:
    result = approve_place(
        args.places,
        args.map,
        place_id=args.place_id,
        candidate_id=args.candidate_id,
        reviewer=args.reviewer,
        evidence=args.evidence,
    )
    return {"approved_place": result["id"], "candidate": args.candidate_id}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lingbot-nav")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-upstreams")
    verify.set_defaults(handler=_cmd_verify_upstreams)

    parse = commands.add_parser("parse")
    parse.add_argument("instruction")
    parse.add_argument("--places", required=True)
    parse.add_argument("--provider", default="deepseek", choices=("deepseek", "rule"))
    parse.add_argument("--allow-rule-fallback", action="store_true")
    parse.add_argument("--allow-legacy", action="store_true")
    parse.set_defaults(handler=_cmd_parse)

    validate = commands.add_parser("validate-places")
    validate.add_argument("--places", required=True)
    validate.add_argument("--map", required=True)
    validate.add_argument("--robot-radius", type=float, required=True)
    validate.add_argument("--allow-legacy", action="store_true")
    validate.set_defaults(handler=_cmd_validate_places)

    lingbot = commands.add_parser("lingbot-infer")
    lingbot.add_argument("--rgb", required=True)
    lingbot.add_argument("--checkpoint", required=True)
    lingbot.add_argument("--output", required=True)
    lingbot.add_argument("--mode", choices=("streaming", "windowed"), default="streaming")
    lingbot.set_defaults(handler=_cmd_lingbot)

    align = commands.add_parser("align-survey")
    align.add_argument("--predictions", required=True)
    align.add_argument("--survey-manifest", required=True)
    align.add_argument("--inlier-threshold", default=0.45, type=float)
    align.add_argument("--output", required=True)
    align.set_defaults(handler=_cmd_align_survey)

    sam3 = commands.add_parser("sam3-track")
    sam3.add_argument("--video", required=True, help="MP4 or pixel-aligned RGB frame directory")
    sam3.add_argument("--prompts", required=True, help="UTF-8 file with one concept per line")
    sam3.add_argument("--output", required=True)
    sam3.add_argument("--checkpoint")
    sam3.add_argument("--threshold", type=float, default=0.5)
    sam3.set_defaults(handler=_cmd_sam3)

    build_map = commands.add_parser("build-map")
    build_map.add_argument("--predictions", required=True)
    build_map.add_argument("--alignment", required=True)
    build_map.add_argument("--scale", required=True, type=float)
    build_map.add_argument("--resolution", default=0.05, type=float)
    build_map.add_argument("--ground-z", default=0.0, type=float)
    build_map.add_argument("--output", required=True)
    build_map.set_defaults(handler=_cmd_build_map)

    render_map = commands.add_parser("render-map")
    render_map.add_argument("--map", required=True)
    render_map.add_argument("--pointcloud")
    render_map.add_argument("--output", required=True)
    render_map.set_defaults(handler=_cmd_render_map)

    project = commands.add_parser("project-tracks")
    project.add_argument("--predictions", required=True)
    project.add_argument("--sam3", required=True)
    project.add_argument("--alignment", required=True)
    project.add_argument("--scale", required=True, type=float)
    project.add_argument("--prompt", required=True)
    project.add_argument("--output", required=True)
    project.set_defaults(handler=_cmd_project)

    build_places = commands.add_parser("build-place-candidates")
    build_places.add_argument("--observations", required=True)
    build_places.add_argument("--map", required=True)
    build_places.add_argument("--map-id", required=True)
    build_places.add_argument("--start-x", required=True, type=float)
    build_places.add_argument("--start-y", required=True, type=float)
    build_places.add_argument("--start-yaw", default=0.0, type=float)
    build_places.add_argument("--robot-radius", required=True, type=float)
    build_places.add_argument("--output", required=True)
    build_places.set_defaults(handler=_cmd_build_places)

    approve = commands.add_parser("approve-place")
    approve.add_argument("--places", required=True)
    approve.add_argument("--map", required=True)
    approve.add_argument("--place-id", required=True)
    approve.add_argument("--candidate-id", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--evidence", action="append", required=True)
    approve.set_defaults(handler=_cmd_approve_place)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (SemanticNavError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
