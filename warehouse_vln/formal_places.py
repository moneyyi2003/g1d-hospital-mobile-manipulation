"""Build reviewed Warehouse destinations against the formal RGB-only map."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from lingbot_nav.place_catalog_builder import map_bundle_sha256
from simple_room_vln.artifacts import load_ros_grid
from simple_room_vln.core import path_length

from .artifacts import (
    ROBOT_RADIUS_M,
    WAREHOUSE_START,
    docking_approach_pose,
    plan_docking_path,
    requested_places,
    snap_pose_to_free,
)


_METADATA = {
    "east_shelf_aisle": {
        "description": "仓库东侧长货架之间可供巡检和取放任务前置导航的通道。",
        "functions": ["东侧货架巡检", "移动操作预导航", "查找货物"],
        "typical_requests": ["带我到东侧货架通道", "去东边货架看看"],
    },
    "west_shelf_aisle": {
        "description": "仓库西侧长货架之间可供巡检和取放任务前置导航的通道。",
        "functions": ["西侧货架巡检", "移动操作预导航", "查找货物"],
        "typical_requests": ["带我到西侧货架通道", "去西边货架看看"],
    },
    "loading_zone": {
        "description": "仓库入口附近用于装卸与任务交接的开放区域。",
        "functions": ["装卸", "任务交接", "返回入口"],
        "typical_requests": ["带我去装卸区", "返回仓库入口"],
    },
}


def _load_semantic_evidence(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SAM3 semantic evidence {path}: {exc}") from exc
    observations = payload.get("observations")
    if payload.get("frame_id") != "map" or not isinstance(observations, list):
        raise ValueError("SAM3 observations must be a map-frame artifact")
    if not observations:
        raise ValueError("SAM3 produced no map-frame observations; review is fail-closed")
    prompts = sorted({str(item.get("prompt", "")).strip() for item in observations})
    return payload, {
        "observation_count": len(observations),
        "track_ids": sorted({str(item["track_id"]) for item in observations}),
        "prompts": [item for item in prompts if item],
        "frame_indices": sorted({int(item["frame_index"]) for item in observations}),
    }


def build_formal_place_catalog(
    map_yaml: Path,
    semantic_observations: Path,
    alignment_file: Path,
    output_file: Path,
    *,
    reviewer: str = "warehouse_map_engineering_review",
    maximum_snap_distance_m: float = 0.75,
) -> dict[str, Any]:
    """Approve only measured poses that remain footprint-safe and reachable."""

    map_yaml = map_yaml.expanduser().resolve()
    semantic_observations = semantic_observations.expanduser().resolve()
    alignment_file = alignment_file.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    if not alignment_file.is_file():
        raise ValueError(f"alignment artifact is missing: {alignment_file}")
    alignment = json.loads(alignment_file.read_text(encoding="utf-8"))
    _, semantic_summary = _load_semantic_evidence(semantic_observations)
    grid = load_ros_grid(map_yaml, robot_radius_m=ROBOT_RADIUS_M)
    start_cell = grid.world_to_cell(WAREHOUSE_START.x, WAREHOUSE_START.y)
    if not grid.is_free(start_cell):
        raise ValueError(
            "formal occupancy does not contain the surveyed G1-D start footprint"
        )

    places = []
    for requested in requested_places():
        try:
            reviewed_pose = snap_pose_to_free(
                grid,
                requested.pose,
                maximum_distance_m=maximum_snap_distance_m,
            )
            route = plan_docking_path(
                grid,
                (WAREHOUSE_START.x, WAREHOUSE_START.y),
                reviewed_pose,
            )
        except ValueError as exc:
            places.append(
                {
                    "id": requested.place_id,
                    "name": requested.name,
                    "aliases": list(requested.aliases),
                    "status": "rejected",
                    "entrance_pose": {
                        "x": requested.pose.x,
                        "y": requested.pose.y,
                        "yaw": requested.pose.yaw,
                        "frame_id": "map",
                    },
                    "docking_candidates": [],
                    "selected_docking_candidate": "",
                    "target": {
                        "type": "semantic_region",
                        "source_id": requested.place_id,
                    },
                    "metadata": {
                        **_METADATA[requested.place_id],
                        "sam3_evidence": semantic_summary,
                    },
                    "review": {
                        "status": "rejected",
                        "reviewer": reviewer,
                        "reason": str(exc),
                        "required_action": (
                            "extend RGB survey coverage and rebuild the formal map"
                        ),
                    },
                }
            )
            continue
        snap_distance = math.dist(
            (requested.pose.x, requested.pose.y),
            (reviewed_pose.x, reviewed_pose.y),
        )
        candidate_id = f"{requested.place_id}_reviewed_v1"
        pose_payload = {
            "x": reviewed_pose.x,
            "y": reviewed_pose.y,
            "yaw": reviewed_pose.yaw,
            "frame_id": "map",
        }
        approach_pose = docking_approach_pose(reviewed_pose)
        places.append(
            {
                "id": requested.place_id,
                "name": requested.name,
                "aliases": list(requested.aliases),
                "status": "approved",
                "entrance_pose": pose_payload,
                "docking_candidates": [
                    {
                        "id": candidate_id,
                        "pose": pose_payload,
                        "checks": {
                            "clearance_m": ROBOT_RADIUS_M,
                            "footprint_radius_m": ROBOT_RADIUS_M,
                            "occupancy_status": "free",
                            "reachable": True,
                            "requested_pose_snap_distance_m": snap_distance,
                            "approach_pose": {
                                "x": approach_pose.x,
                                "y": approach_pose.y,
                                "yaw": approach_pose.yaw,
                                "frame_id": "map",
                            },
                            "approach_heading_aligned": True,
                        },
                        "review": {
                            "status": "accepted",
                            "reviewer": reviewer,
                            "evidence": [
                                str(map_yaml),
                                str(semantic_observations),
                                str(alignment_file),
                            ],
                        },
                    }
                ],
                "selected_docking_candidate": candidate_id,
                "target": {
                    "type": "semantic_region",
                    "source_id": requested.place_id,
                    "adjacent_sam3_prompts": semantic_summary["prompts"],
                },
                "metadata": {
                    **_METADATA[requested.place_id],
                    "requested_scene_pose": {
                        "x": requested.pose.x,
                        "y": requested.pose.y,
                        "yaw": requested.pose.yaw,
                    },
                    "sam3_evidence": semantic_summary,
                },
                "review": {
                    "status": "approved",
                    "reviewer": reviewer,
                    "source": (
                        "g1d_rgb_survey+lingbot_rgb_only_geometry+"
                        "sam3.1_semantics+formal_occupancy_reachability"
                    ),
                    "planned_path_length_m": path_length(route),
                },
            }
        )

    map_source = {
        "lingbot_to_metric_survey_sim3": "lingbot_rgb_only_global_sim3",
        "lingbot_depth_to_metric_survey_pose_anchor": (
            "lingbot_rgb_only_offline_survey_pose_anchored"
        ),
    }.get(str(alignment.get("artifact_type")), "lingbot_rgb_only_unknown_alignment")
    payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-mobilemanibench-warehouse-lingbot-sam3-v1",
            "sha256": map_bundle_sha256(map_yaml),
            "frame_id": "map",
            "source": map_source,
            "yaml": str(map_yaml),
        },
        "semantic_evidence": {
            "source": "official_sam3.1_text_video_tracking",
            "artifact": str(semantic_observations),
            **semantic_summary,
        },
        "places": places,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = ["build_formal_place_catalog"]
