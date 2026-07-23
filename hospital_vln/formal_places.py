"""Build the reviewed Hospital place catalog against a formal occupancy map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_room_vln.artifacts import load_ros_grid
from simple_room_vln.core import path_length

from .artifacts import (
    HOSPITAL_START,
    RECEPTION_POSE,
    ROBOT_RADIUS_M,
    WAITING_AREA_POSE,
)


_PLACES = (
    (
        "reception",
        "医院前台",
        ("前台", "接待处", "护士站", "reception", "reception desk", "front desk"),
        RECEPTION_POSE,
        "SM_ReceptionDesk",
    ),
    (
        "waiting_area",
        "候诊区",
        ("候诊区", "等候区", "椅子", "waiting area", "waiting chairs"),
        WAITING_AREA_POSE,
        "SM_Chair_02a",
    ),
)


def _map_digest(map_yaml: Path) -> str:
    image_name = None
    for line in map_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("image:"):
            image_name = line.split(":", 1)[1].strip().strip("'\"")
            break
    if not image_name:
        raise ValueError(f"ROS map has no image field: {map_yaml}")
    image = (map_yaml.parent / image_name).resolve()
    digest = hashlib.sha256()
    digest.update(map_yaml.read_bytes())
    digest.update(b"\0")
    digest.update(image.read_bytes())
    return digest.hexdigest()


def build_formal_place_catalog(map_yaml: Path, output_file: Path) -> dict:
    map_yaml = map_yaml.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    grid = load_ros_grid(map_yaml, robot_radius_m=ROBOT_RADIUS_M)
    places = []
    for place_id, name, aliases, pose, source_id in _PLACES:
        route = grid.plan(
            (HOSPITAL_START.x, HOSPITAL_START.y), (pose.x, pose.y)
        )
        candidate_id = f"{place_id}_reviewed_v1"
        pose_payload = {
            "x": pose.x,
            "y": pose.y,
            "yaw": pose.yaw,
            "frame_id": "map",
        }
        places.append(
            {
                "id": place_id,
                "name": name,
                "aliases": list(aliases),
                "status": "approved",
                "entrance_pose": pose_payload,
                "docking_candidates": [
                    {
                        "id": candidate_id,
                        "pose": pose_payload,
                        "status": "approved",
                        "clearance_radius_m": ROBOT_RADIUS_M,
                    }
                ],
                "selected_docking_candidate": candidate_id,
                "target": {"type": "semantic_region", "source_id": source_id},
                "review": {
                    "status": "approved",
                    "source": "measured_usd_bounds_and_formal_occupancy_reachability",
                    "planned_path_length_m": path_length(route),
                },
            }
        )
    payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-hospital-lobby-lingbot-pose-anchored-v1",
            "sha256": _map_digest(map_yaml),
            "frame_id": "map",
            "source": "lingbot_rgb_depth_offline_survey_pose_anchored",
            "yaml": str(map_yaml),
        },
        "places": places,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


__all__ = ["build_formal_place_catalog"]
