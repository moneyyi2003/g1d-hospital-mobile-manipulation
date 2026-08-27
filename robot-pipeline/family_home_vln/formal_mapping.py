"""Build scan-derived semantic, region and reviewed place artifacts."""

from __future__ import annotations

from collections import deque
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lingbot_nav.mapping.semantic_map import load_ros_occupancy
from lingbot_nav.place_catalog_builder import map_bundle_sha256
from simple_room_vln.artifacts import load_ros_grid
from simple_room_vln.core import Pose2D, path_length

from .layout import PLACES, ROBOT_RADIUS_M, START_POSE


PROMPT_BY_PLACE = {
    "living_room_sofa": "sofa",
    "bedroom_bed": "bed",
    "dining_area": "dining table",
    "kitchen_counter": "kitchen counter",
}
SEMANTIC_LABELS = tuple(PROMPT_BY_PLACE.values())
PROMPT_ALIASES_BY_PLACE = {
    "living_room_sofa": ("sofa", "couch"),
    "bedroom_bed": ("bed", "mattress"),
    # The category-free model called the dining table a "desk" in frames
    # 167-172. This alias is added only after RGB review; it was not supplied
    # to Florence during discovery.
    "dining_area": ("dining table", "dinner table", "desk"),
    "kitchen_counter": ("kitchen counter", "kitchen countertop", "countertop"),
}


def _load_observations(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SAM3 observations {path}: {exc}") from exc
    observations = payload.get("observations")
    if payload.get("frame_id") != "map" or not isinstance(observations, list):
        raise ValueError("SAM3 observations must be a map-frame artifact")
    return payload, observations


def _prompt_observations(
    observations: list[dict[str, Any]], prompt: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in observations
        if str(item.get("prompt", "")).strip().casefold() == prompt.casefold()
        and float(item.get("score", 0.0)) >= 0.35
        and int(item.get("point_count", 0)) >= 30
    ]


def _accepted_labels(observations: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return model-generated labels in first-observation order."""

    labels: list[str] = []
    seen: set[str] = set()
    for item in observations:
        prompt = str(item.get("prompt", "")).strip()
        key = prompt.casefold()
        if (
            prompt
            and key not in seen
            and float(item.get("score", 0.0)) >= 0.35
            and int(item.get("point_count", 0)) >= 30
        ):
            labels.append(prompt)
            seen.add(key)
    return tuple(labels)


def _place_evidence(
    observations: list[dict[str, Any]], place_id: str
) -> tuple[str, list[dict[str, Any]]]:
    """Match autonomous labels to navigation concepts after discovery."""

    aliases = tuple(alias.casefold() for alias in PROMPT_ALIASES_BY_PLACE[place_id])
    for prompt in _accepted_labels(observations):
        key = prompt.casefold()
        if any(key == alias or alias in key for alias in aliases):
            return prompt, _prompt_observations(observations, prompt)
    return PROMPT_BY_PLACE[place_id], []


def _anchor_xyz(items: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not items:
        raise ValueError("semantic anchor requires at least one observation")
    weighted_x = []
    weighted_y = []
    weighted_z = []
    for item in items:
        centroid = item["centroid_xyz"]
        weight = max(1, min(20, int(item.get("point_count", 1)) // 100))
        weighted_x.extend([float(centroid[0])] * weight)
        weighted_y.extend([float(centroid[1])] * weight)
        weighted_z.extend([float(centroid[2])] * weight)
    weighted_x.sort()
    weighted_y.sort()
    weighted_z.sort()
    middle = len(weighted_x) // 2
    return weighted_x[middle], weighted_y[middle], weighted_z[middle]


def _anchor(items: list[dict[str, Any]]) -> tuple[float, float]:
    x, y, _z = _anchor_xyz(items)
    return x, y


def _nearest_free_cell(cells, row: int, col: int) -> tuple[int, int] | None:
    height, width = cells.shape
    row = min(max(row, 0), height - 1)
    col = min(max(col, 0), width - 1)
    queue = deque([(row, col)])
    visited = {(row, col)}
    while queue:
        rr, cc = queue.popleft()
        if cells[rr, cc] == 0:
            return rr, cc
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidate = rr + dr, cc + dc
            if (
                0 <= candidate[0] < height
                and 0 <= candidate[1] < width
                and candidate not in visited
            ):
                visited.add(candidate)
                queue.append(candidate)
    return None


def build_scan_semantic_layers(
    map_yaml: Path,
    semantic_observations: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Rasterize SAM3 evidence and divide sensed free space by semantic anchors."""

    import numpy as np

    grid = load_ros_occupancy(map_yaml)
    _payload, observations = _load_observations(semantic_observations)
    height, width = grid.cells.shape
    semantic_labels = _accepted_labels(observations)
    votes = np.zeros((len(semantic_labels), height, width), dtype=np.float32)
    anchors: dict[str, tuple[float, float]] = {}

    for label_index, prompt in enumerate(semantic_labels):
        items = _prompt_observations(observations, prompt)
        if not items:
            continue
        anchors[prompt] = _anchor(items)
        for item in items:
            centroid = item["centroid_xyz"]
            minimum = item["minimum_xyz"]
            maximum = item["maximum_xyz"]
            cx, cy = float(centroid[0]), float(centroid[1])
            # Quantile boxes can contain depth outliers. Keep the observed
            # footprint local to its robust median while preserving its size.
            radius_x = min(1.50, max(0.12, (float(maximum[0]) - float(minimum[0])) / 2))
            radius_y = min(1.50, max(0.12, (float(maximum[1]) - float(minimum[1])) / 2))
            min_col = max(0, int(math.floor((cx - radius_x - grid.origin_x) / grid.resolution)))
            max_col = min(width - 1, int(math.floor((cx + radius_x - grid.origin_x) / grid.resolution)))
            min_row = max(0, int(math.floor((cy - radius_y - grid.origin_y) / grid.resolution)))
            max_row = min(height - 1, int(math.floor((cy + radius_y - grid.origin_y) / grid.resolution)))
            score = float(item.get("score", 0.0))
            for row in range(min_row, max_row + 1):
                y = grid.origin_y + (row + 0.5) * grid.resolution
                for col in range(min_col, max_col + 1):
                    x = grid.origin_x + (col + 0.5) * grid.resolution
                    if ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2 <= 1.0:
                        votes[label_index, row, col] += score

    semantic = np.zeros((height, width), dtype=np.uint16)
    if votes.size:
        strength = votes.max(axis=0)
        semantic = np.where(strength > 0, votes.argmax(axis=0) + 1, 0).astype(np.uint16)
    semantic[grid.cells == -1] = 0

    # Multi-source geodesic Voronoi: regions occupy only measured free cells
    # and are seeded exclusively by scan-derived semantic anchors.
    regions = np.zeros((height, width), dtype=np.uint16)
    distances = np.full((height, width), np.inf, dtype=np.float64)
    frontier: list[tuple[float, int, int, int]] = []
    region_labels: dict[int, str] = {}
    for region_id, prompt in enumerate(semantic_labels, start=1):
        if prompt not in anchors:
            continue
        x, y = anchors[prompt]
        seed = _nearest_free_cell(
            grid.cells,
            int(math.floor((y - grid.origin_y) / grid.resolution)),
            int(math.floor((x - grid.origin_x) / grid.resolution)),
        )
        if seed is None:
            continue
        row, col = seed
        if distances[row, col] == 0:
            continue
        distances[row, col] = 0.0
        regions[row, col] = region_id
        region_labels[region_id] = prompt
        heapq.heappush(frontier, (0.0, region_id, row, col))
    while frontier:
        distance, region_id, row, col = heapq.heappop(frontier)
        if distance != distances[row, col] or regions[row, col] != region_id:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = row + dr, col + dc
            if not (0 <= rr < height and 0 <= cc < width) or grid.cells[rr, cc] != 0:
                continue
            candidate = distance + grid.resolution
            if candidate < distances[rr, cc]:
                distances[rr, cc] = candidate
                regions[rr, cc] = region_id
                heapq.heappush(frontier, (candidate, region_id, rr, cc))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic_map.npy", semantic)
    np.save(output_dir / "semantic_votes.npy", votes)
    np.save(output_dir / "region_map.npy", regions)
    metadata = {
        "schema_version": 1,
        "artifact_type": "family_home_scan_derived_semantic_region_layers",
        "frame_id": "map",
        "labels": {str(index + 1): prompt for index, prompt in enumerate(semantic_labels)},
        "region_labels": {str(key): value for key, value in region_labels.items()},
        "anchors": {key: [value[0], value[1]] for key, value in anchors.items()},
        "shape": [height, width],
        "resolution": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y],
        "semantic_source": (
            "florence2_category_free_labels+official_sam3.1_masks_projected_through_"
            "lingbot_rgb_only_geometry"
        ),
        "region_source": "geodesic_partition_of_lingbot_occupancy_from_semantic_anchors",
        "isaac_fixture_geometry_used": False,
    }
    (output_dir / "semantic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _reachable_cells(grid, start: tuple[int, int]) -> set[tuple[int, int]]:
    if not grid.is_free(start):
        return set()
    queue = deque([start])
    visited = {start}
    while queue:
        row, col = queue.popleft()
        for candidate in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            if candidate not in visited and grid.is_free(candidate):
                visited.add(candidate)
                queue.append(candidate)
    return visited


def _scan_docking_pose(grid, anchor: tuple[float, float]) -> Pose2D:
    start_cell = grid.world_to_cell(START_POSE.x, START_POSE.y)
    reachable = _reachable_cells(grid, start_cell)
    if not reachable:
        raise ValueError("formal occupancy does not contain the surveyed G1-D start footprint")
    candidates = []
    for row, col in reachable:
        x, y = grid.cell_to_world((row, col))
        distance = math.dist((x, y), anchor)
        if 0.65 <= distance <= 1.45:
            candidates.append((abs(distance - 0.90), distance, row, col, x, y))
    if not candidates:
        raise ValueError("no footprint-safe reachable docking cell 0.65–1.45 m from semantic object")
    _cost, _distance, _row, _col, x, y = min(candidates)
    return Pose2D(x, y, math.atan2(anchor[1] - y, anchor[0] - x))


def plan_object_approach(
    grid,
    start: Pose2D,
    anchor: tuple[float, float],
    *,
    stand_off_m: float,
    tolerance_m: float,
    preferred_view_bearing_rad: float | None = None,
) -> tuple[Pose2D, list[tuple[float, float]]]:
    """Plan a footprint-safe object-facing base pose from a live robot pose.

    The target position is scan-derived.  The search is performed on the
    inflated formal occupancy, so the returned base cell is both reachable
    from ``start`` and collision-safe for the G1-D footprint.
    """

    if stand_off_m <= 0.0 or tolerance_m <= 0.0:
        raise ValueError("object stand-off and tolerance must be positive")
    start_cell = grid.world_to_cell(start.x, start.y)
    reachable = _reachable_cells(grid, start_cell)
    if not reachable:
        raise ValueError("current G1-D pose is not in reachable formal free space")
    candidates: list[tuple[float, float, float, float, float]] = []
    for row, col in reachable:
        x, y = grid.cell_to_world((row, col))
        distance = math.dist((x, y), anchor)
        if abs(distance - stand_off_m) <= max(tolerance_m, grid.resolution):
            bearing = math.atan2(y - anchor[1], x - anchor[0])
            bearing_error = (
                abs(
                    math.atan2(
                        math.sin(bearing - preferred_view_bearing_rad),
                        math.cos(bearing - preferred_view_bearing_rad),
                    )
                )
                if preferred_view_bearing_rad is not None
                else 0.0
            )
            candidates.append(
                (
                    bearing_error,
                    abs(distance - stand_off_m),
                    math.dist((start.x, start.y), (x, y)),
                    x,
                    y,
                )
            )
    if not candidates:
        raise ValueError(
            "no footprint-safe reachable base pose satisfies the object stand-off"
        )
    _bearing_error, _distance_error, _travel_hint, x, y = min(candidates)
    pose = Pose2D(x, y, math.atan2(anchor[1] - y, anchor[0] - x))
    route = grid.plan((start.x, start.y), (pose.x, pose.y))
    return pose, route


def build_formal_object_catalog(
    map_yaml: Path,
    semantic_observations: Path,
    discovery_file: Path,
    review_file: Path,
    output_file: Path,
    *,
    household_object_set_signature: str = "",
    triangulated_anchors_file: Path | None = None,
) -> dict[str, Any]:
    """Build a reviewed object memory without reading Isaac scene truth."""

    discovery = json.loads(discovery_file.read_text(encoding="utf-8"))
    review = json.loads(review_file.read_text(encoding="utf-8"))
    if int(review.get("schema_version", 0)) != 1:
        raise ValueError("family-home object review schema_version must be 1")
    reviewer = str(review.get("reviewer", "")).strip()
    if not reviewer:
        raise ValueError("family-home object review needs a reviewer")
    _payload, observations = _load_observations(semantic_observations)
    grid = load_ros_grid(map_yaml, robot_radius_m=ROBOT_RADIUS_M)
    discovered = {
        str(item.get("label", "")).strip().casefold(): item
        for item in discovery.get("objects", [])
    }
    policies = review.get("labels", {})
    triangulated = {}
    if triangulated_anchors_file is not None:
        triangulation_payload = json.loads(
            triangulated_anchors_file.read_text(encoding="utf-8")
        )
        triangulated = dict(triangulation_payload.get("objects", {}))
    default_policy = dict(review.get("default", {}))
    objects: list[dict[str, Any]] = []
    approved_index = 0
    for label, discovery_item in discovered.items():
        policy = {**default_policy, **dict(policies.get(label, {}))}
        evidence = _prompt_observations(observations, label)
        status = str(policy.get("status", "rejected"))
        reason = str(policy.get("reason", "")).strip()
        if status == "approved" and reason == str(
            default_policy.get("reason", "")
        ).strip():
            reason = "visual label and map-frame evidence accepted"
        if status == "approved" and not evidence:
            status = "rejected"
            reason = "review policy approved the label but no gated map-frame evidence exists"
        manipulation_ready = bool(policy.get("manipulation_ready", False))
        metric_anchor = triangulated.get(label)
        if status == "approved" and manipulation_ready and metric_anchor is None:
            status = "rejected"
            reason = (
                "manipulation-ready object has no reviewed multiview metric "
                "triangulation"
            )
        base = {
            "source_label": label,
            "aliases": list(dict.fromkeys([label, *policy.get("aliases", [])])),
            "status": status,
            "discovery": {
                "prompt_frame": int(discovery_item.get("prompt_frame", -1)),
                "frame_occurrences": int(discovery_item.get("frame_occurrences", 0)),
                "raw_detection_count": int(discovery_item.get("raw_detection_count", 0)),
            },
            "semantic_observation_count": len(evidence),
            "review": {
                "status": status,
                "reviewer": reviewer,
                "reason": reason,
            },
        }
        if status != "approved":
            objects.append(base)
            continue
        approved_index += 1
        anchor_xyz = (
            tuple(float(value) for value in metric_anchor["point_xyz_m"])
            if metric_anchor is not None
            else _anchor_xyz(evidence)
        )
        anchor = anchor_xyz[:2]
        stand_off = float(policy["search_standoff_m"])
        tolerance = float(policy["alignment_tolerance_m"])
        camera_origins = (
            [
                frame.get("camera_origin_map_m", ())
                for frame in metric_anchor.get("frames", [])
                if isinstance(frame, dict)
            ]
            if metric_anchor is not None
            else []
        )
        camera_origins = [
            origin
            for origin in camera_origins
            if isinstance(origin, (list, tuple)) and len(origin) >= 2
        ]
        preferred_view_bearing_rad = (
            math.atan2(
                float(np.median([origin[1] for origin in camera_origins]))
                - anchor[1],
                float(np.median([origin[0] for origin in camera_origins]))
                - anchor[0],
            )
            if camera_origins
            else None
        )
        camera_forward_offset_m = (
            float(
                metric_anchor.get("camera_calibration", {}).get(
                    "forward_offset_m", 0.0
                )
            )
            if metric_anchor is not None
            else 0.0
        )
        visibility_standoff_m = (
            float(
                np.median(
                    [
                        math.dist(
                            (float(origin[0]), float(origin[1])),
                            anchor,
                        )
                        for origin in camera_origins
                    ]
                )
                + camera_forward_offset_m
            )
            if camera_origins
            else stand_off
        )
        try:
            pose, route = plan_object_approach(
                grid,
                START_POSE,
                anchor,
                stand_off_m=stand_off,
                tolerance_m=tolerance,
                preferred_view_bearing_rad=preferred_view_bearing_rad,
            )
            visibility_pose, visibility_route = plan_object_approach(
                grid,
                START_POSE,
                anchor,
                stand_off_m=visibility_standoff_m,
                tolerance_m=max(tolerance, 0.08),
                preferred_view_bearing_rad=preferred_view_bearing_rad,
            )
        except ValueError as exc:
            objects.append(
                {
                    **base,
                    "status": "rejected",
                    "review": {
                        "status": "rejected",
                        "reviewer": reviewer,
                        "reason": str(exc),
                    },
                }
            )
            continue
        object_id = (
            "scan_"
            + "".join(character if character.isalnum() else "_" for character in label)
            .strip("_")
            + f"_{approved_index:02d}"
        )
        objects.append(
            {
                **base,
                "object_id": object_id,
                "object_class": str(policy.get("object_class", "unknown")),
                "map_position": {
                    "x": anchor[0],
                    "y": anchor[1],
                    "z": anchor_xyz[2],
                    "frame_id": "map",
                    "source": (
                        "reviewed_sam3_mask_multiview_ray_triangulation"
                        if metric_anchor is not None
                        else "lingbot_rgb_only_geometry+sam3.1_mask"
                    ),
                },
                "approach": {
                    "pose": {
                        "x": pose.x,
                        "y": pose.y,
                        "yaw": pose.yaw,
                        "frame_id": "map",
                    },
                    "stand_off_m": stand_off,
                    "visibility_pose": {
                        "x": visibility_pose.x,
                        "y": visibility_pose.y,
                        "yaw": visibility_pose.yaw,
                        "frame_id": "map",
                    },
                    "visibility_standoff_m": visibility_standoff_m,
                    "visibility_path_length_from_survey_start_m": path_length(
                        visibility_route
                    ),
                    "alignment_tolerance_m": tolerance,
                    "planned_path_length_from_survey_start_m": path_length(route),
                    "faces_object_anchor": True,
                    "preferred_view_bearing_rad": preferred_view_bearing_rad,
                    "view_bearing_source": (
                        "reviewed_rgb_survey_visibility_rays"
                        if preferred_view_bearing_rad is not None
                        else "unavailable"
                    ),
                    "footprint_radius_m": ROBOT_RADIUS_M,
                },
                "manipulation_ready": manipulation_ready,
                "metric_triangulation": metric_anchor,
                "track_ids": sorted(
                    {str(item.get("track_id", "")) for item in evidence}
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "reviewed_scan_derived_household_object_catalog",
        "map": {
            "sha256": map_bundle_sha256(map_yaml),
            "yaml": str(map_yaml),
            "frame_id": "map",
            "household_object_set_signature": household_object_set_signature,
        },
        "sources": {
            "discovery": str(discovery_file),
            "semantic_observations": str(semantic_observations),
            "review_policy": str(review_file),
            "triangulated_anchors": (
                str(triangulated_anchors_file)
                if triangulated_anchors_file is not None
                else None
            ),
            "isaac_scene_truth_used": False,
        },
        "objects": objects,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_formal_place_catalog(
    map_yaml: Path,
    semantic_observations: Path,
    alignment_file: Path,
    region_map: Path,
    output_file: Path,
    *,
    reviewer: str = "family_home_scan_map_engineering_review",
    household_object_set_signature: str = "",
) -> dict[str, Any]:
    """Generate each docking pose from its matching scan-derived object anchor."""

    import numpy as np

    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    alignment = json.loads(alignment_file.read_text(encoding="utf-8"))
    _payload, observations = _load_observations(semantic_observations)
    grid = load_ros_grid(map_yaml, robot_radius_m=ROBOT_RADIUS_M)
    regions = np.load(region_map, allow_pickle=False)
    places: list[dict[str, Any]] = []
    for definition in PLACES:
        prompt, evidence = _place_evidence(observations, definition.place_id)
        base = {
            "id": definition.place_id,
            "name": definition.name,
            "aliases": list(definition.aliases),
            "target": {"type": "semantic_instance", "source_id": prompt},
        }
        if not evidence:
            places.append({
                **base,
                "status": "rejected",
                "docking_candidates": [],
                "selected_docking_candidate": "",
                "review": {
                    "status": "rejected",
                    "reviewer": reviewer,
                    "reason": f"no matching SAM3 map-frame evidence for {prompt!r}",
                    "required_action": "extend RGB survey coverage or review the SAM3 prompt frame",
                },
            })
            continue
        anchor = _anchor(evidence)
        try:
            pose = _scan_docking_pose(grid, anchor)
            route = grid.plan((START_POSE.x, START_POSE.y), (pose.x, pose.y))
        except ValueError as exc:
            places.append({
                **base,
                "status": "rejected",
                "docking_candidates": [],
                "selected_docking_candidate": "",
                "review": {
                    "status": "rejected",
                    "reviewer": reviewer,
                    "reason": str(exc),
                    "required_action": "extend/rebuild the scan-derived occupancy",
                },
            })
            continue
        row, col = grid.world_to_cell(pose.x, pose.y)
        region_id = int(regions[row, col])
        candidate_id = f"{definition.place_id}_scan_v1"
        pose_payload = {"x": pose.x, "y": pose.y, "yaw": pose.yaw, "frame_id": "map"}
        places.append({
            **base,
            "status": "approved",
            "entrance_pose": pose_payload,
            "docking_candidates": [{
                "id": candidate_id,
                "pose": pose_payload,
                "checks": {
                    "clearance_m": ROBOT_RADIUS_M,
                    "footprint_radius_m": ROBOT_RADIUS_M,
                    "occupancy_status": "free",
                    "reachable": True,
                    "stand_off_m": math.dist((pose.x, pose.y), anchor),
                    "faces_semantic_anchor": True,
                },
                "review": {
                    "status": "accepted",
                    "reviewer": reviewer,
                    "evidence": [str(map_yaml), str(semantic_observations), str(alignment_file)],
                },
            }],
            "selected_docking_candidate": candidate_id,
            "metadata": {
                "semantic_prompt": prompt,
                "discovered_labels": [prompt],
                "semantic_anchor_xy": [anchor[0], anchor[1]],
                "semantic_observation_count": len(evidence),
                "track_ids": sorted({str(item["track_id"]) for item in evidence}),
                "region_id": region_id,
            },
            "review": {
                "status": "approved",
                "reviewer": reviewer,
                "source": "g1d_rgb_survey+lingbot_rgb_only_geometry+sam3.1_matching_semantics",
                "planned_path_length_m": path_length(route),
            },
        })

    payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-family-home-lingbot-sam3-v1",
            "sha256": map_bundle_sha256(map_yaml),
            "frame_id": "map",
            "source": (
                "lingbot_rgb_only_offline_survey_pose_anchored"
                if alignment.get("artifact_type") == "lingbot_depth_to_metric_survey_pose_anchor"
                else "lingbot_rgb_only_global_sim3"
            ),
            "yaml": str(map_yaml),
            "household_object_set_signature": household_object_set_signature,
        },
        "semantic_evidence": {
            "source": "florence2_category_free_discovery+official_sam3.1_text_video_tracking",
            "artifact": str(semantic_observations),
        },
        "places": places,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "PROMPT_BY_PLACE",
    "PROMPT_ALIASES_BY_PLACE",
    "SEMANTIC_LABELS",
    "build_formal_object_catalog",
    "build_formal_place_catalog",
    "build_scan_semantic_layers",
    "plan_object_approach",
]
