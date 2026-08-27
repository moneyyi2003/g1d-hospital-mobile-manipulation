"""Build outward exploration routes from LingBot occupancy and region maps.

This module intentionally has no Habitat imports.  Region membership, region
adjacency, and the distinction between the start room and outside targets are
derived only from LingBot map artifacts.
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from ..models import Place
from ..place_db import PlaceDatabase
from ..topology import TopologyEdge, TopologyGraph
from .semantic_map import load_ros_occupancy


_OUTSIDE_ALIASES = ("房间外区域", "外部区域", "扫描房间外", "探索房间外")


def _frame_index(value: str) -> int | None:
    stem = Path(value).stem
    token = stem.removeprefix("frame_")
    return int(token) if token.isdigit() else None


def _trajectory_yaw(centres, index: int) -> float:
    before = centres[max(0, index - 1)]
    after = centres[min(len(centres) - 1, index + 1)]
    return math.atan2(float(after[1] - before[1]), float(after[0] - before[0]))


def _snap_region_goal(point, region_id: int, region_map, grid, minimum_clearance: float):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    obstacle_clearance = distance_transform_edt(grid.cells != 100) * grid.resolution
    valid = (
        (region_map == region_id)
        & (grid.cells == 0)
        & (obstacle_clearance >= minimum_clearance)
    )
    rows, cols = np.nonzero(valid)
    if not len(rows):
        raise ConfigurationError(f"Region {region_id} has no safe outward scan goal")
    row, col = _cell((float(point[0]), float(point[1])), grid)
    distances = np.hypot(rows - row, cols - col)
    index = int(distances.argmin())
    return (
        grid.origin_x + (int(cols[index]) + 0.5) * grid.resolution,
        grid.origin_y + (int(rows[index]) + 0.5) * grid.resolution,
    )


def retarget_outward_regions(
    places: PlaceDatabase,
    *,
    map_yaml: str | Path,
    region_map_path: str | Path,
    candidates_path: str | Path,
    topology_start: str,
    minimum_clearance: float = 0.30,
    waypoint_clearance: float = 0.30,
    waypoint_spacing: float = 0.35,
) -> tuple[PlaceDatabase, dict[str, Any]]:
    """Move hallway/dining goals onto RGB-observed outward trajectory points.

    Broad CLIP region masks can label furniture in the living room as a dining
    area.  This refinement uses only LingBot-predicted camera extrinsics and
    RGB object source frames: the hallway goal is placed at the last safe cell
    before the next region, and the dining goal at the farthest observed chair
    frame outside the start region.
    """
    import json
    import numpy as np

    from .rgb_only import _predicted_camera_centres

    grid = load_ros_occupancy(map_yaml)
    region_map = np.load(Path(region_map_path).expanduser().resolve(), allow_pickle=False)
    payload = json.loads(Path(candidates_path).expanduser().resolve().read_text(encoding="utf-8"))
    geometry = payload.get("geometry", {})
    if geometry.get("backend") != "lingbot_map":
        raise ConfigurationError("Outward region refinement requires LingBot-Map geometry")
    prediction_root = Path(str(geometry.get("predictions", ""))).expanduser().resolve()
    prediction_files = sorted(prediction_root.glob("frame_*.npz"))
    alignment = np.asarray(geometry.get("alignment_matrix"), dtype=np.float64)
    if not prediction_files or alignment.shape != (4, 4):
        raise ConfigurationError("Object catalog lacks LingBot trajectory provenance")
    centres = _predicted_camera_centres(prediction_files, alignment)
    start = places.resolve(topology_start).place.entrance_pose
    start_match = _nearest_region((start.x, start.y), region_map, grid)
    if start_match is None:
        raise ConfigurationError("Cannot identify the LingBot start region")
    start_region = start_match[0]
    trajectory_regions = []
    for point in centres:
        match = _nearest_region((float(point[0]), float(point[1])), region_map, grid)
        trajectory_regions.append(match[0] if match is not None and match[1] <= 0.12 else 0)
    outward_regions = [item for item in trajectory_regions if item > 0 and item != start_region]
    if not outward_regions:
        raise ConfigurationError("LingBot trajectory never leaves the start region")
    transition_region = outward_regions[0]
    transition_frames = [
        index for index, region_id in enumerate(trajectory_regions)
        if region_id == transition_region
    ]
    hallway_frame = max(transition_frames)

    chair_frames: list[int] = []
    supporting_chairs: list[str] = []
    supporting_chair_centres: list[tuple[float, float]] = []
    for item in payload.get("instances", []):
        if item.get("semantic_label") != "chair" or not item.get("candidate_poses"):
            continue
        frames = [
            index for source in item.get("source_frames", [])
            if (index := _frame_index(str(source))) is not None
        ]
        outside = [
            index for index in frames
            if index > hallway_frame and trajectory_regions[index] not in {0, start_region}
        ]
        if outside:
            chair_frames.extend(outside)
            supporting_chairs.append(str(item.get("instance_id", "")))
            center = item.get("center_map", {})
            supporting_chair_centres.append((float(center["x"]), float(center["y"])))
    if not chair_frames:
        raise ConfigurationError("No RGB-observed chair evidence exists outside the start room")
    dining_frame = max(chair_frames)
    dining_region = trajectory_regions[dining_frame]
    if dining_region <= 0 or dining_region == transition_region:
        raise ConfigurationError("Dining evidence does not reach a distinct outside region")

    hallway_xy = _snap_region_goal(
        centres[hallway_frame], transition_region, region_map, grid, minimum_clearance
    )
    dining_xy = _snap_region_goal(
        centres[dining_frame], dining_region, region_map, grid, minimum_clearance
    )
    replacements: dict[str, tuple[int, int, tuple[float, float]]] = {}
    for place in places.places:
        label = str(place.metadata.get("semantic_label", ""))
        if label == "hallway":
            replacements[place.place_id] = (transition_region, hallway_frame, hallway_xy)
        elif label == "dining area":
            replacements[place.place_id] = (dining_region, dining_frame, dining_xy)
    if len(replacements) != 2:
        raise ConfigurationError("Outward refinement needs one hallway and one dining region")

    living_frames: dict[str, int] = {}
    for place in places.places:
        if str(place.metadata.get("semantic_label", "")) != "living room":
            continue
        distances = np.hypot(
            centres[:, 0] - place.entrance_pose.x,
            centres[:, 1] - place.entrance_pose.y,
        )
        living_frames[place.place_id] = int(distances.argmin())

    refined = []
    for place in places.places:
        replacement = replacements.get(place.place_id)
        if replacement is None:
            if place.place_id not in living_frames:
                refined.append(place)
                continue
            value = place.to_dict()
            metadata = dict(value["metadata"])
            metadata["goal_refinement"] = {
                "source": "lingbot_predicted_trajectory_nearest_pose",
                "frame": living_frames[place.place_id],
                "region_id": start_region,
                "habitat_ground_truth_used": False,
            }
            value["metadata"] = metadata
            refined.append(Place.from_mapping(value, places.frame_id))
            continue
        region_id, frame_index, (x, y) = replacement
        yaw = _trajectory_yaw(centres, frame_index)
        if (
            place.metadata.get("semantic_label") == "dining area"
            and supporting_chair_centres
        ):
            target_x = sum(item[0] for item in supporting_chair_centres) / len(
                supporting_chair_centres
            )
            target_y = sum(item[1] for item in supporting_chair_centres) / len(
                supporting_chair_centres
            )
            yaw = math.atan2(target_y - y, target_x - x)
        value = place.to_dict()
        value["region"] = f"semantic_region_{region_id}"
        value["entrance_pose"] = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "frame_id": place.entrance_pose.frame_id,
        }
        metadata = dict(value["metadata"])
        metadata["goal_refinement"] = {
            "source": "lingbot_predicted_trajectory+rgb_object_source_frames",
            "frame": frame_index,
            "region_id": region_id,
            "minimum_clearance": minimum_clearance,
            "habitat_ground_truth_used": False,
        }
        if metadata.get("semantic_label") == "dining area":
            metadata["object_evidence"] = supporting_chairs
        value["metadata"] = metadata
        refined.append(Place.from_mapping(value, places.frame_id))

    selected_frames: list[int] = []
    last_point = centres[0]
    for frame_index in range(1, dining_frame + 1):
        point = centres[frame_index]
        if float(np.linalg.norm(point[:2] - last_point[:2])) < waypoint_spacing:
            continue
        region_id = trajectory_regions[frame_index]
        if region_id <= 0:
            continue
        selected_frames.append(frame_index)
        last_point = point
    semantic_frames = set(living_frames.values()) | {hallway_frame, dining_frame}
    selected_frames = [
        frame_index for frame_index in selected_frames if frame_index not in semantic_frames
    ]
    kept_frames: list[int] = []
    last_xy: tuple[float, float] | None = None
    for frame_index in selected_frames:
        region_id = trajectory_regions[frame_index]
        x, y = _snap_region_goal(
            centres[frame_index], region_id, region_map, grid, waypoint_clearance
        )
        if last_xy is not None and math.hypot(x - last_xy[0], y - last_xy[1]) < 0.12:
            continue
        kept_frames.append(frame_index)
        last_xy = (x, y)
        ordinal = len(kept_frames)
        refined.append(
            Place.from_mapping(
                {
                    "id": f"lingbot_scan_{ordinal:02d}",
                    "name": f"RGB扫描航点{ordinal:02d}",
                    "aliases": [f"lingbot_scan_{ordinal:02d}"],
                    "entrance_pose": {
                        "x": x,
                        "y": y,
                        "yaw": _trajectory_yaw(centres, frame_index),
                        "frame_id": places.frame_id,
                    },
                    "region": f"semantic_region_{region_id}",
                    "metadata": {
                        "target_type": "exploration_waypoint",
                        "trajectory_frame": frame_index,
                        "region_id": region_id,
                        "internal": True,
                        "source": "lingbot_predicted_camera_trajectory",
                        "habitat_ground_truth_used": False,
                    },
                },
                places.frame_id,
            )
        )
    selected_frames = kept_frames
    summary = {
        "hallway_frame": hallway_frame,
        "hallway_region_id": transition_region,
        "dining_frame": dining_frame,
        "dining_region_id": dining_region,
        "dining_object_evidence": supporting_chairs,
        "scan_waypoint_frames": selected_frames,
        "scan_waypoint_count": len(selected_frames),
        "source": "lingbot_predicted_trajectory+rgb_object_source_frames",
        "habitat_ground_truth_used": False,
    }
    return PlaceDatabase(refined, places.frame_id), summary


def _region_id(place: Place) -> int | None:
    if place.metadata.get("target_type") != "semantic_region":
        return None
    prefix = "semantic_region_"
    if not place.region.startswith(prefix):
        return None
    try:
        value = int(place.region.removeprefix(prefix))
    except ValueError:
        return None
    return value if value > 0 else None


def _cell(point: tuple[float, float], grid) -> tuple[int, int]:
    return (
        math.floor((point[1] - grid.origin_y) / grid.resolution),
        math.floor((point[0] - grid.origin_x) / grid.resolution),
    )


def _nearest_region(point: tuple[float, float], region_map, grid) -> tuple[int, float] | None:
    import numpy as np

    row, col = _cell(point, grid)
    if 0 <= row < region_map.shape[0] and 0 <= col < region_map.shape[1]:
        direct = int(region_map[row, col])
        if direct > 0:
            return direct, 0.0
    rows, cols = np.nonzero(region_map > 0)
    if not len(rows):
        return None
    distances = np.hypot(rows - row, cols - col)
    index = int(distances.argmin())
    return int(region_map[rows[index], cols[index]]), float(distances[index] * grid.resolution)


def _region_adjacency(region_map, resolution: float, maximum_gap: float):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    ids = sorted(int(item) for item in np.unique(region_map) if int(item) > 0)
    adjacency: dict[int, set[int]] = {region_id: set() for region_id in ids}
    for index, source in enumerate(ids):
        distance = distance_transform_edt(region_map != source) * resolution
        for target in ids[index + 1 :]:
            if float(distance[region_map == target].min()) <= maximum_gap:
                adjacency[source].add(target)
                adjacency[target].add(source)
    return adjacency


def _annotated_place(
    place: Place,
    *,
    assigned_region: int | None,
    start_region: int,
    hops: dict[int, int],
    add_outside_aliases: bool,
) -> Place:
    value = place.to_dict()
    metadata = dict(value["metadata"])
    reachable = assigned_region in hops if assigned_region is not None else False
    metadata["exploration"] = {
        "source": "lingbot_occupancy+semantic_region_map",
        "region_id": assigned_region,
        "start_region_id": start_region,
        "region_hops_from_start": (
            hops.get(assigned_region) if assigned_region is not None else None
        ),
        "outside_start_region": bool(reachable and assigned_region != start_region),
        "reachable_via_region_map": reachable,
        "habitat_ground_truth_used": False,
    }
    value["metadata"] = metadata
    if add_outside_aliases:
        value["aliases"] = list(dict.fromkeys((*value.get("aliases", []), *_OUTSIDE_ALIASES)))
    return Place.from_mapping(value, place.entrance_pose.frame_id)


def build_exploration_topology(
    places: PlaceDatabase,
    *,
    map_yaml: str | Path,
    region_map_path: str | Path,
    topology_start: str,
    maximum_region_gap: float = 0.18,
    maximum_assignment_distance: float = 0.75,
) -> tuple[PlaceDatabase, TopologyGraph, dict[str, Any]]:
    """Route outside goals through adjacent RGB-recognized semantic regions.

    The returned topology is a directed tree rooted at ``topology_start``.
    Objects are attached to their nearest semantic region, so an outside
    object mission first visits each recognized region on the outward path.
    """
    import numpy as np

    if maximum_region_gap <= 0 or maximum_assignment_distance <= 0:
        raise ConfigurationError("Exploration distance thresholds must be positive")
    grid = load_ros_occupancy(map_yaml)
    region_map = np.load(Path(region_map_path).expanduser().resolve(), allow_pickle=False)
    if region_map.shape != grid.cells.shape:
        raise ConfigurationError(
            f"Region map shape {region_map.shape} does not match occupancy {grid.cells.shape}"
        )
    try:
        start_place = places.resolve(topology_start).place
    except Exception as exc:
        raise ConfigurationError(f"Unknown exploration start: {topology_start}") from exc
    semantic_places = {
        region_id: place
        for place in places.places
        if (region_id := _region_id(place)) is not None
    }
    if not semantic_places:
        raise ConfigurationError("Exploration needs semantic-region places")
    start_match = _nearest_region(
        (start_place.entrance_pose.x, start_place.entrance_pose.y), region_map, grid
    )
    if start_match is None or start_match[1] > maximum_assignment_distance:
        raise ConfigurationError("Start pose is not supported by the LingBot semantic region map")
    start_region = start_match[0]
    if start_region not in semantic_places:
        raise ConfigurationError(f"Start region {start_region} has no semantic-region place")

    adjacency = _region_adjacency(region_map, grid.resolution, maximum_region_gap)
    parent: dict[int, int | None] = {start_region: None}
    hops = {start_region: 0}
    frontier = deque([start_region])
    while frontier:
        current = frontier.popleft()
        for neighbour in sorted(adjacency.get(current, ())):
            if neighbour in parent or neighbour not in semantic_places:
                continue
            parent[neighbour] = current
            hops[neighbour] = hops[current] + 1
            frontier.append(neighbour)

    assignments: dict[str, int | None] = {topology_start: start_region}
    for place in places.places:
        explicit = _region_id(place)
        if explicit is not None:
            assignments[place.place_id] = explicit
            continue
        match = _nearest_region(
            (place.entrance_pose.x, place.entrance_pose.y), region_map, grid
        )
        assignments[place.place_id] = (
            match[0] if match is not None and match[1] <= maximum_assignment_distance else None
        )

    outside_regions = sorted(
        (region_id for region_id in hops if region_id != start_region),
        key=lambda region_id: (hops[region_id], region_id),
    )
    exploration_goal = outside_regions[-1] if outside_regions else None
    annotated = [
        _annotated_place(
            place,
            assigned_region=assignments.get(place.place_id),
            start_region=start_region,
            hops=hops,
            add_outside_aliases=_region_id(place) == exploration_goal,
        )
        for place in places.places
    ]
    expanded_places = PlaceDatabase(annotated, places.frame_id)

    scan_chain = [
        place for place in annotated
        if place.metadata.get("target_type") == "exploration_waypoint"
    ]
    semantic_chain = [
        place for place in annotated
        if _region_id(place) in hops
        and place.metadata.get("goal_refinement", {}).get("frame") is not None
    ]
    edges: list[TopologyEdge] = []
    if scan_chain and len(semantic_chain) == len(semantic_places):
        ordered_scan = sorted(
            scan_chain, key=lambda place: int(place.metadata["trajectory_frame"])
        )
        cursor = topology_start
        for place in ordered_scan:
            edges.append(TopologyEdge(cursor, place.place_id))
            cursor = place.place_id
        for place in semantic_chain:
            goal_frame = int(place.metadata["goal_refinement"]["frame"])
            anchors = [
                waypoint for waypoint in ordered_scan
                if int(waypoint.metadata["trajectory_frame"]) <= goal_frame
            ]
            anchor = anchors[-1].place_id if anchors else topology_start
            edges.append(TopologyEdge(anchor, place.place_id))
    else:
        edges.append(TopologyEdge(topology_start, semantic_places[start_region].place_id))
        for region_id in outside_regions:
            parent_id = parent[region_id]
            if parent_id is not None:
                edges.append(
                    TopologyEdge(
                        semantic_places[parent_id].place_id,
                        semantic_places[region_id].place_id,
                    )
                )
    for place in annotated:
        if (
            place.place_id == topology_start
            or _region_id(place) is not None
            or place.metadata.get("target_type") == "exploration_waypoint"
        ):
            continue
        assigned = assignments.get(place.place_id)
        if assigned in hops:
            edges.append(TopologyEdge(semantic_places[assigned].place_id, place.place_id))

    topology = TopologyGraph(
        tuple(place.place_id for place in annotated), tuple(edges), expanded_places
    )
    outside_place_ids = [
        place.place_id
        for place in annotated
        if place.metadata.get("exploration", {}).get("outside_start_region")
    ]
    summary: dict[str, Any] = {
        "source": "lingbot_occupancy+semantic_region_map",
        "start_region_id": start_region,
        "start_region_place_id": semantic_places[start_region].place_id,
        "outside_region_ids": outside_regions,
        "outside_region_place_ids": [semantic_places[item].place_id for item in outside_regions],
        "outside_place_ids": outside_place_ids,
        "region_hops": {str(key): value for key, value in sorted(hops.items())},
        "exploration_goal_place_id": (
            semantic_places[exploration_goal].place_id if exploration_goal is not None else None
        ),
        "scan_waypoint_count": len(scan_chain),
        "habitat_ground_truth_used": False,
    }
    return expanded_places, topology, summary


__all__ = ["build_exploration_topology", "retarget_outward_regions"]
