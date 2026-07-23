"""Generate and rank isolated multi-candidate docking poses for Hospital chairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from simple_room_vln.artifacts import load_ros_grid
from simple_room_vln.core import GridMap, Pose2D, path_length, wrap_angle

from .artifacts import HOSPITAL_START, ROBOT_RADIUS_M, WAITING_AREA_POSE


@dataclass(frozen=True)
class ChairInstance:
    instance_id: str
    prim_path: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)


@dataclass(frozen=True)
class DockingCandidate:
    candidate_id: str
    chair_instance_id: str
    side: str
    pose: Pose2D
    occupancy_status: str
    reachable: bool
    clearance_m: float
    path_length_m: float | None
    heading_error_rad: float
    score: float | None
    rejection_reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.rejection_reasons

    def to_dict(self) -> dict:
        value = asdict(self)
        value["pose"] = {"x": self.pose.x, "y": self.pose.y, "yaw": self.pose.yaw}
        value["eligible"] = self.eligible
        return value


# Measured read-only from Assets/room/IsaacSim/Hospital.usd on 2026-07-23.
# The extraction command and exact prim paths are documented in the runbook.
WAITING_CHAIR_INSTANCES = (
    ChairInstance(
        "chair_02a4",
        "/World/hospital/SM_Chair_02a4",
        -8.743,
        -0.078,
        -6.387,
        0.663,
    ),
    ChairInstance(
        "chair_02a5",
        "/World/hospital/SM_Chair_02a5",
        -8.743,
        0.671,
        -6.387,
        1.412,
    ),
    ChairInstance(
        "chair_02a6",
        "/World/hospital/SM_Chair_02a6",
        -6.295,
        -0.078,
        -3.939,
        0.663,
    ),
    ChairInstance(
        "chair_02a7",
        "/World/hospital/SM_Chair_02a7",
        -6.295,
        0.671,
        -3.939,
        1.412,
    ),
)


def _map_digest(map_yaml: Path) -> str:
    image_name = ""
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


def generate_candidates(
    chairs: Sequence[ChairInstance],
    *,
    standoff_m: float = 0.80,
) -> list[tuple[str, ChairInstance, str, Pose2D]]:
    if standoff_m <= ROBOT_RADIUS_M:
        raise ValueError("chair standoff must exceed the robot footprint radius")
    result = []
    for chair in chairs:
        center_x, _ = chair.center
        result.extend(
            (
                (
                    f"{chair.instance_id}_south",
                    chair,
                    "south",
                    Pose2D(center_x, chair.min_y - standoff_m, math.pi / 2.0),
                ),
                (
                    f"{chair.instance_id}_north",
                    chair,
                    "north",
                    Pose2D(center_x, chair.max_y + standoff_m, -math.pi / 2.0),
                ),
            )
        )
    return result


def _clearance(grid: GridMap, pose: Pose2D, *, search_radius_m: float = 2.0) -> float:
    row, col = grid.world_to_cell(pose.x, pose.y)
    radius = int(math.ceil(search_radius_m / grid.resolution))
    best = search_radius_m
    for rr in range(row - radius, row + radius + 1):
        for cc in range(col - radius, col + radius + 1):
            if grid.is_free((rr, cc)):
                continue
            x, y = grid.cell_to_world((rr, cc))
            best = min(best, math.dist((pose.x, pose.y), (x, y)))
    return best


def evaluate_candidates(
    map_yaml: Path,
    chairs: Sequence[ChairInstance] = WAITING_CHAIR_INSTANCES,
    *,
    start: Pose2D = HOSPITAL_START,
    standoff_m: float = 0.80,
    blocked_candidate_ids: Iterable[str] = (),
) -> list[DockingCandidate]:
    map_yaml = map_yaml.resolve()
    raw_grid = load_ros_grid(map_yaml, robot_radius_m=0.0)
    navigation_grid = load_ros_grid(map_yaml, robot_radius_m=ROBOT_RADIUS_M)
    dynamic_blocked = set(blocked_candidate_ids)
    result: list[DockingCandidate] = []
    for candidate_id, chair, side, pose in generate_candidates(
        chairs, standoff_m=standoff_m
    ):
        reasons = []
        raw_free = raw_grid.is_free(raw_grid.world_to_cell(pose.x, pose.y))
        footprint_free = navigation_grid.is_free(
            navigation_grid.world_to_cell(pose.x, pose.y)
        )
        if not raw_free:
            reasons.append("occupied_or_unknown")
        elif not footprint_free:
            reasons.append("insufficient_footprint_clearance")
        if candidate_id in dynamic_blocked:
            reasons.append("dynamic_blocked")

        chair_x, chair_y = chair.center
        desired_yaw = math.atan2(chair_y - pose.y, chair_x - pose.x)
        heading_error = abs(wrap_angle(desired_yaw - pose.yaw))
        if heading_error > math.radians(10.0):
            reasons.append("chair_facing_error")

        route = None
        if footprint_free:
            try:
                route = navigation_grid.plan(
                    (start.x, start.y),
                    (pose.x, pose.y),
                )
            except ValueError:
                reasons.append("unreachable")
        clearance = _clearance(raw_grid, pose)
        route_length = path_length(route) if route is not None else None
        score = None
        if not reasons and route_length is not None:
            score = (
                2.0 * min(clearance, 1.2)
                - 0.12 * route_length
                - 0.5 * heading_error
            )
        result.append(
            DockingCandidate(
                candidate_id=candidate_id,
                chair_instance_id=chair.instance_id,
                side=side,
                pose=pose,
                occupancy_status="free" if raw_free else "blocked_or_unknown",
                reachable=route is not None,
                clearance_m=clearance,
                path_length_m=route_length,
                heading_error_rad=heading_error,
                score=score,
                rejection_reasons=tuple(reasons),
            )
        )
    return sorted(
        result,
        key=lambda item: (
            not item.eligible,
            -(item.score if item.score is not None else -math.inf),
            item.candidate_id,
        ),
    )


def select_candidate(
    candidates: Sequence[DockingCandidate],
    *,
    blocked_candidate_ids: Iterable[str] = (),
) -> DockingCandidate:
    blocked = set(blocked_candidate_ids)
    for candidate in candidates:
        if candidate.eligible and candidate.candidate_id not in blocked:
            return candidate
    raise ValueError("no eligible Hospital docking candidate remains")


def build_waiting_area_artifact(
    map_yaml: Path,
    output_file: Path,
    *,
    blocked_candidate_ids: Iterable[str] = (),
) -> dict:
    map_yaml = map_yaml.resolve()
    output_file = output_file.resolve()
    candidates = evaluate_candidates(
        map_yaml,
        blocked_candidate_ids=blocked_candidate_ids,
    )
    selected = select_candidate(candidates)
    payload = {
        "schema_version": 1,
        "artifact_type": "hospital_experimental_dynamic_docking",
        "activation": "explicit_opt_in_only",
        "map": {
            "yaml": str(map_yaml),
            "sha256": _map_digest(map_yaml),
        },
        "place_id": "waiting_area",
        "source": {
            "type": "reviewed_usd_chair_bounds",
            "chairs": [asdict(item) for item in WAITING_CHAIR_INSTANCES],
        },
        "policy": {
            "robot_radius_m": ROBOT_RADIUS_M,
            "standoff_m": 0.80,
            "score": "2*min(clearance,1.2)-0.12*path_length-0.5*heading_error",
            "dynamic_occupancy": "blocked_candidate_ids",
        },
        "candidates": [item.to_dict() for item in candidates],
        "selected_candidate_id": selected.candidate_id,
        "selected_pose": {
            "x": selected.pose.x,
            "y": selected.pose.y,
            "yaw": selected.pose.yaw,
        },
        "fallback": {
            "candidate_id": "waiting_area_reviewed_v1",
            "pose": {
                "x": WAITING_AREA_POSE.x,
                "y": WAITING_AREA_POSE.y,
                "yaw": WAITING_AREA_POSE.yaw,
            },
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "ChairInstance",
    "DockingCandidate",
    "WAITING_CHAIR_INSTANCES",
    "build_waiting_area_artifact",
    "evaluate_candidates",
    "generate_candidates",
    "select_candidate",
]
