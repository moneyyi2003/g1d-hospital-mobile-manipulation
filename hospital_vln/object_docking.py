"""Object-level docking poses for the isolated Hospital manipulation demo."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from simple_room_vln.artifacts import load_ros_grid
from simple_room_vln.core import GridMap, Pose2D, path_length, wrap_angle

from .artifacts import HOSPITAL_START, ROBOT_RADIUS_M


_DISTANCE_PATTERNS = (
    re.compile(r"(?:前|前方|前面)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:米|m)\b", re.I),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:米|m)\s*(?:前|前方|前面)", re.I),
)


@dataclass(frozen=True)
class ObjectTarget:
    object_id: str
    name: str
    aliases: tuple[str, ...]
    x: float
    y: float
    z: float
    interaction_face_yaw: float
    size_m: float


@dataclass(frozen=True)
class ObjectDockingPlan:
    target: ObjectTarget
    requested_standoff_m: float
    docking_pose: Pose2D
    path: tuple[tuple[float, float], ...]
    path_length_m: float
    object_distance_m: float
    facing_error_rad: float

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "artifact_type": "hospital_object_docking_demo_plan",
            "activation": "isolated_demo_only",
            "target": {
                "object_id": self.target.object_id,
                "name": self.target.name,
                "position": {
                    "x": self.target.x,
                    "y": self.target.y,
                    "z": self.target.z,
                },
                "interaction_face_yaw": self.target.interaction_face_yaw,
                "size_m": self.target.size_m,
            },
            "constraint": {
                "relation": "in_front_of_interaction_face",
                "requested_standoff_m": self.requested_standoff_m,
                "actual_object_distance_m": self.object_distance_m,
                "facing_error_rad": self.facing_error_rad,
            },
            "docking_pose": {
                "x": self.docking_pose.x,
                "y": self.docking_pose.y,
                "yaw": self.docking_pose.yaw,
            },
            "path_length_m": self.path_length_m,
            "path": [{"x": x, "y": y} for x, y in self.path],
        }


def load_object_targets(path: Path) -> list[ObjectTarget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("activation") != "isolated_demo_only":
        raise ValueError("object target catalog must be isolated_demo_only")
    result = []
    for value in payload.get("objects", []):
        position = value["position"]
        result.append(
            ObjectTarget(
                object_id=str(value["id"]),
                name=str(value["name"]),
                aliases=tuple(str(item) for item in value.get("aliases", [])),
                x=float(position["x"]),
                y=float(position["y"]),
                z=float(position["z"]),
                interaction_face_yaw=float(value["interaction_face_yaw"]),
                size_m=float(value.get("size_m", 0.10)),
            )
        )
    if not result:
        raise ValueError("object target catalog is empty")
    return result


def resolve_object(command: str, targets: list[ObjectTarget]) -> ObjectTarget:
    normalized = command.casefold().strip()
    matches = [
        target
        for target in targets
        if any(alias.casefold() in normalized for alias in (target.object_id, target.name, *target.aliases))
    ]
    if len(matches) != 1:
        raise ValueError(f"object command must resolve to exactly one target, got {len(matches)}")
    return matches[0]


def parse_standoff(command: str, *, default_m: float = 0.80) -> float:
    for pattern in _DISTANCE_PATTERNS:
        match = pattern.search(command)
        if match:
            return float(match.group(1))
    return default_m


def compute_docking_pose(target: ObjectTarget, standoff_m: float) -> Pose2D:
    minimum = ROBOT_RADIUS_M + target.size_m / 2.0 + 0.05
    if not math.isfinite(standoff_m) or standoff_m < minimum:
        raise ValueError(
            f"standoff {standoff_m:.3f} m is unsafe; minimum is {minimum:.3f} m"
        )
    if standoff_m > 2.0:
        raise ValueError("standoff exceeds the 2.0 m manipulation-demo limit")
    x = target.x + standoff_m * math.cos(target.interaction_face_yaw)
    y = target.y + standoff_m * math.sin(target.interaction_face_yaw)
    yaw = wrap_angle(target.interaction_face_yaw + math.pi)
    return Pose2D(x, y, yaw)


def build_object_docking_plan(
    map_yaml: Path,
    target: ObjectTarget,
    standoff_m: float,
    *,
    start: Pose2D = HOSPITAL_START,
) -> ObjectDockingPlan:
    grid: GridMap = load_ros_grid(map_yaml, robot_radius_m=ROBOT_RADIUS_M)
    docking_pose = compute_docking_pose(target, standoff_m)
    cell = grid.world_to_cell(docking_pose.x, docking_pose.y)
    if not grid.is_free(cell):
        raise ValueError("requested object docking pose lacks robot-footprint clearance")
    path = grid.plan((start.x, start.y), (docking_pose.x, docking_pose.y))
    object_distance = math.dist(
        (docking_pose.x, docking_pose.y),
        (target.x, target.y),
    )
    desired_yaw = math.atan2(target.y - docking_pose.y, target.x - docking_pose.x)
    facing_error = abs(wrap_angle(desired_yaw - docking_pose.yaw))
    return ObjectDockingPlan(
        target=target,
        requested_standoff_m=standoff_m,
        docking_pose=docking_pose,
        path=tuple(path),
        path_length_m=path_length(path),
        object_distance_m=object_distance,
        facing_error_rad=facing_error,
    )


__all__ = [
    "ObjectDockingPlan",
    "ObjectTarget",
    "build_object_docking_plan",
    "compute_docking_pose",
    "load_object_targets",
    "parse_standoff",
    "resolve_object",
]
