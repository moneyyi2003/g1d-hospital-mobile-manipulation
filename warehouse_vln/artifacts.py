"""Dependency-light warehouse collision map and semantic-place artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from simple_room_vln.core import GridMap, Place, Pose2D


WAREHOUSE_SCENE_URL = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.1/Isaac/Environments/Simple_Warehouse/"
    "warehouse_multiple_shelves.usd"
)
MAP_BOUNDS = (-11.8, -17.8, 11.8, 20.5)
RESOLUTION_M = 0.10
ROBOT_RADIUS_M = 0.42
ROBOT_VERTICAL_RANGE_M = (0.08, 1.55)

WAREHOUSE_START = Pose2D(-5.0, -10.0, 0.0)


@dataclass(frozen=True)
class CollisionBounds:
    """World-axis-aligned bounds for one composed USD collision prim."""

    path: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = (*self.minimum, *self.maximum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite collision bounds: {self.path}")
        if any(left > right for left, right in zip(self.minimum, self.maximum)):
            raise ValueError(f"inverted collision bounds: {self.path}")

    def overlaps_robot_height(self) -> bool:
        minimum_z, maximum_z = ROBOT_VERTICAL_RANGE_M
        return self.maximum[2] >= minimum_z and self.minimum[2] <= maximum_z

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
        }


def _cell_range(
    minimum: float,
    maximum: float,
    *,
    origin: float,
    resolution: float,
    count: int,
) -> range:
    first = max(0, int(math.floor((minimum - origin) / resolution)))
    last = min(count - 1, int(math.floor((maximum - origin) / resolution)))
    if first > last:
        return range(0)
    return range(first, last + 1)


def build_collision_grid(
    collision_bounds: Iterable[CollisionBounds],
    *,
    map_bounds: tuple[float, float, float, float] = MAP_BOUNDS,
    resolution_m: float = RESOLUTION_M,
    robot_radius_m: float = ROBOT_RADIUS_M,
) -> tuple[GridMap, tuple[CollisionBounds, ...]]:
    """Rasterize collision AABBs that intersect the G1-D body height.

    This is a simulator-ground-truth bootstrap, not a replacement for the
    project's LingBot RGB-only occupancy map.  It is deliberately conservative:
    each collider is projected to XY and inflated by the G1-D footprint radius.
    """

    if resolution_m <= 0.0 or robot_radius_m < 0.0:
        raise ValueError("resolution must be positive and radius non-negative")
    xmin, ymin, xmax, ymax = map_bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("invalid map bounds")
    width = int(math.ceil((xmax - xmin) / resolution_m))
    height = int(math.ceil((ymax - ymin) / resolution_m))
    rows = [[True] * width for _ in range(height)]

    boundary_cells = int(math.ceil(robot_radius_m / resolution_m))
    for row in range(height):
        for col in range(width):
            if (
                row < boundary_cells
                or row >= height - boundary_cells
                or col < boundary_cells
                or col >= width - boundary_cells
            ):
                rows[row][col] = False

    used: list[CollisionBounds] = []
    for bounds in collision_bounds:
        if not bounds.overlaps_robot_height():
            continue
        inflated_xmin = bounds.minimum[0] - robot_radius_m
        inflated_ymin = bounds.minimum[1] - robot_radius_m
        inflated_xmax = bounds.maximum[0] + robot_radius_m
        inflated_ymax = bounds.maximum[1] + robot_radius_m
        if (
            inflated_xmax < xmin
            or inflated_xmin > xmax
            or inflated_ymax < ymin
            or inflated_ymin > ymax
        ):
            continue
        used.append(bounds)
        rows_to_block = _cell_range(
            inflated_ymin,
            inflated_ymax,
            origin=ymin,
            resolution=resolution_m,
            count=height,
        )
        cols_to_block = _cell_range(
            inflated_xmin,
            inflated_xmax,
            origin=xmin,
            resolution=resolution_m,
            count=width,
        )
        for row in rows_to_block:
            center_y = ymin + (row + 0.5) * resolution_m
            if not inflated_ymin <= center_y <= inflated_ymax:
                continue
            for col in cols_to_block:
                center_x = xmin + (col + 0.5) * resolution_m
                if inflated_xmin <= center_x <= inflated_xmax:
                    rows[row][col] = False

    return (
        GridMap(
            rows,
            resolution=resolution_m,
            origin_x=xmin,
            origin_y=ymin,
        ),
        tuple(used),
    )


def snap_pose_to_free(
    grid: GridMap,
    pose: Pose2D,
    *,
    maximum_distance_m: float = 1.0,
) -> Pose2D:
    """Snap a measured semantic pose to the nearest footprint-safe map cell."""

    requested = grid.world_to_cell(pose.x, pose.y)
    if grid.is_free(requested):
        return pose
    radius = int(math.ceil(maximum_distance_m / grid.resolution))
    candidates: list[tuple[float, int, int]] = []
    for row in range(requested[0] - radius, requested[0] + radius + 1):
        for col in range(requested[1] - radius, requested[1] + radius + 1):
            cell = (row, col)
            if not grid.is_free(cell):
                continue
            x, y = grid.cell_to_world(cell)
            distance = math.dist((pose.x, pose.y), (x, y))
            if distance <= maximum_distance_m:
                candidates.append((distance, row, col))
    if not candidates:
        raise ValueError(
            f"no free cell within {maximum_distance_m:.2f} m of "
            f"({pose.x:.2f}, {pose.y:.2f})"
        )
    _, row, col = min(candidates)
    x, y = grid.cell_to_world((row, col))
    return Pose2D(x, y, pose.yaw)


def _requested_places() -> tuple[Place, ...]:
    return (
        Place(
            "east_shelf_aisle",
            "东侧货架通道",
            (
                "东侧货架",
                "东边货架",
                "东侧通道",
                "east shelf",
                "east aisle",
            ),
            Pose2D(4.0, 9.0, math.pi / 2.0),
        ),
        Place(
            "west_shelf_aisle",
            "西侧货架通道",
            (
                "西侧货架",
                "西边货架",
                "西侧通道",
                "west shelf",
                "west aisle",
            ),
            Pose2D(-5.0, 9.0, math.pi / 2.0),
        ),
        Place(
            "loading_zone",
            "装卸区",
            (
                "装货区",
                "卸货区",
                "仓库入口",
                "loading zone",
                "loading area",
            ),
            Pose2D(4.0, -10.0, 0.0),
        ),
    )


def _serialize_grid(
    grid: GridMap,
    *,
    source_collision_count: int,
    used_collision_count: int,
) -> dict:
    return {
        "schema_version": 1,
        "source": "isaac_collision_aabb_bootstrap",
        "truth_boundary": (
            "Isaac AABB bootstrap uses reviewed shelf and pallet-bin collision "
            "roots plus the scene boundary. Wall/pillar aggregates with openings "
            "are excluded because their AABBs erase traversable gaps; replace "
            "this map with aligned LingBot RGB-only occupancy before formal "
            "navigation."
        ),
        "frame_id": "map",
        "scene": "MobileManiBench/warehouse_multiple_shelves",
        "scene_url": WAREHOUSE_SCENE_URL,
        "resolution_m": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y],
        "width": grid.width,
        "height": grid.height,
        "robot_radius_m": ROBOT_RADIUS_M,
        "robot_vertical_range_m": list(ROBOT_VERTICAL_RANGE_M),
        "source_collision_count": source_collision_count,
        "used_collision_count": used_collision_count,
        "obstacle_selection": [
            "Shelf_*",
            "PalletBin_*",
        ],
        "rows": ["".join("." if cell else "#" for cell in row) for row in grid.free],
    }


def build_bootstrap_artifacts(
    output_dir: Path,
    collision_bounds: Sequence[CollisionBounds],
) -> tuple[GridMap, list[Place], Pose2D, tuple[CollisionBounds, ...]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, used = build_collision_grid(collision_bounds)
    start = snap_pose_to_free(grid, WAREHOUSE_START)
    places = [
        Place(
            item.place_id,
            item.name,
            item.aliases,
            snap_pose_to_free(grid, item.pose),
            item.status,
        )
        for item in _requested_places()
    ]
    map_payload = _serialize_grid(
        grid,
        source_collision_count=len(collision_bounds),
        used_collision_count=len(used),
    )
    map_bytes = json.dumps(
        map_payload, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    place_payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-mobilemanibench-warehouse-bootstrap-v1",
            "sha256": hashlib.sha256(map_bytes).hexdigest(),
            "frame_id": "map",
            "source": "isaac_collision_aabb_bootstrap",
        },
        "places": [
            {
                "id": place.place_id,
                "name": place.name,
                "aliases": list(place.aliases),
                "status": place.status,
                "entrance_pose": {
                    "x": place.pose.x,
                    "y": place.pose.y,
                    "yaw": place.pose.yaw,
                    "frame_id": "map",
                },
                "target": {
                    "type": "semantic_region",
                    "source_id": place.place_id,
                },
                "review": {
                    "status": "accepted_for_bootstrap_demo",
                    "source": "MobileManiBench scene collision bounds",
                },
            }
            for place in places
        ],
    }
    (output_dir / "bootstrap_occupancy.json").write_text(
        json.dumps(map_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "places.json").write_text(
        json.dumps(place_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "semantic_prompts.txt").write_text(
        "warehouse shelf\npallet bin\nloading zone\neast aisle\nwest aisle\n",
        encoding="utf-8",
    )
    return grid, places, start, used


def build_survey_path(
    grid: GridMap,
    start: Pose2D,
    places: Sequence[Place],
) -> list[tuple[float, float]]:
    """Visit both long shelf aisles and return to the start."""

    by_id = {place.place_id: place for place in places}
    waypoints = [
        (start.x, start.y),
        (
            by_id["west_shelf_aisle"].pose.x,
            by_id["west_shelf_aisle"].pose.y,
        ),
        (start.x, start.y),
        (
            by_id["loading_zone"].pose.x,
            by_id["loading_zone"].pose.y,
        ),
        (
            by_id["east_shelf_aisle"].pose.x,
            by_id["east_shelf_aisle"].pose.y,
        ),
        (
            by_id["loading_zone"].pose.x,
            by_id["loading_zone"].pose.y,
        ),
        (start.x, start.y),
    ]
    combined = [waypoints[0]]
    for left, right in zip(waypoints, waypoints[1:]):
        segment = grid.plan(left, right)
        combined.extend(segment[1:])
    return combined


__all__ = [
    "CollisionBounds",
    "MAP_BOUNDS",
    "RESOLUTION_M",
    "ROBOT_RADIUS_M",
    "ROBOT_VERTICAL_RANGE_M",
    "WAREHOUSE_SCENE_URL",
    "WAREHOUSE_START",
    "build_bootstrap_artifacts",
    "build_collision_grid",
    "build_survey_path",
    "snap_pose_to_free",
]
