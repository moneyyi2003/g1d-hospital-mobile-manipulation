"""Dependency-light layout, occupancy and places for the family-home scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from simple_room_vln.core import GridMap, Place, Pose2D


MAP_BOUNDS = (-4.20, -3.10, 4.20, 4.55)
RESOLUTION_M = 0.10
ROBOT_RADIUS_M = 0.40
START_POSE = Pose2D(0.0, 0.0, 0.0)
SCENE_NAME = "G1-D multi-zone family home"


@dataclass(frozen=True)
class HomeFixture:
    fixture_id: str
    category: str
    center_xy: tuple[float, float]
    size_xyz: tuple[float, float, float]
    color_rgb: tuple[float, float, float]

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        cx, cy = self.center_xy
        sx, sy, _ = self.size_xyz
        return (cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0)


@dataclass(frozen=True)
class HomeRegion:
    region_id: str
    name: str
    bounds_xy: tuple[float, float, float, float]
    color_rgb: tuple[int, int, int]


# These measured SimpleRoom/SofaTablePlant bounds remain part of the scene.
BASE_OBSTACLES = (
    (-3.205, 0.985, -2.246, 2.981),  # sofa
    (-4.061, 1.605, -3.260, 2.408),  # coffee table
    (2.965, -1.274, 4.048, 1.274),   # existing kitchen-side table
)

# Added fixtures make four recognizable household zones while preserving
# wheelchair-scale door openings for the 0.8 m-diameter G1-D footprint.
HOME_FIXTURES = (
    HomeFixture(
        "bed",
        "bed",
        (-2.75, -2.18),
        (1.75, 1.25, 0.55),
        (0.28, 0.48, 0.82),
    ),
    HomeFixture(
        "bedroom_partition",
        "wall",
        (-2.85, -1.05),
        (2.70, 0.12, 2.35),
        (0.82, 0.80, 0.74),
    ),
    HomeFixture(
        "living_kitchen_partition_south",
        "wall",
        (0.72, 1.35),
        (0.12, 0.70, 2.35),
        (0.82, 0.80, 0.74),
    ),
    HomeFixture(
        "living_kitchen_partition_north",
        "wall",
        (0.72, 4.12),
        (0.12, 0.86, 2.35),
        (0.82, 0.80, 0.74),
    ),
    HomeFixture(
        "dining_table",
        "dining_table",
        (2.05, 3.05),
        (1.35, 0.82, 0.76),
        (0.48, 0.27, 0.12),
    ),
    HomeFixture(
        "kitchen_counter",
        "kitchen_counter",
        (3.60, 3.82),
        (0.62, 1.15, 0.92),
        (0.72, 0.72, 0.68),
    ),
    HomeFixture(
        "media_cabinet",
        "cabinet",
        (-0.20, 3.80),
        (0.65, 0.95, 0.82),
        (0.30, 0.22, 0.16),
    ),
)

# Regions are a reviewed UI/semantic overlay, not an occupancy source. The
# dining/kitchen split follows the furniture clusters on the right side.
HOME_REGIONS = (
    HomeRegion("bedroom", "卧室", (-4.20, -3.10, 0.72, -1.05), (95, 132, 255)),
    HomeRegion("living_room", "客厅", (-4.20, -1.05, 0.72, 4.55), (61, 214, 157)),
    HomeRegion("dining_area", "餐区", (0.72, 1.75, 2.90, 4.55), (247, 183, 49)),
    HomeRegion("kitchen", "厨房", (2.90, 1.75, 4.20, 4.55), (239, 111, 108)),
    HomeRegion("transition", "通行区", (0.72, -3.10, 4.20, 1.75), (131, 145, 161)),
)


PLACES = (
    Place(
        "living_room_sofa",
        "客厅沙发旁",
        ("客厅", "沙发", "沙发旁", "living room", "sofa", "couch"),
        Pose2D(-2.72, 0.27, math.pi / 2.0),
    ),
    Place(
        "bedroom_bed",
        "卧室床边",
        ("卧室", "床边", "床旁", "bedroom", "bed"),
        Pose2D(-0.95, -2.18, math.pi),
    ),
    Place(
        "dining_area",
        "餐桌旁",
        ("餐厅", "餐桌", "吃饭的地方", "dining room", "dining table"),
        Pose2D(2.05, 2.08, math.pi / 2.0),
    ),
    Place(
        "kitchen_counter",
        "厨房操作台",
        ("厨房", "操作台", "厨房台面", "kitchen", "counter"),
        Pose2D(3.60, 2.30, math.pi / 2.0),
    ),
)

# Extra destinations for the runtime web demo.  They are kept out of PLACES
# so rebuilding the immutable formal SAM3 place catalog still uses exactly
# the four original semantic prompts.
DEMO_PLACES = (
    Place(
        "living_room_center",
        "客厅中央",
        ("客厅中央", "客厅中间", "活动区", "living room center"),
        Pose2D(-0.55, 0.35, math.pi),
    ),
    Place(
        "media_cabinet_front",
        "电视柜前",
        ("电视柜", "电视旁", "媒体柜", "TV cabinet", "media cabinet"),
        Pose2D(-0.20, 2.55, math.pi / 2.0),
    ),
    Place(
        "dining_table_south",
        "餐桌南侧",
        ("餐桌南侧", "餐桌另一边", "餐桌前", "south dining table"),
        Pose2D(1.35, 2.10, math.pi / 2.0),
    ),
    Place(
        "kitchen_entrance",
        "厨房入口",
        ("厨房门口", "厨房入口", "厨房外", "kitchen entrance"),
        Pose2D(2.35, 1.20, math.pi / 2.0),
    ),
    Place(
        "bedroom_entrance",
        "卧室入口",
        ("卧室门口", "卧室入口", "bedroom entrance"),
        Pose2D(-1.10, -1.35, math.pi),
    ),
)

ALL_DEMO_PLACES = PLACES + DEMO_PLACES


def _inflate(
    rectangle: tuple[float, float, float, float], radius: float
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = rectangle
    return xmin - radius, ymin - radius, xmax + radius, ymax + radius


def _inside(
    x: float, y: float, rectangle: tuple[float, float, float, float]
) -> bool:
    xmin, ymin, xmax, ymax = rectangle
    return xmin <= x <= xmax and ymin <= y <= ymax


def build_grid() -> GridMap:
    """Build the explicit bootstrap grid from reviewed fixture bounds."""

    xmin, ymin, xmax, ymax = MAP_BOUNDS
    width = int(math.ceil((xmax - xmin) / RESOLUTION_M))
    height = int(math.ceil((ymax - ymin) / RESOLUTION_M))
    obstacles = [
        _inflate(item, ROBOT_RADIUS_M)
        for item in (*BASE_OBSTACLES, *(item.bounds_xy for item in HOME_FIXTURES))
    ]
    rows = []
    for row in range(height):
        y = ymin + (row + 0.5) * RESOLUTION_M
        values = []
        for col in range(width):
            x = xmin + (col + 0.5) * RESOLUTION_M
            inside_room = (
                xmin + ROBOT_RADIUS_M <= x <= xmax - ROBOT_RADIUS_M
                and ymin + ROBOT_RADIUS_M <= y <= ymax - ROBOT_RADIUS_M
            )
            values.append(
                inside_room and not any(_inside(x, y, item) for item in obstacles)
            )
        rows.append(values)
    return GridMap(
        rows,
        resolution=RESOLUTION_M,
        origin_x=xmin,
        origin_y=ymin,
    )


def _serialize_grid(grid: GridMap) -> dict:
    return {
        "schema_version": 1,
        "source": "reviewed_procedural_family_home_bootstrap",
        "truth_boundary": (
            "Isaac layout geometry for integration only; replace with the "
            "LingBot RGB-only occupancy after a G1-D survey."
        ),
        "frame_id": "map",
        "scene": SCENE_NAME,
        "resolution_m": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y],
        "width": grid.width,
        "height": grid.height,
        "robot_radius_m": ROBOT_RADIUS_M,
        "fixtures": [asdict(item) for item in HOME_FIXTURES],
        "rows": ["".join("." if cell else "#" for cell in row) for row in grid.free],
    }


def build_bootstrap_artifacts(
    output_dir: Path,
) -> tuple[GridMap, list[Place]]:
    """Write an explicit non-formal map and reviewed household place catalog."""

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    for place in PLACES:
        if not grid.is_free(grid.world_to_cell(place.pose.x, place.pose.y)):
            raise ValueError(f"family-home place is not footprint-safe: {place.place_id}")
        grid.plan(
            (START_POSE.x, START_POSE.y),
            (place.pose.x, place.pose.y),
        )
    map_payload = _serialize_grid(grid)
    map_bytes = json.dumps(
        map_payload, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    map_hash = hashlib.sha256(map_bytes).hexdigest()
    places_payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-family-home-bootstrap-v1",
            "sha256": map_hash,
            "frame_id": "map",
            "source": map_payload["source"],
        },
        "places": [
            {
                "id": place.place_id,
                "name": place.name,
                "aliases": list(place.aliases),
                "status": "approved",
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
                    "source": "reviewed_procedural_fixture_bounds",
                },
            }
            for place in PLACES
        ],
    }
    (output_dir / "bootstrap_occupancy.json").write_text(
        json.dumps(map_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "places.json").write_text(
        json.dumps(places_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "semantic_prompts.txt").write_text(
        "sofa\nbed\ndining table\nkitchen counter\ncabinet\n",
        encoding="utf-8",
    )
    return grid, list(PLACES)


def build_survey_path(grid: GridMap) -> list[tuple[float, float]]:
    """Visit every household zone using only footprint-safe paths."""

    places = {place.place_id: place for place in PLACES}
    waypoints = (
        (START_POSE.x, START_POSE.y),
        (
            places["bedroom_bed"].pose.x,
            places["bedroom_bed"].pose.y,
        ),
        (
            places["living_room_sofa"].pose.x,
            places["living_room_sofa"].pose.y,
        ),
        (-1.20, 3.35),
        # Approach the media cabinet head-on so small objects occupy more RGB
        # pixels than they do in the room-scale transit views.
        (-0.20, 2.45),
        (-0.20, 2.80),
        (0.25, 2.35),
        # Approach the dining tabletop from the south. The final segment points
        # the head RGB camera toward the cup/bowl instead of sweeping past them.
        (2.05, 1.55),
        (
            places["dining_area"].pose.x,
            places["dining_area"].pose.y,
        ),
        (1.55, 2.08),
        (2.55, 2.08),
        (2.05, 1.55),
        (
            places["kitchen_counter"].pose.x,
            places["kitchen_counter"].pose.y,
        ),
        (START_POSE.x, START_POSE.y),
    )
    route = [waypoints[0]]
    for start, goal in zip(waypoints, waypoints[1:]):
        segment = grid.plan(start, goal)
        route.extend(segment[1:])
    return route


__all__ = [
    "HOME_FIXTURES",
    "HOME_REGIONS",
    "DEMO_PLACES",
    "ALL_DEMO_PLACES",
    "MAP_BOUNDS",
    "PLACES",
    "ROBOT_RADIUS_M",
    "SCENE_NAME",
    "START_POSE",
    "HomeFixture",
    "HomeRegion",
    "build_bootstrap_artifacts",
    "build_grid",
    "build_survey_path",
]
