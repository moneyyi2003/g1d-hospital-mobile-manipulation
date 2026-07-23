"""Bootstrap navigation artifacts for the Hospital reception-area MVP.

The bootstrap grid is deliberately labelled as Isaac geometry.  It exists to
exercise camera acquisition and control before a LingBot-Map survey has been
processed.  Formal navigation loads the ROS occupancy map and reviewed place
catalog produced by the RGB-only pipeline instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from simple_room_vln.core import GridMap, Place, Pose2D


MAP_BOUNDS = (-11.60, -2.75, 4.35, 12.25)
RESOLUTION_M = 0.10
ROBOT_RADIUS_M = 0.40
HOSPITAL_START = Pose2D(0.0, -1.50, math.pi / 2.0)

# Bounds measured from the composed Hospital USD.  They cover only the lobby
# objects needed by the first reproducible task, not the complete hospital.
RECEPTION_DESKS = (
    (-0.70, 1.98, 2.18, 4.41),
    (1.77, 1.98, 4.65, 4.41),
)
WAITING_CHAIRS = (
    (-8.75, -0.10, -6.37, 1.43),
    (-6.31, -0.10, -3.92, 1.43),
)

RECEPTION_POSE = Pose2D(-1.55, 3.20, 0.0)
WAITING_AREA_POSE = Pose2D(-5.95, 2.20, -math.pi / 2.0)
MAIN_CORRIDOR_POSE = Pose2D(-8.00, 7.20, math.pi / 2.0)


def _inflate(rectangle: tuple[float, float, float, float], radius: float):
    xmin, ymin, xmax, ymax = rectangle
    return xmin - radius, ymin - radius, xmax + radius, ymax + radius


def _inside(x: float, y: float, rectangle: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = rectangle
    return xmin <= x <= xmax and ymin <= y <= ymax


def _bootstrap_grid() -> GridMap:
    xmin, ymin, xmax, ymax = MAP_BOUNDS
    width = int(math.ceil((xmax - xmin) / RESOLUTION_M))
    height = int(math.ceil((ymax - ymin) / RESOLUTION_M))
    obstacles = [
        _inflate(item, ROBOT_RADIUS_M)
        for item in (*RECEPTION_DESKS, *WAITING_CHAIRS)
    ]
    rows: list[list[bool]] = []
    for row in range(height):
        y = ymin + (row + 0.5) * RESOLUTION_M
        values = []
        for col in range(width):
            x = xmin + (col + 0.5) * RESOLUTION_M
            wall_clear = (
                xmin + ROBOT_RADIUS_M <= x <= xmax - ROBOT_RADIUS_M
                and ymin + ROBOT_RADIUS_M <= y <= ymax - ROBOT_RADIUS_M
            )
            values.append(wall_clear and not any(_inside(x, y, item) for item in obstacles))
        rows.append(values)
    return GridMap(rows, resolution=RESOLUTION_M, origin_x=xmin, origin_y=ymin)


def build_survey_path(grid: GridMap) -> list[tuple[float, float]]:
    """Return a wall-safe closed survey of the reception and waiting lobby.

    The bootstrap grid contains measured furniture bounds but not every wall in
    the full Hospital asset.  Keep this route inside the visually verified
    lobby until the formal LingBot occupancy map replaces bootstrap geometry.
    """

    waypoints = [
        (HOSPITAL_START.x, HOSPITAL_START.y),
        (-2.20, -1.50),
        (-2.20, 3.20),
        (-8.20, 3.20),
        (-3.00, 3.20),
        (-3.00, -1.50),
        (HOSPITAL_START.x, HOSPITAL_START.y),
    ]
    combined = [waypoints[0]]
    for start, goal in zip(waypoints, waypoints[1:]):
        segment = grid.plan(start, goal)
        combined.extend(segment[1:])
    return combined


def _place_payload(place: Place, source_id: str) -> dict:
    approved = place.status == "approved"
    return {
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
        "target": {"type": "semantic_region", "source_id": source_id},
        "review": {
            "status": (
                "accepted_for_bootstrap_demo"
                if approved
                else "pending_safe_route_validation"
            ),
            "source": "measured_usd_bounds",
        },
    }


def build_bootstrap_artifacts(output_dir: Path) -> tuple[GridMap, list[Place]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = _bootstrap_grid()
    map_payload = {
        "schema_version": 1,
        "source": "isaac_geometry_bootstrap",
        "truth_boundary": (
            "Integration-only lobby grid; replace with aligned LingBot RGB-only occupancy."
        ),
        "frame_id": "map",
        "scene": "IsaacSim/Hospital.usd reception area",
        "resolution_m": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y],
        "width": grid.width,
        "height": grid.height,
        "robot_radius_m": ROBOT_RADIUS_M,
        "rows": ["".join("." if cell else "#" for cell in row) for row in grid.free],
    }
    map_bytes = json.dumps(map_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    places = [
        Place(
            "reception",
            "医院前台",
            ("前台", "接待处", "护士站", "reception", "reception desk", "front desk"),
            RECEPTION_POSE,
        ),
        Place(
            "waiting_area",
            "候诊区",
            ("候诊区", "等候区", "椅子", "waiting area", "waiting chairs"),
            WAITING_AREA_POSE,
        ),
        Place(
            "main_corridor",
            "主走廊",
            ("主走廊", "走廊", "main corridor", "hallway", "corridor"),
            MAIN_CORRIDOR_POSE,
            "pending_safe_route_validation",
        ),
    ]
    place_payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-hospital-lobby-bootstrap-v1",
            "sha256": hashlib.sha256(map_bytes).hexdigest(),
            "frame_id": "map",
            "source": "isaac_geometry_bootstrap",
        },
        "places": [
            _place_payload(places[0], "SM_ReceptionDesk"),
            _place_payload(places[1], "SM_Chair_02a"),
            _place_payload(places[2], "lobby_to_main_corridor"),
        ],
    }
    (output_dir / "bootstrap_occupancy.json").write_text(
        json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "places.json").write_text(
        json.dumps(place_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "semantic_prompts.txt").write_text(
        "reception desk\nwaiting chair\nhospital bed\nsupply cart\ndoor\n",
        encoding="utf-8",
    )
    return grid, places


__all__ = [
    "HOSPITAL_START",
    "ROBOT_RADIUS_M",
    "build_bootstrap_artifacts",
    "build_survey_path",
]
