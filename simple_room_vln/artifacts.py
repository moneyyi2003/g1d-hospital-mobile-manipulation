"""Create and load explicit map/place artifacts for the runnable MVP."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .core import GridMap, Place, Pose2D


DEFAULT_OUTPUT_DIR = Path("E:/robot/outputs/simple_room_vln")
MAP_BOUNDS = (-4.20, -3.10, 4.20, 4.55)
RESOLUTION_M = 0.10
ROBOT_RADIUS_M = 0.40

# Measured from the loaded USD bounds. The sofa set root is translated by this
# value in run_g1d_simple_room_vln.py. The rug is intentionally not an obstacle.
SOFA_SET_TRANSLATION = (-9.50, -4.30, -0.7695)
SOFA_BOUNDS = (-3.205, 0.985, -2.246, 2.981)
SOFA_TABLE_BOUNDS = (-4.061, 1.605, -3.260, 2.408)
ROOM_TABLE_BOUNDS = (2.965, -1.274, 4.048, 1.274)

SOFA_SIDE_POSE = Pose2D(-2.72, 0.27, math.pi / 2.0)


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
        _inflate(SOFA_BOUNDS, ROBOT_RADIUS_M),
        _inflate(SOFA_TABLE_BOUNDS, ROBOT_RADIUS_M),
        _inflate(ROOM_TABLE_BOUNDS, ROBOT_RADIUS_M),
    ]
    rows = []
    for row in range(height):
        y = ymin + (row + 0.5) * RESOLUTION_M
        values = []
        for col in range(width):
            x = xmin + (col + 0.5) * RESOLUTION_M
            wall_clear = (
                x >= xmin + ROBOT_RADIUS_M
                and x <= xmax - ROBOT_RADIUS_M
                and y >= ymin + ROBOT_RADIUS_M
                and y <= ymax - ROBOT_RADIUS_M
            )
            values.append(wall_clear and not any(_inside(x, y, item) for item in obstacles))
        rows.append(values)
    return GridMap(rows, resolution=RESOLUTION_M, origin_x=xmin, origin_y=ymin)


def _serialize_grid(grid: GridMap) -> dict:
    return {
        "schema_version": 1,
        "source": "isaac_geometry_bootstrap",
        "truth_boundary": (
            "For end-to-end integration only; replace with aligned LingBot RGB-only occupancy."
        ),
        "frame_id": "map",
        "scene": "SimpleRoom+SofaTablePlant",
        "resolution_m": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y],
        "width": grid.width,
        "height": grid.height,
        "robot_radius_m": ROBOT_RADIUS_M,
        "rows": ["".join("." if cell else "#" for cell in row) for row in grid.free],
    }


def build_bootstrap_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[GridMap, list[Place]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = _bootstrap_grid()
    map_payload = _serialize_grid(grid)
    map_bytes = json.dumps(map_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()
    places = [
        Place(
            "sofa_side",
            "沙发旁边",
            ("沙发", "沙发旁", "沙发旁边", "sofa", "couch"),
            SOFA_SIDE_POSE,
        )
    ]
    place_payload = {
        "schema_version": 2,
        "map": {
            "id": "isaac-simple-room-bootstrap-v1",
            "sha256": map_sha256,
            "frame_id": "map",
            "source": "isaac_geometry_bootstrap",
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
                "target": {"type": "semantic_instance", "source_id": "sofa_1"},
                "review": {
                    "status": "accepted_for_bootstrap_demo",
                    "source": "measured_usd_bounds",
                },
            }
            for place in places
        ],
    }
    (output_dir / "bootstrap_occupancy.json").write_text(
        json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "places.json").write_text(
        json.dumps(place_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "semantic_prompts.txt").write_text(
        "sofa\ntable\ndoor\n", encoding="utf-8"
    )
    return grid, places


def load_grid(path: Path) -> GridMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [[cell == "." for cell in row] for row in payload["rows"]]
    return GridMap(
        rows,
        resolution=float(payload["resolution_m"]),
        origin_x=float(payload["origin"][0]),
        origin_y=float(payload["origin"][1]),
    )


def _read_pgm(path: Path) -> tuple[int, int, bytes]:
    """Read the binary P5 maps emitted by lingbot-nav without extra deps."""
    with path.open("rb") as stream:
        tokens: list[bytes] = []
        while len(tokens) < 4:
            line = stream.readline()
            if not line:
                raise ValueError(f"invalid PGM header: {path}")
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        if tokens[0] != b"P5" or int(tokens[3]) != 255:
            raise ValueError(f"only 8-bit binary PGM maps are supported: {path}")
        width, height = int(tokens[1]), int(tokens[2])
        pixels = stream.read(width * height)
    if len(pixels) != width * height:
        raise ValueError(f"truncated PGM map: {path}")
    return width, height, pixels


def _inflate_free_rows(rows: list[list[bool]], radius_cells: int) -> list[list[bool]]:
    if radius_cells <= 0:
        return rows
    height, width = len(rows), len(rows[0])
    blocked = [(r, c) for r in range(height) for c in range(width) if not rows[r][c]]
    inflated = [row[:] for row in rows]
    radius_sq = radius_cells * radius_cells
    for row, col in blocked:
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr, cc = row + dr, col + dc
                if dr * dr + dc * dc <= radius_sq and 0 <= rr < height and 0 <= cc < width:
                    inflated[rr][cc] = False
    return inflated


def load_ros_grid(map_yaml: Path, *, robot_radius_m: float = ROBOT_RADIUS_M) -> GridMap:
    """Load and footprint-inflate a ROS trinary map built from LingBot points."""
    text = map_yaml.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip("'\"")
    try:
        resolution = float(fields["resolution"])
        origin_values = [float(item) for item in re.findall(r"[-+0-9.eE]+", fields["origin"])]
        image_path = (map_yaml.parent / fields["image"]).resolve()
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"invalid ROS map YAML: {map_yaml}") from exc
    width, height, pixels = _read_pgm(image_path)
    # PGM row zero is max-Y, while GridMap row zero is min-Y.
    rows = []
    for output_row in range(height):
        pgm_row = height - 1 - output_row
        offset = pgm_row * width
        rows.append([pixels[offset + col] >= 250 for col in range(width)])
    radius_cells = int(math.ceil(robot_radius_m / resolution))
    return GridMap(
        _inflate_free_rows(rows, radius_cells),
        resolution=resolution,
        origin_x=origin_values[0],
        origin_y=origin_values[1],
    )


def load_approved_places(path: Path) -> list[Place]:
    """Load only operator-approved docking poses from the v2 place catalog."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: list[Place] = []
    for value in payload.get("places", []):
        if value.get("status") != "approved":
            continue
        candidates = value.get("docking_candidates", [])
        selected_id = str(value.get("selected_docking_candidate", ""))
        selected = next((item for item in candidates if str(item.get("id")) == selected_id), None)
        pose = selected.get("pose") if selected is not None else value.get("entrance_pose")
        if not isinstance(pose, dict):
            raise ValueError(f"approved place {value.get('id')!r} has no selected pose")
        result.append(
            Place(
                str(value["id"]),
                str(value["name"]),
                tuple(str(item) for item in value.get("aliases", [])),
                Pose2D(float(pose["x"]), float(pose["y"]), float(pose.get("yaw", 0.0))),
                "approved",
            )
        )
    if not result:
        raise ValueError(f"place catalog has no approved destinations: {path}")
    return result


def load_lingbot_artifacts(
    map_yaml: Path,
    places_json: Path,
    *,
    robot_radius_m: float = ROBOT_RADIUS_M,
) -> tuple[GridMap, list[Place]]:
    return (
        load_ros_grid(map_yaml, robot_radius_m=robot_radius_m),
        load_approved_places(places_json),
    )
