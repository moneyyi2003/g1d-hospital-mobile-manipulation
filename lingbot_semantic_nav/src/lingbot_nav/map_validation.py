"""Check that every semantic entrance pose lies in free occupancy-map space."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import BinaryIO

from .errors import ConfigurationError
from .place_db import PlaceDatabase


@dataclass(frozen=True)
class MapMetadata:
    image: Path
    resolution: float
    origin_x: float
    origin_y: float
    negate: bool
    occupied_thresh: float
    free_thresh: float


@dataclass(frozen=True)
class PlaceMapCheck:
    place_id: str
    name: str
    status: str
    pixel_row: int
    pixel_col: int
    grayscale: int | None
    robot_radius_m: float = 0.0
    blocking_cells: int = 0

    def to_dict(self):
        return asdict(self)


def _yaml_scalar_lines(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_map_metadata(path: str | Path) -> MapMetadata:
    source = Path(path).expanduser().resolve()
    try:
        values = _yaml_scalar_lines(source)
        origin = ast.literal_eval(values["origin"])
        image = Path(values["image"])
        if not image.is_absolute():
            image = source.parent / image
        return MapMetadata(
            image=image.resolve(),
            resolution=float(values["resolution"]),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            negate=bool(int(values.get("negate", "0"))),
            occupied_thresh=float(values.get("occupied_thresh", "0.65")),
            free_thresh=float(values.get("free_thresh", "0.196")),
        )
    except (OSError, KeyError, ValueError, SyntaxError, TypeError, IndexError) as exc:
        raise ConfigurationError(f"Cannot parse ROS map YAML {source}: {exc}") from exc


def _read_pnm_token(stream: BinaryIO) -> bytes:
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            break
        if char == b"#":
            stream.readline()
            if token:
                break
            continue
        if char.isspace():
            if token:
                break
            continue
        token.extend(char)
    if not token:
        raise ConfigurationError("Unexpected end of PGM header")
    return bytes(token)


def load_pgm(path: str | Path) -> tuple[int, int, bytes]:
    source = Path(path)
    try:
        with source.open("rb") as stream:
            magic = _read_pnm_token(stream)
            width = int(_read_pnm_token(stream))
            height = int(_read_pnm_token(stream))
            max_value = int(_read_pnm_token(stream))
            if magic != b"P5" or max_value != 255:
                raise ConfigurationError("Only 8-bit binary P5 PGM maps are supported")
            pixels = stream.read(width * height)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read PGM {source}: {exc}") from exc
    if len(pixels) != width * height:
        raise ConfigurationError(f"PGM {source} has truncated pixel data")
    return width, height, pixels


def validate_places_on_map(
    places: PlaceDatabase,
    map_yaml: str | Path,
    *,
    robot_radius: float = 0.0,
) -> list[PlaceMapCheck]:
    if not math.isfinite(robot_radius) or robot_radius < 0.0:
        raise ConfigurationError("robot_radius must be finite and non-negative")
    meta = load_map_metadata(map_yaml)
    width, height, pixels = load_pgm(meta.image)

    def cell_status(row: int, col: int) -> tuple[str, int | None]:
        if not (0 <= row < height and 0 <= col < width):
            return "outside", None
        gray = pixels[row * width + col]
        occupancy = gray / 255.0 if meta.negate else (255 - gray) / 255.0
        if occupancy > meta.occupied_thresh:
            return "occupied", gray
        if occupancy < meta.free_thresh:
            return "free", gray
        return "unknown", gray

    checks = []
    for place in places.places:
        pose = place.entrance_pose
        col = math.floor((pose.x - meta.origin_x) / meta.resolution)
        row_from_bottom = math.floor((pose.y - meta.origin_y) / meta.resolution)
        row = height - 1 - row_from_bottom
        center_status, gray = cell_status(row, col)
        if center_status != "free":
            checks.append(PlaceMapCheck(
                place.place_id, place.name, center_status, row, col, gray,
                robot_radius, 1,
            ))
            continue
        blocked: list[str] = []
        if robot_radius > 0.0:
            min_col = math.floor((pose.x - robot_radius - meta.origin_x) / meta.resolution)
            max_col = math.floor((pose.x + robot_radius - meta.origin_x) / meta.resolution)
            min_bottom = math.floor((pose.y - robot_radius - meta.origin_y) / meta.resolution)
            max_bottom = math.floor((pose.y + robot_radius - meta.origin_y) / meta.resolution)
            for candidate_bottom in range(min_bottom, max_bottom + 1):
                candidate_row = height - 1 - candidate_bottom
                for candidate_col in range(min_col, max_col + 1):
                    cell_min_x = meta.origin_x + candidate_col * meta.resolution
                    cell_max_x = cell_min_x + meta.resolution
                    cell_min_y = meta.origin_y + candidate_bottom * meta.resolution
                    cell_max_y = cell_min_y + meta.resolution
                    dx = max(cell_min_x - pose.x, 0.0, pose.x - cell_max_x)
                    dy = max(cell_min_y - pose.y, 0.0, pose.y - cell_max_y)
                    if math.hypot(dx, dy) > robot_radius:
                        continue
                    status, _ = cell_status(candidate_row, candidate_col)
                    if status != "free":
                        blocked.append(status)
        if "outside" in blocked:
            status = "outside"
        elif "occupied" in blocked:
            status = "occupied"
        elif "unknown" in blocked:
            status = "unknown"
        else:
            status = "free"
        checks.append(PlaceMapCheck(
            place.place_id, place.name, status, row, col, gray,
            robot_radius, len(blocked),
        ))
    return checks
