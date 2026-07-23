"""Project an aligned metric point cloud into a ROS-compatible trinary map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from .pointcloud import _numpy


@dataclass(frozen=True)
class OccupancyBuildConfig:
    resolution: float = 0.05
    ground_z: float = 0.0
    ground_band: float = 0.10
    obstacle_min_height: float = 0.12
    obstacle_max_height: float = 1.80
    padding: float = 0.50
    minimum_points_per_cell: int = 2
    bounds_quantile: float = 0.002

    def validate(self) -> None:
        values = (
            self.resolution,
            self.ground_z,
            self.ground_band,
            self.obstacle_min_height,
            self.obstacle_max_height,
            self.padding,
        )
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("Occupancy settings contain non-finite values")
        if self.resolution <= 0 or self.ground_band <= 0 or self.padding < 0:
            raise ConfigurationError("Invalid occupancy resolution/ground band/padding")
        if self.obstacle_max_height <= self.obstacle_min_height:
            raise ConfigurationError("obstacle_max_height must exceed obstacle_min_height")
        if self.minimum_points_per_cell < 1:
            raise ConfigurationError("minimum_points_per_cell must be positive")
        if not 0 <= self.bounds_quantile < 0.1:
            raise ConfigurationError("bounds_quantile must be in [0, 0.1)")


@dataclass(frozen=True)
class OccupancyGrid:
    cells: Any  # int8: -1 unknown, 0 free, 100 occupied; row 0 is minimum y
    resolution: float
    origin_x: float
    origin_y: float
    config: OccupancyBuildConfig


def build_occupancy(points, config: OccupancyBuildConfig) -> OccupancyGrid:
    np = _numpy()
    config.validate()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ConfigurationError("Point cloud must have shape [N, 3]")
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 100:
        raise ConfigurationError("At least 100 finite points are required to build a map")

    q = config.bounds_quantile
    if q:
        lower = np.quantile(points[:, :2], q, axis=0)
        upper = np.quantile(points[:, :2], 1.0 - q, axis=0)
    else:
        lower, upper = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    lower -= config.padding
    upper += config.padding
    span = upper - lower
    width, height = np.ceil(span / config.resolution).astype(int) + 1
    if width < 2 or height < 2 or width * height > 25_000_000:
        raise ConfigurationError(
            f"Unreasonable map size {width}x{height}; check scale/alignment/outliers"
        )

    in_bounds = (
        (points[:, 0] >= lower[0])
        & (points[:, 0] <= upper[0])
        & (points[:, 1] >= lower[1])
        & (points[:, 1] <= upper[1])
    )
    points = points[in_bounds]
    col = np.floor((points[:, 0] - lower[0]) / config.resolution).astype(np.int64)
    row = np.floor((points[:, 1] - lower[1]) / config.resolution).astype(np.int64)
    flat = row * width + col

    relative_z = points[:, 2] - config.ground_z
    ground_mask = np.abs(relative_z) <= config.ground_band
    obstacle_mask = (
        (relative_z >= config.obstacle_min_height)
        & (relative_z <= config.obstacle_max_height)
    )
    ground_counts = np.bincount(flat[ground_mask], minlength=width * height)
    obstacle_counts = np.bincount(flat[obstacle_mask], minlength=width * height)

    cells = np.full(width * height, -1, dtype=np.int8)
    cells[ground_counts >= config.minimum_points_per_cell] = 0
    cells[obstacle_counts >= config.minimum_points_per_cell] = 100
    cells = cells.reshape((height, width))
    return OccupancyGrid(cells, config.resolution, float(lower[0]), float(lower[1]), config)


def clear_traversed_footprints(
    grid: OccupancyGrid, positions, *, radius_m: float
) -> OccupancyGrid:
    """Mark collision-validated robot footprints along a survey as free."""

    np = _numpy()
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ConfigurationError("Traversed footprint positions must have shape [N, 2]")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ConfigurationError("Traversed footprint radius must be positive")
    cells = np.array(grid.cells, copy=True)
    radius_cells = int(math.ceil(radius_m / grid.resolution))
    radius_sq = radius_m * radius_m
    for x, y in positions:
        center_col = int(math.floor((x - grid.origin_x) / grid.resolution))
        center_row = int(math.floor((y - grid.origin_y) / grid.resolution))
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not (0 <= row < cells.shape[0] and 0 <= col < cells.shape[1]):
                    continue
                cell_x = grid.origin_x + (col + 0.5) * grid.resolution
                cell_y = grid.origin_y + (row + 0.5) * grid.resolution
                if (cell_x - x) ** 2 + (cell_y - y) ** 2 <= radius_sq:
                    cells[row, col] = 0
    return OccupancyGrid(
        cells, grid.resolution, grid.origin_x, grid.origin_y, grid.config
    )


def write_ros_map(output_dir: str | Path, grid: OccupancyGrid, stem: str = "map") -> dict[str, Any]:
    np = _numpy()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pgm_path = output / f"{stem}.pgm"
    yaml_path = output / f"{stem}.yaml"
    metadata_path = output / f"{stem}_metadata.json"

    pixels = np.full(grid.cells.shape, 205, dtype=np.uint8)
    pixels[grid.cells == 0] = 254
    pixels[grid.cells == 100] = 0
    pixels = np.flipud(pixels)  # ROS PGM row 0 represents maximum map y.
    with pgm_path.open("wb") as stream:
        stream.write(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii"))
        stream.write(pixels.tobytes(order="C"))

    yaml_text = (
        f"image: {pgm_path.name}\n"
        f"resolution: {grid.resolution:.9f}\n"
        f"origin: [{grid.origin_x:.9f}, {grid.origin_y:.9f}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    metadata = {
        "width": int(grid.cells.shape[1]),
        "height": int(grid.cells.shape[0]),
        "resolution": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y, 0.0],
        "cell_counts": {
            "unknown": int((grid.cells == -1).sum()),
            "free": int((grid.cells == 0).sum()),
            "occupied": int((grid.cells == 100).sum()),
        },
        "build_config": asdict(grid.config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "pgm": str(pgm_path),
        "yaml": str(yaml_path),
        "metadata": str(metadata_path),
        **metadata,
    }
