"""Generate footprint-safe docking candidates around fused 3D instances."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path

from ..errors import ConfigurationError
from ..models import DockingCandidate, Pose2D
from ..map_validation import load_map_metadata, load_pgm


class TraversabilityGrid:
    def __init__(self, map_yaml: str | Path, footprint_radius_m: float) -> None:
        if not math.isfinite(footprint_radius_m) or footprint_radius_m <= 0:
            raise ConfigurationError("Footprint radius must be finite and positive")
        self.meta = load_map_metadata(map_yaml)
        self.width, self.height, self.pixels = load_pgm(self.meta.image)
        self.footprint_radius_m = footprint_radius_m
        self.radius_cells = int(math.ceil(footprint_radius_m / self.meta.resolution))
        self._traversable_cache: dict[tuple[int, int], bool] = {}

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = math.floor((x - self.meta.origin_x) / self.meta.resolution)
        bottom = math.floor((y - self.meta.origin_y) / self.meta.resolution)
        return self.height - 1 - bottom, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        bottom = self.height - 1 - row
        return (
            self.meta.origin_x + (col + 0.5) * self.meta.resolution,
            self.meta.origin_y + (bottom + 0.5) * self.meta.resolution,
        )

    def cell_status(self, row: int, col: int) -> str:
        if not (0 <= row < self.height and 0 <= col < self.width):
            return "outside"
        gray = self.pixels[row * self.width + col]
        occupancy = gray / 255.0 if self.meta.negate else (255 - gray) / 255.0
        if occupancy > self.meta.occupied_thresh:
            return "occupied"
        if occupancy < self.meta.free_thresh:
            return "free"
        return "unknown"

    def footprint_is_free(self, row: int, col: int) -> bool:
        key = (row, col)
        cached = self._traversable_cache.get(key)
        if cached is not None:
            return cached
        x, y = self.cell_to_world(row, col)
        free = True
        for dr in range(-self.radius_cells, self.radius_cells + 1):
            for dc in range(-self.radius_cells, self.radius_cells + 1):
                candidate_row, candidate_col = row + dr, col + dc
                cell_x, cell_y = self.cell_to_world(candidate_row, candidate_col)
                if math.hypot(cell_x - x, cell_y - y) > (
                    self.footprint_radius_m + self.meta.resolution * math.sqrt(2) / 2
                ):
                    continue
                if self.cell_status(candidate_row, candidate_col) != "free":
                    free = False
                    break
            if not free:
                break
        self._traversable_cache[key] = free
        return free

    def reachable_cells(self, start: Pose2D) -> set[tuple[int, int]]:
        start_cell = self.world_to_cell(start.x, start.y)
        if not self.footprint_is_free(*start_cell):
            raise ConfigurationError("Docking reachability start pose is not footprint-safe")
        reached = {start_cell}
        frontier = deque([start_cell])
        while frontier:
            row, col = frontier.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                candidate = (row + dr, col + dc)
                if candidate in reached or not self.footprint_is_free(*candidate):
                    continue
                reached.add(candidate)
                frontier.append(candidate)
        return reached

    def clearance(self, row: int, col: int, maximum_m: float = 1.0) -> float:
        x, y = self.cell_to_world(row, col)
        search = int(math.ceil(maximum_m / self.meta.resolution))
        nearest = maximum_m
        for dr in range(-search, search + 1):
            for dc in range(-search, search + 1):
                if self.cell_status(row + dr, col + dc) == "free":
                    continue
                cell_x, cell_y = self.cell_to_world(row + dr, col + dc)
                nearest = min(nearest, math.hypot(cell_x - x, cell_y - y))
        return max(0.0, nearest - self.meta.resolution * math.sqrt(2) / 2)


def generate_docking_candidates(
    grid: TraversabilityGrid,
    target_x: float,
    target_y: float,
    reachable: set[tuple[int, int]],
    *,
    radii_m: tuple[float, ...] = (0.55, 0.70, 0.85),
    angular_samples: int = 24,
    maximum_candidates: int = 8,
) -> tuple[DockingCandidate, ...]:
    if angular_samples < 4 or maximum_candidates < 1:
        raise ConfigurationError("Docking sampling settings are invalid")
    by_cell: dict[tuple[int, int], DockingCandidate] = {}
    for radius in radii_m:
        if not math.isfinite(radius) or radius <= grid.footprint_radius_m:
            raise ConfigurationError("Docking radius must exceed the robot footprint radius")
        for index in range(angular_samples):
            angle = 2.0 * math.pi * index / angular_samples
            x = target_x + radius * math.cos(angle)
            y = target_y + radius * math.sin(angle)
            cell = grid.world_to_cell(x, y)
            if cell not in reachable or not grid.footprint_is_free(*cell):
                continue
            snapped_x, snapped_y = grid.cell_to_world(*cell)
            candidate = DockingCandidate(
                candidate_id=f"dock_r{cell[0]}_c{cell[1]}",
                pose=Pose2D(
                    snapped_x,
                    snapped_y,
                    math.atan2(target_y - snapped_y, target_x - snapped_x),
                    "map",
                ),
                clearance_m=grid.clearance(*cell),
                footprint_radius_m=grid.footprint_radius_m,
                occupancy_status="free",
                reachable=True,
                review_status="pending",
            )
            previous = by_cell.get(cell)
            if previous is None or candidate.clearance_m > previous.clearance_m:
                by_cell[cell] = candidate
    ranked = sorted(
        by_cell.values(),
        key=lambda item: (-item.clearance_m, math.hypot(item.pose.x - target_x, item.pose.y - target_y)),
    )
    return tuple(ranked[:maximum_candidates])


__all__ = ["TraversabilityGrid", "generate_docking_candidates"]

