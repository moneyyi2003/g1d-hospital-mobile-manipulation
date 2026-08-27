"""Path planning on a generated ROS occupancy map.

This module deliberately has no Habitat imports.  In particular, it cannot
query a navmesh or simulator pathfinder while producing a route.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path

from ..errors import ConfigurationError
from ..mapping.semantic_map import load_ros_occupancy


@dataclass(frozen=True)
class OccupancyPlannerConfig:
    robot_radius: float = 0.14
    max_snap_distance: float = 0.60
    unknown_is_occupied: bool = True


class OccupancyPathPlanner:
    """Inflated-grid A* planner operating only on LingBot map artifacts."""

    def __init__(
        self,
        map_yaml: str | Path,
        config: OccupancyPlannerConfig = OccupancyPlannerConfig(),
    ) -> None:
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        self.np = np
        self.map_yaml = Path(map_yaml).expanduser().resolve()
        self.grid = load_ros_occupancy(self.map_yaml)
        if config.robot_radius < 0 or config.max_snap_distance <= 0:
            raise ConfigurationError("Invalid occupancy planner clearance settings")
        self.config = config
        blocked = self.grid.cells == 100
        if config.unknown_is_occupied:
            blocked |= self.grid.cells == -1
        clearance = distance_transform_edt(~blocked) * self.grid.resolution
        self.free = (~blocked) & (clearance >= config.robot_radius)
        if not bool(self.free.any()):
            raise ConfigurationError(f"Occupancy map has no traversable cells: {self.map_yaml}")

    def _cell(self, point: tuple[float, float]) -> tuple[int, int]:
        col = math.floor((point[0] - self.grid.origin_x) / self.grid.resolution)
        row = math.floor((point[1] - self.grid.origin_y) / self.grid.resolution)
        return int(row), int(col)

    def _point(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (
            self.grid.origin_x + (col + 0.5) * self.grid.resolution,
            self.grid.origin_y + (row + 0.5) * self.grid.resolution,
        )

    def _snap(self, point: tuple[float, float], label: str) -> tuple[int, int]:
        np = self.np
        row, col = self._cell(point)
        height, width = self.free.shape
        if 0 <= row < height and 0 <= col < width and bool(self.free[row, col]):
            return row, col
        rows, cols = np.where(self.free)
        distances = np.hypot(rows - row, cols - col) * self.grid.resolution
        index = int(distances.argmin())
        if float(distances[index]) > self.config.max_snap_distance:
            raise ConfigurationError(
                f"{label} is {float(distances[index]):.2f} map units from generated free space"
            )
        return int(rows[index]), int(cols[index])

    def plan(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        source = self._snap(start, "Route start")
        target = self._snap(goal, "Route goal")
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
        cost = {source: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        neighbours = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )
        height, width = self.free.shape
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == target:
                break
            base = cost[current]
            for dr, dc, step_cost in neighbours:
                row, col = current[0] + dr, current[1] + dc
                if not (0 <= row < height and 0 <= col < width and self.free[row, col]):
                    continue
                if dr and dc and not (
                    self.free[current[0] + dr, current[1]]
                    and self.free[current[0], current[1] + dc]
                ):
                    continue
                nxt = (row, col)
                candidate = base + step_cost
                if candidate >= cost.get(nxt, math.inf):
                    continue
                cost[nxt] = candidate
                parent[nxt] = current
                heuristic = math.hypot(target[0] - row, target[1] - col)
                heapq.heappush(frontier, (candidate + heuristic, nxt))
        if target not in cost:
            raise ConfigurationError("No route through the generated occupancy map")
        cells = [target]
        while cells[-1] != source:
            cells.append(parent[cells[-1]])
        cells.reverse()
        # Keep direction changes only; this is sufficient for the waypoint follower
        # and avoids sending hundreds of adjacent grid cells to the dashboard.
        reduced = [cells[0]]
        previous_direction: tuple[int, int] | None = None
        for before, after in zip(cells, cells[1:]):
            direction = (after[0] - before[0], after[1] - before[1])
            if previous_direction is not None and direction != previous_direction:
                reduced.append(before)
            previous_direction = direction
        if reduced[-1] != cells[-1]:
            reduced.append(cells[-1])
        points = [self._point(cell) for cell in reduced]
        points[0] = start
        points[-1] = goal
        return points


__all__ = ["OccupancyPathPlanner", "OccupancyPlannerConfig"]
