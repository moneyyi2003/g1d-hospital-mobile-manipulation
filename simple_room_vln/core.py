"""Dependency-light planning and control for the SimpleRoom demo."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class Place:
    place_id: str
    name: str
    aliases: tuple[str, ...]
    pose: Pose2D
    status: str = "approved"


def normalize_text(value: str) -> str:
    return "".join(value.strip().casefold().split())


def resolve_place(command: str, places: Sequence[Place]) -> Place:
    """Resolve only cataloged places; coordinates never come from language."""

    normalized = normalize_text(command)
    if not normalized:
        raise ValueError("导航指令为空")
    matches: list[tuple[int, Place]] = []
    for place in places:
        if place.status != "approved":
            continue
        labels = (place.place_id, place.name, *place.aliases)
        score = max(
            (len(normalize_text(label)) for label in labels if normalize_text(label) in normalized),
            default=0,
        )
        if score:
            matches.append((score, place))
    if not matches:
        known = "、".join(item.name for item in places if item.status == "approved")
        raise ValueError(f"指令没有匹配已审核地点；可用地点：{known}")
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError("指令同时匹配多个地点")
    return matches[0][1]


class GridMap:
    """2-D occupancy grid where True means traversable."""

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(
        self,
        free_rows: Sequence[Sequence[bool]],
        *,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        if resolution <= 0.0 or not free_rows or not free_rows[0]:
            raise ValueError("invalid occupancy grid")
        width = len(free_rows[0])
        if any(len(row) != width for row in free_rows):
            raise ValueError("occupancy rows must have equal length")
        self.free = tuple(tuple(bool(cell) for cell in row) for row in free_rows)
        self.height = len(self.free)
        self.width = width
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    def is_free(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width and self.free[row][col]

    def _segment_is_free(self, left: tuple[float, float], right: tuple[float, float]) -> bool:
        distance = math.dist(left, right)
        samples = max(2, int(math.ceil(distance / (0.35 * self.resolution))))
        for index in range(samples + 1):
            ratio = index / samples
            x = left[0] + ratio * (right[0] - left[0])
            y = left[1] + ratio * (right[1] - left[1])
            if not self.is_free(self.world_to_cell(x, y)):
                return False
        return True

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        start_cell = self.world_to_cell(*start)
        goal_cell = self.world_to_cell(*goal)
        if not self.is_free(start_cell):
            raise ValueError(f"start is not free: {start}")
        if not self.is_free(goal_cell):
            raise ValueError(f"goal is not free: {goal}")

        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        cost = {start_cell: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            for dr, dc, weight in self._NEIGHBORS:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self.is_free(neighbor):
                    continue
                if dr and dc:
                    if not self.is_free((current[0] + dr, current[1])):
                        continue
                    if not self.is_free((current[0], current[1] + dc)):
                        continue
                new_cost = cost[current] + weight
                if new_cost >= cost.get(neighbor, math.inf):
                    continue
                cost[neighbor] = new_cost
                heuristic = math.dist(neighbor, goal_cell)
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current
        if goal_cell not in came_from:
            raise ValueError("no free path to goal")

        cells = []
        cursor: tuple[int, int] | None = goal_cell
        while cursor is not None:
            cells.append(cursor)
            cursor = came_from[cursor]
        cells.reverse()
        raw = [start, *(self.cell_to_world(cell) for cell in cells[1:-1]), goal]
        return self.simplify(raw)

    def simplify(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return list(points)
        result = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if self._segment_is_free(points[anchor], points[candidate]):
                    break
                candidate -= 1
            result.append(points[candidate])
            anchor = candidate
        return result


class PathFollower:
    """Waypoint follower producing differential-drive twist commands."""

    def __init__(
        self,
        path: Sequence[tuple[float, float]],
        *,
        goal_yaw: float,
        max_linear: float = 0.45,
        max_angular: float = 1.15,
        position_tolerance: float = 0.12,
        yaw_tolerance: float = 0.12,
    ) -> None:
        if len(path) < 2:
            raise ValueError("path needs start and goal")
        self.path = list(path)
        self.goal_yaw = goal_yaw
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.index = 1
        self.done = False

    def command(self, pose: Pose2D) -> tuple[float, float, str]:
        if self.done:
            return 0.0, 0.0, "arrived"
        while self.index < len(self.path) - 1:
            if math.dist((pose.x, pose.y), self.path[self.index]) > 0.18:
                break
            self.index += 1

        goal = self.path[self.index]
        distance = math.dist((pose.x, pose.y), goal)
        if self.index == len(self.path) - 1 and distance <= self.position_tolerance:
            yaw_error = wrap_angle(self.goal_yaw - pose.yaw)
            if abs(yaw_error) <= self.yaw_tolerance:
                self.done = True
                return 0.0, 0.0, "arrived"
            return 0.0, max(-self.max_angular, min(self.max_angular, 1.8 * yaw_error)), "align"

        desired = math.atan2(goal[1] - pose.y, goal[0] - pose.x)
        heading_error = wrap_angle(desired - pose.yaw)
        angular = max(-self.max_angular, min(self.max_angular, 2.2 * heading_error))
        if abs(heading_error) > 0.55:
            return 0.0, angular, "turn"
        linear = min(self.max_linear, 0.75 * distance)
        linear *= max(0.20, math.cos(heading_error))
        return linear, angular, "follow"


def path_length(points: Iterable[tuple[float, float]]) -> float:
    points = list(points)
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))
