"""Collision-derived G1-D survey coverage for ``home_lab.usda``.

The grid is extracted from the scene's supplied PhysX collision mesh by
``scripts/extract_living_room_collision_grid.py``.  It is only a safe survey
bootstrap; LingBot-Map remains the source of the formal VLN occupancy map.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

from simple_room_vln.core import GridMap


ROOT = Path(__file__).resolve().parents[1]
GRID_ARTIFACT = ROOT / "outputs/living_room_vln/bootstrap_collision_grid.json"
SCENE_NAME = "home_lab"


def build_survey_grid() -> GridMap:
    if not GRID_ARTIFACT.is_file():
        raise FileNotFoundError(
            f"missing collision survey grid: {GRID_ARTIFACT}; run "
            "scripts/extract_living_room_collision_grid.py first"
        )
    payload = json.loads(GRID_ARTIFACT.read_text(encoding="utf-8"))
    origin_x, origin_y = payload["origin_xy_m"]
    return GridMap(
        payload["free"], resolution=float(payload["resolution_m"]),
        origin_x=float(origin_x), origin_y=float(origin_y),
    )


def _largest_component(grid: GridMap) -> set[tuple[int, int]]:
    unseen = {
        (row, col)
        for row in range(grid.height) for col in range(grid.width)
        if grid.is_free((row, col))
    }
    largest: set[tuple[int, int]] = set()
    while unseen:
        root = unseen.pop()
        component = {root}
        queue = deque([root])
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + dr, col + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        if len(component) > len(largest):
            largest = component
    if not largest:
        raise ValueError("collision grid contains no G1-D traversable cells")
    return largest


def build_survey_path(grid: GridMap) -> list[tuple[float, float]]:
    """Plan a connected, collision-cleared RGB coverage tour.

    A farthest-point selection spreads camera positions across the largest
    collision-free component.  Every transition is then A* planned on that
    same grid, so the assisted survey pose never jumps through furniture.
    """
    component = _largest_component(grid)
    requested_start = (0.0, -2.8)  # verified in-room camera region
    start = min(
        component,
        key=lambda cell: sum((left - right) ** 2 for left, right in zip(grid.cell_to_world(cell), requested_start)),
    )
    # Candidate cell spacing makes the tour compact but gives LingBot enough
    # parallax and SAM3 views from across the room.
    candidates = [cell for cell in component if cell[0] % 6 == 0 and cell[1] % 6 == 0]
    selected = [start]
    while candidates and len(selected) < 14:
        next_cell = max(
            candidates,
            key=lambda cell: min((cell[0] - item[0]) ** 2 + (cell[1] - item[1]) ** 2 for item in selected),
        )
        selected.append(next_cell)
        candidates.remove(next_cell)
    path = [grid.cell_to_world(start)]
    for cell in selected[1:]:
        goal = grid.cell_to_world(cell)
        leg = grid.plan(path[-1], goal)
        path.extend(leg[1:])
    return path
