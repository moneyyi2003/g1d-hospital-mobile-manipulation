import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from lingbot_nav.mapping.metric_map import build_metric_occupancy_map
from lingbot_nav.mapping.occupancy import (
    OccupancyBuildConfig,
    OccupancyGrid,
    clear_traversed_footprints,
)


@unittest.skipIf(np is None, "NumPy mapping extra is not installed")
class MetricMapTest(unittest.TestCase):
    def test_clears_collision_validated_robot_footprint(self):
        cells = np.full((9, 9), 100, dtype=np.int8)
        grid = OccupancyGrid(cells, 0.1, -0.45, -0.45, OccupancyBuildConfig())

        cleared = clear_traversed_footprints(grid, [[0.0, 0.0]], radius_m=0.21)

        self.assertEqual(cleared.cells[4, 4], 0)
        self.assertEqual(cleared.cells[0, 0], 100)
        self.assertEqual(grid.cells[4, 4], 100)

    def test_builds_ros_map_from_lingbot_artifacts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        predictions = root / "predictions"
        predictions.mkdir()
        rows, cols = np.indices((20, 20))
        points = np.stack((cols * 0.05, rows * 0.05, np.zeros_like(rows)), axis=-1).astype(np.float32)
        points[8:12, 8:12, 2] = 0.5
        np.savez_compressed(
            predictions / "frame_000000.npz",
            images=np.full((20, 20, 3), 127, dtype=np.uint8),
            world_points=points,
            world_points_conf=np.ones((20, 20), dtype=np.float32),
        )
        alignment = root / "alignment.json"
        alignment.write_text(json.dumps({"matrix": np.eye(4).tolist()}), encoding="utf-8")
        output = root / "map"
        manifest = build_metric_occupancy_map(
            predictions,
            alignment,
            output,
            scale_m_per_unit=1.0,
            resolution_m=0.05,
        )
        self.assertTrue((output / "map.pgm").is_file())
        self.assertTrue((output / "map.yaml").is_file())
        self.assertFalse(manifest["ground_truth_inputs"]["habitat_depth"])
        self.assertGreater(manifest["map"]["cell_counts"]["occupied"], 0)


if __name__ == "__main__":
    unittest.main()
