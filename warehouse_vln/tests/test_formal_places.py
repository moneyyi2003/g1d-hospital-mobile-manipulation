import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LINGBOT_SRC = ROOT / "lingbot_semantic_nav/src"
if str(LINGBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LINGBOT_SRC))

from lingbot_nav.place_catalog_builder import map_bundle_sha256
from lingbot_nav.place_db import PlaceDatabase
from simple_room_vln.artifacts import load_lingbot_artifacts
from simple_room_vln.core import path_length
from warehouse_vln.artifacts import (
    ROBOT_RADIUS_M,
    WAREHOUSE_START,
    plan_docking_path,
)
from warehouse_vln.formal_places import build_formal_place_catalog


class FormalWarehousePlacesTest(unittest.TestCase):
    def test_only_footprint_safe_reachable_places_are_approved(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        resolution = 0.25
        origin = (-15.0, -15.0)
        width = height = 120
        pixels = bytearray([255] * (width * height))

        # Make a measured unknown/occupied patch around the requested loading
        # zone. The east/west shelf destinations and the G1-D start stay free.
        for grid_row in range(height):
            y = origin[1] + (grid_row + 0.5) * resolution
            for col in range(width):
                x = origin[0] + (col + 0.5) * resolution
                if abs(x - 4.0) <= 1.25 and abs(y + 10.0) <= 1.25:
                    pgm_row = height - 1 - grid_row
                    pixels[pgm_row * width + col] = 0

        pgm = root / "map.pgm"
        pgm.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels)
        map_yaml = root / "map.yaml"
        map_yaml.write_text(
            "\n".join(
                (
                    "image: map.pgm",
                    f"resolution: {resolution}",
                    f"origin: [{origin[0]}, {origin[1]}, 0.0]",
                    "negate: 0",
                    "occupied_thresh: 0.65",
                    "free_thresh: 0.196",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        alignment = root / "alignment.json"
        alignment.write_text(
            json.dumps(
                {
                    "artifact_type": "lingbot_depth_to_metric_survey_pose_anchor",
                    "scale_m_per_unit": 1.0,
                }
            ),
            encoding="utf-8",
        )
        semantics = root / "sam3_observations.json"
        semantics.write_text(
            json.dumps(
                {
                    "frame_id": "map",
                    "observations": [
                        {
                            "track_id": "warehouse-shelf-1",
                            "prompt": "warehouse shelf",
                            "frame_index": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        catalog_path = root / "places.json"
        catalog = build_formal_place_catalog(
            map_yaml,
            semantics,
            alignment,
            catalog_path,
        )

        statuses = {item["id"]: item["status"] for item in catalog["places"]}
        self.assertEqual(statuses["east_shelf_aisle"], "approved")
        self.assertEqual(statuses["west_shelf_aisle"], "approved")
        self.assertEqual(statuses["loading_zone"], "rejected")
        database = PlaceDatabase.load(
            catalog_path,
            expected_map_id="isaac-mobilemanibench-warehouse-lingbot-sam3-v1",
            expected_map_sha256=map_bundle_sha256(map_yaml),
        )
        self.assertEqual(
            {place.place_id for place in database.places},
            {"east_shelf_aisle", "west_shelf_aisle"},
        )
        self.assertEqual(
            database.resolve("带我到东侧货架通道").place.place_id,
            "east_shelf_aisle",
        )
        grid, approved = load_lingbot_artifacts(
            map_yaml,
            catalog_path,
            robot_radius_m=ROBOT_RADIUS_M,
        )
        east = next(
            place for place in approved if place.place_id == "east_shelf_aisle"
        )
        route = plan_docking_path(
            grid,
            (WAREHOUSE_START.x, WAREHOUSE_START.y),
            east.pose,
        )
        reviewed_length = next(
            item["review"]["planned_path_length_m"]
            for item in catalog["places"]
            if item["id"] == "east_shelf_aisle"
        )
        self.assertAlmostEqual(path_length(route), reviewed_length)


if __name__ == "__main__":
    unittest.main()
