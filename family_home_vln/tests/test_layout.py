import math
import tempfile
import unittest
from pathlib import Path

from family_home_vln.layout import (
    HOME_FIXTURES,
    PLACES,
    START_POSE,
    build_bootstrap_artifacts,
    build_grid,
    build_survey_path,
)
from simple_room_vln.core import path_length, resolve_place


class FamilyHomeLayoutTest(unittest.TestCase):
    def test_four_household_regions_are_reachable_and_footprint_safe(self):
        grid = build_grid()
        self.assertEqual(
            {item.place_id for item in PLACES},
            {
                "living_room_sofa",
                "bedroom_bed",
                "dining_area",
                "kitchen_counter",
            },
        )
        for place in PLACES:
            self.assertTrue(grid.is_free(grid.world_to_cell(place.pose.x, place.pose.y)))
            route = grid.plan(
                (START_POSE.x, START_POSE.y),
                (place.pose.x, place.pose.y),
            )
            self.assertGreaterEqual(len(route), 2)

    def test_language_never_generates_free_form_coordinates(self):
        self.assertEqual(
            resolve_place("我困了，请带我到卧室床边", PLACES).place_id,
            "bedroom_bed",
        )
        self.assertEqual(
            resolve_place("take me to the dining table", PLACES).place_id,
            "dining_area",
        )
        with self.assertRaises(ValueError):
            resolve_place("去一个没有审核过的阳台", PLACES)

    def test_survey_covers_all_zones_and_is_nontrivial(self):
        route = build_survey_path(build_grid())
        self.assertGreater(len(route), 8)
        self.assertGreater(path_length(route), 15.0)
        self.assertLess(path_length(route), 40.0)
        route_points = set(route)
        for place in PLACES:
            self.assertIn((place.pose.x, place.pose.y), route_points)

    def test_bootstrap_artifacts_are_explicitly_nonformal(self):
        with tempfile.TemporaryDirectory() as directory:
            grid, places = build_bootstrap_artifacts(Path(directory))
            self.assertEqual(len(places), 4)
            self.assertTrue((Path(directory) / "bootstrap_occupancy.json").is_file())
            self.assertTrue((Path(directory) / "places.json").is_file())
            self.assertTrue(grid.is_free(grid.world_to_cell(0.0, 0.0)))

    def test_layout_contains_real_partitions_and_household_furniture(self):
        categories = {item.category for item in HOME_FIXTURES}
        self.assertTrue({"wall", "bed", "dining_table", "kitchen_counter"} <= categories)
        for fixture in HOME_FIXTURES:
            self.assertTrue(all(math.isfinite(value) for value in fixture.center_xy))
            self.assertTrue(all(value > 0.0 for value in fixture.size_xyz))


if __name__ == "__main__":
    unittest.main()
