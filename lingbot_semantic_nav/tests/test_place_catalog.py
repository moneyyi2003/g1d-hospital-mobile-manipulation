import json
from pathlib import Path
import tempfile
import unittest

from lingbot_nav.errors import ConfigurationError
from lingbot_nav.models import RouteAction
from lingbot_nav.place_db import PlaceDatabase


def place_value(place_id: str, x: float, status: str = "approved") -> dict:
    return {
        "id": place_id,
        "name": place_id.upper(),
        "aliases": [place_id, place_id.upper()],
        "status": status,
        "target": {"type": "sam3_instance", "source_id": f"track-{place_id}"},
        "docking_candidates": [
            {
                "id": f"dock-{place_id}",
                "pose": {"x": x, "y": 0.0, "yaw": 0.0, "frame_id": "map"},
                "checks": {
                    "clearance_m": 0.42,
                    "footprint_radius_m": 0.22,
                    "occupancy_status": "free",
                    "reachable": True,
                },
                "review": {"status": "accepted"},
            }
        ],
        "selected_docking_candidate": f"dock-{place_id}",
    }


def catalog(*places: dict) -> dict:
    return {
        "schema_version": 2,
        "map": {"id": "office-1", "sha256": "a" * 64, "frame_id": "map"},
        "places": list(places),
    }


class PlaceCatalogTest(unittest.TestCase):
    def write(self, payload: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "places.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_only_approved_places_enter_navigation_catalog(self):
        path = self.write(catalog(place_value("a", 1.0), place_value("draft", 2.0, "candidate")))
        places = PlaceDatabase.load(path)
        self.assertEqual([item.place_id for item in places.places], ["a"])
        self.assertEqual(places.resolve("A").place.entrance_pose.x, 1.0)

    def test_map_identity_is_enforced(self):
        path = self.write(catalog(place_value("a", 1.0)))
        with self.assertRaises(ConfigurationError):
            PlaceDatabase.load(path, expected_map_id="another-map")
        with self.assertRaises(ConfigurationError):
            PlaceDatabase.load(path, expected_map_sha256="b" * 64)

    def test_unsafe_selected_candidate_cannot_be_approved(self):
        value = place_value("a", 1.0)
        value["docking_candidates"][0]["checks"]["occupancy_status"] = "unknown"
        path = self.write(catalog(value))
        with self.assertRaises(ConfigurationError):
            PlaceDatabase.load(path)

    def test_legacy_catalog_requires_explicit_opt_in(self):
        legacy = {
            "schema_version": 1,
            "frame_id": "map",
            "places": [{
                "id": "a",
                "name": "A",
                "aliases": ["A"],
                "entrance_pose": {"x": 0, "y": 0, "yaw": 0},
            }],
        }
        path = self.write(legacy)
        with self.assertRaises(ConfigurationError):
            PlaceDatabase.load(path)
        self.assertEqual(PlaceDatabase.load(path, allow_legacy=True).places[0].place_id, "a")

    def test_reviewed_approach_pose_precedes_arrival_only(self):
        value = place_value("a", 1.0)
        value["docking_candidates"][0]["checks"]["approach_pose"] = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "frame_id": "map",
        }
        place = PlaceDatabase.load(self.write(catalog(value))).places[0]

        self.assertEqual(
            [pose.x for pose in place.navigation_poses(RouteAction.ARRIVE)],
            [0.0, 1.0],
        )
        self.assertEqual(
            [pose.x for pose in place.navigation_poses(RouteAction.PASS)],
            [1.0],
        )


if __name__ == "__main__":
    unittest.main()
