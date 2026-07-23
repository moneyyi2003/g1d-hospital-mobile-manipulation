import json
from pathlib import Path
import tempfile
import unittest

from lingbot_nav.models import Pose2D
from lingbot_nav.place_catalog_builder import approve_place, build_candidate_catalog
from lingbot_nav.place_db import PlaceDatabase


class PlaceBuilderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        width = height = 31
        pixels = bytearray([254] * (width * height))
        for row in range(height):
            for col in range(width):
                if row in {0, height - 1} or col in {0, width - 1}:
                    pixels[row * width + col] = 0
        self.pgm = self.root / "map.pgm"
        self.pgm.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels)
        self.yaml = self.root / "map.yaml"
        self.yaml.write_text(
            "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
            encoding="utf-8",
        )
        self.observations = self.root / "observations.json"
        self.observations.write_text(json.dumps({
            "schema_version": 1,
            "frame_id": "map",
            "observations": [
                {
                    "track_id": "p0:o1",
                    "prompt": "sofa",
                    "frame_index": 0,
                    "score": 0.9,
                    "point_count": 500,
                    "centroid_xyz": [1.5, 1.5, 0.4],
                    "minimum_xyz": [1.2, 1.2, 0.0],
                    "maximum_xyz": [1.8, 1.8, 0.8],
                }
            ],
        }), encoding="utf-8")

    def test_candidate_must_be_explicitly_approved(self):
        catalog_path = self.root / "places.json"
        catalog = build_candidate_catalog(
            self.observations,
            self.yaml,
            catalog_path,
            map_id="test-map",
            reachability_start=Pose2D(1.0, 1.0),
            footprint_radius_m=0.15,
        )
        self.assertGreater(len(catalog["places"]), 0)
        self.assertEqual(catalog["places"][0]["status"], "candidate")
        with self.assertRaises(Exception):
            PlaceDatabase.load(catalog_path)

        place = catalog["places"][0]
        candidate = place["docking_candidates"][0]
        approve_place(
            catalog_path,
            self.yaml,
            place_id=place["id"],
            candidate_id=candidate["id"],
            reviewer="operator",
            evidence=["review.png"],
        )
        database = PlaceDatabase.load(catalog_path)
        self.assertEqual(database.places[0].place_id, place["id"])


if __name__ == "__main__":
    unittest.main()
