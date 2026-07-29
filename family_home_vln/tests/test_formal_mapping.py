import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from family_home_vln.formal_mapping import (
    SEMANTIC_LABELS,
    build_formal_place_catalog,
    build_scan_semantic_layers,
)


class FamilyHomeFormalMappingTest(unittest.TestCase):
    def make_inputs(self, root: Path, *, prompts=SEMANTIC_LABELS):
        map_root = root / "map"
        map_root.mkdir()
        width = height = 120
        pixels = bytearray([254]) * (width * height)
        (map_root / "map.pgm").write_bytes(
            f"P5\n{width} {height}\n255\n".encode() + pixels
        )
        map_yaml = map_root / "map.yaml"
        map_yaml.write_text(
            "image: map.pgm\nresolution: 0.1\norigin: [-6.0, -6.0, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
            encoding="utf-8",
        )
        anchors = {
            "sofa": (-2.7, 1.2),
            "couch": (-2.7, 1.2),
            "bed": (-2.7, -2.2),
            "dining table": (2.0, 3.0),
            "kitchen counter": (3.6, 3.8),
            "book": (-0.2, 3.6),
        }
        observations = []
        for index, prompt in enumerate(prompts):
            x, y = anchors[prompt]
            observations.append({
                "track_id": f"{index}:1",
                "prompt": prompt,
                "frame_index": index,
                "score": 0.9,
                "point_count": 500,
                "centroid_xyz": [x, y, 0.7],
                "minimum_xyz": [x - 0.4, y - 0.3, 0.1],
                "maximum_xyz": [x + 0.4, y + 0.3, 1.2],
            })
        evidence = root / "observations.json"
        evidence.write_text(
            json.dumps({"schema_version": 1, "frame_id": "map", "observations": observations}),
            encoding="utf-8",
        )
        alignment = root / "alignment.json"
        alignment.write_text(
            json.dumps({"artifact_type": "lingbot_depth_to_metric_survey_pose_anchor"}),
            encoding="utf-8",
        )
        return map_yaml, evidence, alignment

    def test_layers_are_derived_from_observations_and_free_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_yaml, evidence, _alignment = self.make_inputs(root)
            output = root / "semantic"

            metadata = build_scan_semantic_layers(map_yaml, evidence, output)
            semantic = np.load(output / "semantic_map.npy")
            regions = np.load(output / "region_map.npy")

            self.assertEqual(set(metadata["anchors"]), set(SEMANTIC_LABELS))
            self.assertGreater(int((semantic > 0).sum()), 0)
            self.assertEqual(set(np.unique(regions)) - {0}, {1, 2, 3, 4})
            self.assertFalse(metadata["isaac_fixture_geometry_used"])

    def test_layers_include_labels_discovered_beyond_navigation_ontology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = (*SEMANTIC_LABELS, "book")
            map_yaml, evidence, _alignment = self.make_inputs(root, prompts=prompts)

            metadata = build_scan_semantic_layers(
                map_yaml, evidence, root / "semantic"
            )

            self.assertIn("book", metadata["anchors"])
            self.assertIn("book", metadata["labels"].values())

    def test_post_discovery_alias_can_approve_navigation_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = ("couch", "bed", "dining table", "kitchen counter")
            map_yaml, evidence, alignment = self.make_inputs(root, prompts=prompts)
            semantic = root / "semantic"
            build_scan_semantic_layers(map_yaml, evidence, semantic)

            payload = build_formal_place_catalog(
                map_yaml,
                evidence,
                alignment,
                semantic / "region_map.npy",
                root / "places.json",
                household_object_set_signature="test-signature",
            )

            sofa = next(
                item for item in payload["places"] if item["id"] == "living_room_sofa"
            )
            self.assertEqual(sofa["metadata"]["semantic_prompt"], "couch")
            self.assertEqual(
                payload["map"]["household_object_set_signature"], "test-signature"
            )

    def test_places_require_matching_semantic_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_yaml, evidence, alignment = self.make_inputs(
                root, prompts=tuple(item for item in SEMANTIC_LABELS if item != "bed")
            )
            semantic = root / "semantic"
            build_scan_semantic_layers(map_yaml, evidence, semantic)

            payload = build_formal_place_catalog(
                map_yaml,
                evidence,
                alignment,
                semantic / "region_map.npy",
                root / "places.json",
            )

            status = {item["id"]: item["status"] for item in payload["places"]}
            self.assertEqual(status["bedroom_bed"], "rejected")
            self.assertEqual(status["living_room_sofa"], "approved")
            approved = next(item for item in payload["places"] if item["id"] == "living_room_sofa")
            self.assertEqual(approved["metadata"]["semantic_prompt"], "sofa")
            self.assertNotIn("requested_scene_pose", approved["metadata"])


if __name__ == "__main__":
    unittest.main()
