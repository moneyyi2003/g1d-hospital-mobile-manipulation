import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from family_home_vln.formal_mapping import PROMPT_BY_PLACE
from family_home_vln.household_objects import OBJECT_SET_SIGNATURE
from family_home_vln.layout import PLACES
from scripts.serve_family_home_dashboard import FamilyHomeDashboardSession


class FamilyHomeDashboardSessionTest(unittest.TestCase):
    def write_formal_bundle(self, root: Path, *, omit_layer: str | None = None) -> None:
        artifacts = root / "artifacts"
        map_root = artifacts / "lingbot_map"
        preview = artifacts / "map_preview"
        map_root.mkdir(parents=True)
        preview.mkdir(parents=True)
        width = height = 100
        (map_root / "map.pgm").write_bytes(
            f"P5\n{width} {height}\n255\n".encode() + bytes([254]) * width * height
        )
        (map_root / "map.yaml").write_text(
            "image: map.pgm\nresolution: 0.1\norigin: [-5.0, -5.0, 0.0]\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256()
        for path in (map_root / "map.yaml", map_root / "map.pgm"):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        places = []
        for place in PLACES:
            pose = {
                "x": place.pose.x,
                "y": place.pose.y,
                "yaw": place.pose.yaw,
                "frame_id": "map",
            }
            places.append({
                "id": place.place_id,
                "name": place.name,
                "aliases": list(place.aliases),
                "target": {
                    "type": "semantic_instance",
                    "source_id": PROMPT_BY_PLACE[place.place_id],
                },
                "status": "approved",
                "entrance_pose": pose,
                "docking_candidates": [{"id": "scan", "pose": pose}],
                "selected_docking_candidate": "scan",
            })
        (artifacts / "places_formal.json").write_text(
            json.dumps({
                "schema_version": 2,
                "map": {
                    "sha256": digest.hexdigest(),
                    "household_object_set_signature": OBJECT_SET_SIGNATURE,
                },
                "places": places,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        (artifacts / "objects_formal.json").write_text(
            json.dumps({
                "schema_version": 1,
                "objects": [{
                    "object_id": "scan_cup_06",
                    "source_label": "cup",
                    "aliases": ["杯子", "水杯"],
                    "status": "approved",
                    "manipulation_ready": True,
                }],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        survey_manifest = artifacts / "survey/capture_manifest.json"
        survey_manifest.parent.mkdir(parents=True)
        survey_manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "rgb_is_only_model_input": True,
                "camera": {"resolution": [640, 360]},
                "frames": [{"frame": 0}, {"frame": 1}],
            }),
            encoding="utf-8",
        )
        sam3_manifest = artifacts / "sam3/sam3_manifest.json"
        sam3_manifest.parent.mkdir(parents=True)
        sam3_manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "prompts": [
                    {
                        "prompt": prompt,
                        "detections": 3 if prompt == "sofa" else 0,
                    }
                    for prompt in PROMPT_BY_PLACE.values()
                ],
            }),
            encoding="utf-8",
        )
        discovery = artifacts / "discovery/object_discovery.json"
        discovery.parent.mkdir(parents=True)
        discovery.write_text(
            json.dumps({
                "schema_version": 1,
                "truth_boundary": {
                    "category_prompt_list_supplied": False,
                    "labels_generated_by_model": True,
                },
                "objects": [
                    {
                        "label": "sofa",
                        "frame_occurrences": 2,
                        "raw_detection_count": 3,
                    },
                    {
                        "label": "book",
                        "frame_occurrences": 2,
                        "raw_detection_count": 2,
                    },
                ],
            }),
            encoding="utf-8",
        )
        observations = artifacts / "semantic/sam3_observations.json"
        observations.parent.mkdir(parents=True)
        observations.write_text(
            json.dumps({
                "schema_version": 1,
                "frame_id": "map",
                "observations": [{
                    "track_id": "sofa:0",
                    "prompt": "sofa",
                    "score": 0.9,
                    "point_count": 100,
                    "centroid_xyz": [-2.0, 0.3, 0.2],
                }],
            }),
            encoding="utf-8",
        )
        (observations.parent / "semantic_metadata.json").write_text(
            json.dumps({
                "anchors": {"sofa": [-2.0, 0.3]},
                "region_labels": {"1": "sofa"},
            }),
            encoding="utf-8",
        )
        assets = {}
        for layer in ("rgb_pointcloud", "semantic", "occupancy", "region"):
            if layer == omit_layer:
                continue
            path = preview / f"{layer}.png"
            path.write_bytes(b"fake-png")
            assets[layer] = str(path)
        (artifacts / "mapping_summary.json").write_text(
            json.dumps({
                "schema_version": 1,
                "assets": assets,
                "map": {
                    "width": width,
                    "height": height,
                    "resolution": 0.1,
                    "flip_y": True,
                    "bounds": {
                        "min_x": -5.0,
                        "max_x": 5.0,
                        "min_z": -5.0,
                        "max_z": 5.0,
                    },
                    "layers": [
                        {"id": layer, "description": f"formal {layer}"}
                        for layer in assets
                    ],
                },
                "inputs": {
                    "survey_manifest": str(survey_manifest),
                    "discovery": str(discovery),
                    "sam3": str(sam3_manifest),
                    "semantic_observations": str(observations),
                },
            }),
            encoding="utf-8",
        )

    def make_session(self, root: Path) -> FamilyHomeDashboardSession:
        return FamilyHomeDashboardSession(
            argparse.Namespace(
                artifacts=root / "artifacts",
                output=root / "web-output",
            )
        )

    def test_missing_formal_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "拒绝使用 bootstrap"):
                self.make_session(Path(directory))

    def test_config_exposes_four_formal_scan_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_formal_bundle(root)
            session = self.make_session(root)
            config = session.config()
            data = session.map_data()

            self.assertEqual(
                {layer["id"] for layer in config["layers"]},
                {"pointcloud", "semantic", "occupancy", "region"},
            )
            self.assertTrue(all(layer["status"] == "formal" for layer in config["layers"]))
            self.assertEqual(config["map"]["source_status"], "formal")
            self.assertIsNone(data["truth_boundary"])
            self.assertIn("lingbot_rgb_only", data["source"])
            recognition = config["recognition"]
            self.assertEqual(recognition["survey"]["frame_count"], 2)
            self.assertEqual(recognition["summary"]["discovered_categories"], 2)
            self.assertEqual(recognition["summary"]["mapped_categories"], 1)
            self.assertEqual(recognition["objects"][0]["raw_detections"], 3)
            self.assertEqual(recognition["objects"][0]["map_observations"], 1)
            self.assertFalse(
                recognition["truth_boundary"]["category_prompt_list_supplied"]
            )

    def test_incomplete_four_layer_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_formal_bundle(root, omit_layer="semantic")
            with self.assertRaisesRegex(ValueError, "semantic"):
                self.make_session(root)

    def test_map_place_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_formal_bundle(root)
            map_path = root / "artifacts/lingbot_map/map.pgm"
            map_path.write_bytes(map_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.make_session(root)

    def test_plan_uses_formal_place_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_formal_bundle(root)
            session = self.make_session(root)

            target, path = session.plan("请带我到厨房操作台")

            self.assertEqual(target.place_id, "kitchen_counter")
            self.assertGreaterEqual(len(path), 2)
            with self.assertRaises(ValueError):
                session.plan("请带我去阳台")

    def test_interpret_compiles_dual_brain_household_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_formal_bundle(root)
            session = self.make_session(root)

            result = session.interpret(
                "请带我去餐桌，拿杯子，再回到客厅沙发旁"
            )

            self.assertEqual(result["mode"], "dual_brain_task")
            self.assertIn("OPENVLA_PICK", result["steps"])
            self.assertGreaterEqual(len(result["path"]), 2)

    def test_changed_household_object_set_keeps_report_but_blocks_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_formal_bundle(root)
            catalog = root / "artifacts/places_formal.json"
            payload = json.loads(catalog.read_text(encoding="utf-8"))
            payload["map"]["household_object_set_signature"] = "old-object-set"
            catalog.write_text(json.dumps(payload), encoding="utf-8")

            session = self.make_session(root)
            config = session.config()

            self.assertEqual(config["map"]["source_status"], "stale")
            self.assertEqual(config["recognition"]["summary"]["mapped_categories"], 0)
            with self.assertRaisesRegex(ValueError, "禁止导航"):
                session.plan("请带我到客厅沙发旁")


if __name__ == "__main__":
    unittest.main()
