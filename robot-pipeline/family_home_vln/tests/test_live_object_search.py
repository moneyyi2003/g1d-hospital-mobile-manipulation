import json
import tempfile
import unittest
from pathlib import Path

from family_home_vln.live_object_search import (
    load_reviewed_object,
    manipulation_view_gate,
    match_live_discovery,
    search_live_rgb,
)


class LiveObjectSearchTest(unittest.TestCase):
    def make_inputs(self, root: Path):
        rgb = root / "rgb"
        rgb.mkdir()
        frames = []
        for index in range(3):
            name = f"{index:06d}.png"
            (rgb / name).write_bytes(b"fake-rgb")
            frames.append({"frame": index, "image": f"rgb/{name}"})
        manifest = root / "capture_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "object_category_labels_supplied_to_perception": False,
                    "camera": {"resolution": [100, 80]},
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
        catalog = root / "objects.json"
        catalog.write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "object_id": "scan_houseplant_01",
                            "source_label": "houseplant",
                            "aliases": ["houseplant", "plant", "盆栽"],
                            "status": "approved",
                            "map_position": {
                                "x": 1.0,
                                "y": 2.0,
                                "frame_id": "map",
                            },
                        },
                        {
                            "source_label": "bathtub",
                            "aliases": ["bathtub"],
                            "status": "rejected",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return manifest, rgb, catalog

    def test_query_only_resolves_reviewed_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _manifest, _rgb, catalog = self.make_inputs(root)
            item = load_reviewed_object(catalog, "盆栽")
            self.assertEqual(item["object_id"], "scan_houseplant_01")
            with self.assertRaises(ValueError):
                load_reviewed_object(catalog, "bathtub")

    def test_live_model_is_category_free_then_matches_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rgb, catalog = self.make_inputs(root)
            calls = []

            def infer(path, task):
                calls.append((path.name, task))
                return [
                    {
                        "label": "green houseplant",
                        "bbox": [10, 10, 40, 60],
                    }
                ]

            result = search_live_rgb(
                manifest,
                rgb,
                catalog,
                "scan_houseplant_01",
                root / "result/search.json",
                model_path=root / "unused-model",
                infer=infer,
            )

            self.assertTrue(result["success"])
            self.assertFalse(
                result["inference"]["category_list_supplied_to_model"]
            )
            self.assertTrue(
                result["inference"]["target_used_only_after_inference_for_matching"]
            )
            self.assertTrue(
                all(task in ("<OD>", "<DENSE_REGION_CAPTION>") for _, task in calls)
            )

    def test_cup_does_not_match_generic_container_alias(self):
        target = {
            "source_label": "coffee cup",
            "aliases": ["cup", "mug", "box", "container"],
        }
        discovery = {
            "objects": [
                {"label": "box"},
                {"label": "coffee mug"},
            ]
        }
        matches = match_live_discovery(target, discovery)
        self.assertEqual([item["label"] for item in matches], ["coffee mug"])

    def test_target_phrase_grounding_recovers_category_free_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rgb, catalog = self.make_inputs(root)
            payload = json.loads(catalog.read_text(encoding="utf-8"))
            payload["objects"][0].update(
                {
                    "object_id": "scan_coffee_cup_05",
                    "source_label": "coffee cup",
                    "aliases": ["coffee cup", "cup", "mug"],
                }
            )
            catalog.write_text(json.dumps(payload), encoding="utf-8")

            def infer(_path, task):
                if task.startswith("<CAPTION_TO_PHRASE_GROUNDING>"):
                    return [
                        {"label": "coffee cup", "bbox": [20, 10, 50, 60]},
                        {"label": "coffee cup", "bbox": [82, 10, 99, 60]},
                    ]
                return [{"label": "table", "bbox": [5, 5, 95, 70]}]

            result = search_live_rgb(
                manifest,
                rgb,
                catalog,
                "scan_coffee_cup_05",
                root / "result/search.json",
                model_path=root / "unused-model",
                infer=infer,
            )
            self.assertTrue(result["success"])
            self.assertEqual(len(result["live_matches"]), 2)
            self.assertTrue(result["inference"]["target_phrase_grounding_used"])
            self.assertFalse(result["inference"]["usd_semantics_read"])
            gate = manipulation_view_gate(
                result,
                json.loads(manifest.read_text(encoding="utf-8")),
                image_size=(100, 80),
            )
            self.assertTrue(gate["ready"])
            self.assertEqual(gate["selected"]["bbox"], [20.0, 10.0, 50.0, 60.0])


if __name__ == "__main__":
    unittest.main()
