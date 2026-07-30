import json
import tempfile
import unittest
from pathlib import Path

from family_home_vln.live_object_search import (
    load_reviewed_object,
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


if __name__ == "__main__":
    unittest.main()
