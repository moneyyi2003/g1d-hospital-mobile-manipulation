import json
import tempfile
import unittest
from pathlib import Path

from family_home_vln.discovery import (
    aggregate_detections,
    run_object_discovery,
    sample_frame_indices,
    normalize_label,
    validate_survey,
)


class ObjectDiscoveryTest(unittest.TestCase):
    def make_survey(self, root: Path, frame_count: int = 4):
        rgb = root / "survey/rgb"
        rgb.mkdir(parents=True)
        frames = []
        for index in range(frame_count):
            name = f"{index:06d}.png"
            (rgb / name).write_bytes(b"rgb")
            frames.append({"frame": index, "image": f"rgb/{name}"})
        manifest = root / "survey/capture_manifest.json"
        manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "object_category_labels_supplied_to_perception": False,
                "camera": {"resolution": [100, 80]},
                "frames": frames,
            }),
            encoding="utf-8",
        )
        return manifest, rgb

    def test_sampling_is_uniform_and_bounded(self):
        self.assertEqual(sample_frame_indices(4, 8), [0, 1, 2, 3])
        sampled = sample_frame_indices(101, 6)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 100)
        self.assertEqual(len(sampled), 6)

    def test_descriptive_labels_merge_without_using_scene_truth(self):
        self.assertEqual(normalize_label("black monitor"), "monitor")
        self.assertEqual(normalize_label("computer monitor"), "monitor")
        self.assertEqual(normalize_label("brown sofa"), "sofa")

    def test_aggregation_requires_repeated_visual_evidence(self):
        objects, rejected = aggregate_detections(
            [
                {"frame_index": 0, "label": "A Cup", "bbox": [10, 10, 30, 40], "task": "<OD>"},
                {"frame_index": 1, "label": "cups", "bbox": [12, 10, 32, 40], "task": "<OD>"},
                {"frame_index": 0, "label": "wall", "bbox": [0, 0, 90, 70], "task": "<OD>"},
                {"frame_index": 2, "label": "mystery", "bbox": [20, 20, 30, 30], "task": "<OD>"},
            ],
            image_size=(100, 80),
        )
        self.assertEqual([item["label"] for item in objects], ["cup"])
        self.assertEqual(objects[0]["frame_occurrences"], 2)
        self.assertTrue(any(item["reason"] == "invalid_or_structural_label" for item in rejected))

    def test_discovery_backend_receives_only_task_token_and_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rgb = self.make_survey(root)
            calls = []

            def infer(path, task):
                calls.append((path.name, task))
                return [{"label": "book", "bbox": [10, 10, 40, 50]}]

            payload = run_object_discovery(
                manifest,
                rgb,
                root / "objects.json",
                model_path=root / "model",
                infer=infer,
            )

            self.assertEqual(
                payload["truth_boundary"]["category_prompt_list_supplied"],
                False,
            )
            self.assertEqual([item["label"] for item in payload["objects"]], ["book"])
            self.assertTrue(all(task.startswith("<") and task.endswith(">") for _, task in calls))

    def test_incomplete_survey_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rgb = self.make_survey(root)
            (rgb / "000003.png").unlink()
            with self.assertRaisesRegex(ValueError, "巡检不完整"):
                validate_survey(manifest, rgb)


if __name__ == "__main__":
    unittest.main()
