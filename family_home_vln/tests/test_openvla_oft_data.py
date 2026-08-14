import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.g1d_openvla_oft_data import ACTION_CHUNK, build_manifest


class OpenVLAOFTDataTest(unittest.TestCase):
    def _episode(
        self,
        root: Path,
        *,
        black: bool = False,
        unnorm_key: str = "g1d_family_home_cup_head",
    ) -> None:
        episode = root / "episode_0000"
        episode.mkdir(parents=True)
        meta = {
            "instruction": "pick up the cup",
            "object_id": "cup",
            "success": True,
            "ready_for_training": True,
            "camera_mode": "ego_centric_head",
            "capture_hz": 10,
            "camera_intrinsics": {
                "near_clip_m": 0.1,
                "far_clip_m": 1_000_000.0,
            },
            "expert_evidence": {
                "success": True,
                "physical_execution": True,
                "lift_height_m": 0.2,
                "stable_hold_frames": 30,
            },
        }
        (episode / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for index in range(ACTION_CHUNK):
            step = episode / f"step_{index:04d}"
            step.mkdir()
            image = np.full((32, 32, 3), 128, dtype=np.uint8)
            if black:
                image[12:, :, :] = 0
            else:
                image[:, :16, 1] = 180
            image = np.repeat(np.repeat(image, 15, axis=0), 20, axis=1)
            Image.fromarray(image).save(step / "image.png")
            action = {
                "dx_m": 0.001,
                "dy_m": 0.0,
                "dz_m": 0.001,
                "droll_rad": 0.0,
                "dpitch_rad": 0.0,
                "dyaw_rad": 0.0,
                "gripper": 1.0,
                "frame": "world",
                "unnorm_key": unnorm_key,
            }
            (step / "action.json").write_text(json.dumps(action), encoding="utf-8")

    def test_builds_one_action_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._episode(root)
            payload = build_manifest(root, root / "manifest.json")
            self.assertEqual(payload["episode_count"], 1)
            self.assertEqual(payload["sample_count"], 1)

    def test_rejects_large_black_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._episode(root, black=True)
            with self.assertRaisesRegex(RuntimeError, "large black region"):
                build_manifest(root, root / "manifest.json")

    def test_rejects_mismatched_action_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._episode(root, unnorm_key="bridge_orig")
            with self.assertRaisesRegex(RuntimeError, "unexpected unnorm_key"):
                build_manifest(root, root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
