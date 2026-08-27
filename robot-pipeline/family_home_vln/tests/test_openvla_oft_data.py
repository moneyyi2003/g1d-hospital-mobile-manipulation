import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.g1d_openvla_oft_data import ACTION_CHUNK, build_manifest
from scripts.collect_expert_demos_head import _meta_is_physical_training_ready


class OpenVLAOFTDataTest(unittest.TestCase):
    def test_collector_does_not_count_legacy_fixed_joint_episode(self):
        legacy = {
            "success": True,
            "ready_for_training": True,
            "expert_evidence": {
                "success": True,
                "fixed_joint_created": True,
                "stable_hold_frames": 30,
            },
        }
        self.assertFalse(_meta_is_physical_training_ready(legacy))

    def _episode(
        self,
        root: Path,
        *,
        black: bool = False,
        unnorm_key: str = "g1d_family_home_cup_head",
        fixed_joint: bool = False,
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
                "hold_contact_frames": 30,
                "physical_hold_verified": True,
                "fixed_joint_created": fixed_joint,
                "fixed_joint_configured": fixed_joint,
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

    def test_discovers_nested_collection_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._episode(root / "shard_gpu6")
            payload = build_manifest(root, root / "manifest.json")
            self.assertEqual(payload["episode_count"], 1)
            self.assertEqual(payload["episodes"][0]["episode"], "shard_gpu6/episode_0000")

    def test_rejects_fixed_joint_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._episode(root, fixed_joint=True)
            with self.assertRaisesRegex(RuntimeError, "fixed-joint grasp is forbidden"):
                build_manifest(root, root / "manifest.json")

    def test_ignores_internal_expert_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._episode(root)
            internal = root / "expert_run_0000" / "episode_0000"
            internal.mkdir(parents=True)
            payload = build_manifest(root, root / "manifest.json")
            self.assertEqual(payload["episode_count"], 1)
            self.assertEqual(payload["rejected"], [])


if __name__ == "__main__":
    unittest.main()
