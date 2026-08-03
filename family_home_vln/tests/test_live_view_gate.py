from __future__ import annotations

import unittest

from family_home_vln.live_object_search import manipulation_view_gate


class ManipulationViewGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "frames": [
                {
                    "frame": 2,
                    "robot_pose": {"x": 1.0, "y": 2.0, "yaw": 0.25},
                }
            ]
        }

    def test_accepts_centered_target_and_returns_pose(self) -> None:
        result = {
            "live_matches": [
                {
                    "example": {
                        "frame_index": 2,
                        "bbox": [200.0, 100.0, 440.0, 330.0],
                    }
                }
            ]
        }
        gate = manipulation_view_gate(
            result,
            self.manifest,
            image_size=(640, 480),
        )
        self.assertTrue(gate["ready"])
        self.assertEqual(gate["selected"]["robot_pose"]["yaw"], 0.25)

    def test_rejects_target_touching_image_edge(self) -> None:
        result = {
            "live_matches": [
                {
                    "example": {
                        "frame_index": 2,
                        "bbox": [2.0, 10.0, 220.0, 260.0],
                    }
                }
            ]
        }
        gate = manipulation_view_gate(
            result,
            self.manifest,
            image_size=(640, 480),
        )
        self.assertFalse(gate["ready"])
        self.assertEqual(gate["failure_code"], "bad_viewpoint")


if __name__ == "__main__":
    unittest.main()
