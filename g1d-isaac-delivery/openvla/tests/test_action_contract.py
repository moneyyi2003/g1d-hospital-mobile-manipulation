from __future__ import annotations

import math
import json
from pathlib import Path
import tempfile
import unittest

from g1d_openvla.action_contract import (
    ActionContractError,
    OpenVlaAction,
    build_g1d_right_arm_handoff,
)
from g1d_openvla.checkpoint import inspect_checkpoint


class OpenVlaActionContractTest(unittest.TestCase):
    def test_preserves_cartesian_semantics_and_blocks_joint_execution(self) -> None:
        action = OpenVlaAction.from_values(
            [0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 1.0],
            unnorm_key="bridge_orig",
        )

        handoff = build_g1d_right_arm_handoff(action)

        self.assertEqual(handoff["model_action"]["labels"][0], "delta_x")
        self.assertEqual(handoff["model_action"]["labels"][-1], "gripper")
        self.assertEqual(handoff["target"]["side"], "right")
        self.assertEqual(
            handoff["target"]["joint_order"][-1],
            "right_wrist_yaw_joint",
        )
        self.assertIsNone(handoff["joint_command"])
        self.assertFalse(handoff["execution_permitted"])
        self.assertIn(
            "target_visibility_not_revalidated_in_final_openvla_frame",
            handoff["blocked_reasons"],
        )

    def test_rejects_wrong_dimension(self) -> None:
        with self.assertRaisesRegex(ActionContractError, "7 values"):
            OpenVlaAction.from_values([0.0] * 6, unnorm_key="bridge_orig")

    def test_rejects_non_finite_action(self) -> None:
        with self.assertRaisesRegex(ActionContractError, "non-finite"):
            OpenVlaAction.from_values(
                [0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 1.0],
                unnorm_key="bridge_orig",
            )

    def test_checkpoint_requires_all_declared_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 5},
                        "weight_map": {
                            "a": "model-00001.safetensors",
                            "b": "model-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "model-00001.safetensors").write_bytes(b"123")
            incomplete = inspect_checkpoint(root)
            self.assertFalse(incomplete.ready)
            self.assertEqual(incomplete.actual_bytes, 3)
            self.assertEqual(
                incomplete.missing_files,
                ("model-00002.safetensors",),
            )

            (root / "model-00002.safetensors").write_bytes(b"45header")
            complete = inspect_checkpoint(root)
            self.assertTrue(complete.ready)
            self.assertEqual(complete.actual_bytes, 11)

            (root / "model-00002.safetensors.aria2").write_bytes(b"pending")
            downloading = inspect_checkpoint(root)
            self.assertFalse(downloading.ready)
            self.assertIn(
                "model-00002.safetensors.aria2",
                downloading.missing_files,
            )


if __name__ == "__main__":
    unittest.main()
