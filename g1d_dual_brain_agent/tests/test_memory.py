from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from g1d_dual_brain_agent.memory import SharedWorldMemory


class SharedWorldMemoryTest(unittest.TestCase):
    def test_object_and_blackboard_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.json"
            memory = SharedWorldMemory(path)
            memory.begin_mission("mission-1")
            memory.update_object(
                "cup-1",
                {
                    "labels": ["cup", "红色杯子"],
                    "visible": True,
                    "detection_confidence": 0.91,
                },
            )
            memory.blackboard.carried_object_id = "cup-1"
            memory.save()

            loaded = SharedWorldMemory.load(path)

            self.assertEqual(loaded.blackboard.mission_id, "mission-1")
            self.assertEqual(loaded.blackboard.carried_object_id, "cup-1")
            self.assertEqual(loaded.get_object("cup-1").labels[0], "cup")
            self.assertEqual(loaded.get_object("cup-1").revision, 1)

    def test_unknown_object_field_is_rejected(self) -> None:
        memory = SharedWorldMemory()

        with self.assertRaisesRegex(ValueError, "unknown object memory fields"):
            memory.update_object("cup-1", {"invented_pose": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
