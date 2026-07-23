import tempfile
import unittest
from pathlib import Path

from hospital_vln.artifacts import HOSPITAL_START, build_bootstrap_artifacts
from simple_room_vln.core import path_length, resolve_place


class HospitalArtifactsTest(unittest.TestCase):
    def test_waiting_area_command_resolves_and_has_a_route(self):
        with tempfile.TemporaryDirectory() as directory:
            grid, places = build_bootstrap_artifacts(Path(directory))

        target = resolve_place("请带我到候诊区", places)
        route = grid.plan(
            (HOSPITAL_START.x, HOSPITAL_START.y),
            (target.pose.x, target.pose.y),
        )

        self.assertEqual(target.place_id, "waiting_area")
        self.assertGreater(path_length(route), 1.0)

    def test_pending_corridor_is_not_language_selectable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, places = build_bootstrap_artifacts(Path(directory))

        with self.assertRaisesRegex(ValueError, "没有匹配已审核地点"):
            resolve_place("请带我到主走廊", places)


if __name__ == "__main__":
    unittest.main()
