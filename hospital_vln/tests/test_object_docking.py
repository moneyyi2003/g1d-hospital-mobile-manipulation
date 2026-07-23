from __future__ import annotations

import math
from pathlib import Path
import unittest

from hospital_vln.object_docking import (
    ObjectTarget,
    build_object_docking_plan,
    compute_docking_pose,
    load_object_targets,
    parse_standoff,
    resolve_object,
)


ROOT = Path(__file__).resolve().parents[2]


class ObjectDockingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = ObjectTarget(
            "cube",
            "方块",
            ("小方块", "cube"),
            -2.5,
            0.2,
            0.86,
            -math.pi / 2.0,
            0.1,
        )

    def test_parameterized_pose_faces_object_at_exact_distance(self) -> None:
        pose = compute_docking_pose(self.target, 0.8)
        self.assertAlmostEqual(pose.x, -2.5)
        self.assertAlmostEqual(pose.y, -0.6)
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0)
        self.assertAlmostEqual(
            math.dist((pose.x, pose.y), (self.target.x, self.target.y)),
            0.8,
        )

    def test_command_resolves_object_and_distance(self) -> None:
        targets = load_object_targets(ROOT / "hospital_vln/object_targets_demo.json")
        target = resolve_object("请停到红色方块前0.6米", targets)
        self.assertEqual(target.object_id, "red_cube_demo")
        self.assertEqual(parse_standoff("请停到红色方块前0.6米"), 0.6)
        self.assertEqual(parse_standoff("go to the cube"), 0.8)

    def test_rejects_unsafe_standoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe"):
            compute_docking_pose(self.target, 0.45)

    def test_formal_map_plan_is_reachable(self) -> None:
        target = load_object_targets(
            ROOT / "hospital_vln/object_targets_demo.json"
        )[0]
        plan = build_object_docking_plan(
            ROOT / "outputs/hospital_vln/lingbot_map/map.yaml",
            target,
            0.8,
        )
        self.assertGreater(plan.path_length_m, 0.0)
        self.assertAlmostEqual(plan.object_distance_m, 0.8)
        self.assertAlmostEqual(plan.facing_error_rad, 0.0)


if __name__ == "__main__":
    unittest.main()
