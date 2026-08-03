from __future__ import annotations

import unittest

from g1d_dual_brain_agent.models import GoalKind
from g1d_dual_brain_agent.planner import compile_family_home_command


PLACES = {
    "places": [
        {
            "id": "living_room_sofa",
            "name": "客厅沙发旁",
            "aliases": ["客厅", "沙发旁", "sofa"],
            "status": "approved",
        },
        {
            "id": "bedroom_bed",
            "name": "卧室床边",
            "aliases": ["卧室", "床边"],
            "status": "rejected",
        },
    ]
}
OBJECTS = {
    "objects": [
        {
            "object_id": "scan_cup_06",
            "source_label": "cup",
            "aliases": ["杯子", "水杯"],
            "status": "approved",
            "manipulation_ready": True,
        },
        {
            "object_id": "rejected_book",
            "source_label": "book",
            "aliases": ["书"],
            "status": "rejected",
        },
    ]
}


class FamilyLongTaskCompilerTest(unittest.TestCase):
    def test_compiles_go_pick_return_in_order(self) -> None:
        mission = compile_family_home_command(
            "请带我去客厅沙发旁，拿起水杯，再回到客厅",
            places_catalog=PLACES,
            objects_catalog=OBJECTS,
        )

        self.assertEqual(
            [goal.kind for goal in mission.goals],
            [GoalKind.NAVIGATE, GoalKind.INTERACT, GoalKind.NAVIGATE],
        )
        self.assertEqual(
            [goal.instruction for goal in mission.goals],
            ["living_room_sofa", "拿起 scan_cup_06", "living_room_sofa"],
        )
        self.assertEqual(mission.goals[1].target_id, "scan_cup_06")
        self.assertEqual(
            mission.goals[2].metadata["requires_carried_object_id"],
            "scan_cup_06",
        )

    def test_rejected_place_is_not_promoted(self) -> None:
        with self.assertRaisesRegex(ValueError, "审核地点"):
            compile_family_home_command(
                "去卧室，拿杯子，再回到客厅",
                places_catalog=PLACES,
                objects_catalog=OBJECTS,
            )

    def test_rejected_or_unknown_object_is_not_guessed(self) -> None:
        with self.assertRaisesRegex(ValueError, "审核对象"):
            compile_family_home_command(
                "去客厅，拿书，再回到客厅",
                places_catalog=PLACES,
                objects_catalog=OBJECTS,
            )

    def test_requires_return_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "回到/返回"):
            compile_family_home_command(
                "去客厅，拿杯子",
                places_catalog=PLACES,
                objects_catalog=OBJECTS,
            )


if __name__ == "__main__":
    unittest.main()
