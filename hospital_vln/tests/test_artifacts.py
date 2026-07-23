import tempfile
import unittest
from pathlib import Path

from hospital_vln.artifacts import HOSPITAL_START, build_bootstrap_artifacts
from hospital_vln.formal_places import build_formal_place_catalog
from hospital_vln.intent import HospitalIntentResolver
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

    def test_formal_catalog_exposes_function_metadata_to_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            width, height = 180, 120
            (root / "map.pgm").write_bytes(
                f"P5\n{width} {height}\n255\n".encode("ascii")
                + bytes([254]) * (width * height)
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\n"
                "resolution: 0.1\n"
                "origin: [-10.0, -5.0, 0.0]\n"
                "negate: 0\n"
                "occupied_thresh: 0.65\n"
                "free_thresh: 0.196\n",
                encoding="utf-8",
            )
            places_path = root / "places.json"
            build_formal_place_catalog(map_yaml, places_path)
            resolver = HospitalIntentResolver(places_path, provider="rule")

            prompt_catalog = resolver.places.catalog_for_prompt()
            waiting = next(item for item in prompt_catalog if item["id"] == "waiting_area")

            self.assertIn("等待医生", waiting["metadata"]["functions"])
            self.assertIn("坐下休息", waiting["metadata"]["description"])


if __name__ == "__main__":
    unittest.main()
