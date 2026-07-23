from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

from hospital_vln.intent import HospitalPlaceResolution
from scripts.serve_object_docking_dashboard import ObjectDockingSession


class ObjectDockingDashboardTest(unittest.TestCase):
    class FakeIntentResolver:
        name = "deepseek"

        def resolve(self, command: str) -> HospitalPlaceResolution:
            return HospitalPlaceResolution(
                "waiting_area",
                "候诊区",
                "deepseek",
                0.96,
                "navigate",
            )

    def _session(self, root: Path) -> ObjectDockingSession:
        map_dir = root / "map"
        preview = root / "map_preview"
        map_dir.mkdir()
        preview.mkdir()
        (map_dir / "map.pgm").write_bytes(
            b"P5\n80 80\n255\n" + bytes([254]) * (80 * 80)
        )
        (map_dir / "map.yaml").write_text(
            "image: map.pgm\n"
            "resolution: 0.1\n"
            "origin: [-4.0, -4.0, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n",
            encoding="utf-8",
        )
        (preview / "rgb_pointcloud.png").write_bytes(b"rgb")
        (preview / "occupancy.png").write_bytes(b"occupancy")
        mapping = {
            "map": {
                "width": 80,
                "height": 80,
                "resolution": 0.1,
                "flip_y": True,
                "bounds": {
                    "min_x": -4.0,
                    "max_x": 4.0,
                    "min_z": -4.0,
                    "max_z": 4.0,
                },
                "layers": [
                    {"id": "rgb_pointcloud", "asset": "unused"},
                    {"id": "occupancy", "asset": "unused"},
                ],
            },
            "assets": {
                "rgb_pointcloud": str(preview / "rgb_pointcloud.png"),
                "occupancy": str(preview / "occupancy.png"),
            },
        }
        mapping_path = root / "mapping_summary.json"
        mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
        places_path = root / "places.json"
        places_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "map": {
                        "id": "test-map",
                        "sha256": "0" * 64,
                        "frame_id": "map",
                    },
                    "places": [
                        {
                            "id": "waiting_area",
                            "name": "候诊区",
                            "aliases": ["候诊区", "椅子"],
                            "status": "approved",
                            "entrance_pose": {
                                "x": -1.0,
                                "y": 0.0,
                                "yaw": -1.57,
                            },
                            "metadata": {
                                "typical_requests": ["我累了，带我去坐下"]
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        objects = {
            "activation": "isolated_demo_only",
            "objects": [
                {
                    "id": "red_cube",
                    "name": "红色方块",
                    "aliases": ["方块", "cube"],
                    "position": {"x": -2.5, "y": 0.2, "z": 1.05},
                    "interaction_face_yaw": -math.pi / 2.0,
                    "size_m": 0.2,
                }
            ],
        }
        objects_path = root / "objects.json"
        objects_path.write_text(
            json.dumps(objects, ensure_ascii=False),
            encoding="utf-8",
        )
        scene_config = {
            "activation": "explicit_object_docking_dashboard",
            "default_scene_id": "test_scene",
            "scenes": [
                {
                    "id": "test_scene",
                    "name": "Test Scene",
                    "status": "enabled",
                    "runner": "hospital_object_docking",
                    "map": str(map_dir / "map.yaml"),
                    "places": str(places_path),
                    "mapping_summary": str(mapping_path),
                    "objects": str(objects_path),
                    "output": str(root / "web-output"),
                }
            ],
        }
        scene_path = root / "scenes.json"
        scene_path.write_text(
            json.dumps(scene_config, ensure_ascii=False),
            encoding="utf-8",
        )
        return ObjectDockingSession(
            argparse.Namespace(
                scenes=scene_path,
                live_fps=10,
                live_resolution="960x540",
                intent_resolvers={"test_scene": self.FakeIntentResolver()},
            )
        )

    def test_config_exposes_scene_objects_and_relocatable_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(Path(directory))
            config = session.config()

            self.assertEqual(config["default_scene_id"], "test_scene")
            self.assertEqual(config["scenes"][0]["objects"][0]["id"], "red_cube")
            self.assertEqual(
                config["scenes"][0]["map"]["layers"][0]["asset"],
                "/asset/map/test_scene/rgb_pointcloud.png",
            )
            self.assertEqual(config["scenes"][0]["intent_parser"], "deepseek")
            self.assertEqual(
                config["scenes"][0]["places"][0]["examples"],
                ["我累了，带我去坐下"],
            )

    def test_command_changes_distance_and_rejects_unknown_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(Path(directory))
            profile, plan = session.plan("请停到红色方块前0.6米", "test_scene")

            self.assertEqual(profile.scene_id, "test_scene")
            self.assertAlmostEqual(plan.requested_standoff_m, 0.6)
            self.assertAlmostEqual(plan.docking_pose.x, -2.5)
            self.assertAlmostEqual(plan.docking_pose.y, -0.4)
            with self.assertRaisesRegex(ValueError, "未配置或未启用"):
                session.plan("请停到红色方块前0.8米", "missing")

    def test_unified_router_sends_fuzzy_region_and_object_commands_to_different_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = self._session(Path(directory))

            _, region = session.resolve_mission("我累了，带我去坐下", "test_scene")
            _, object_dock = session.resolve_mission(
                "请停到红色方块前0.8米",
                "test_scene",
            )

            self.assertEqual(region["mode"], "semantic_region_navigation")
            self.assertEqual(region["task_id"], "waiting_area")
            self.assertEqual(region["intent_resolution"]["parser"], "deepseek")
            self.assertEqual(object_dock["mode"], "object_relative_docking")
            self.assertEqual(object_dock["task_id"], "red_cube")
            self.assertIsNone(object_dock["intent_resolution"])


if __name__ == "__main__":
    unittest.main()
