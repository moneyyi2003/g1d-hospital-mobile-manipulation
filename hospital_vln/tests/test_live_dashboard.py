import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hospital_vln.intent import HospitalPlaceResolution
from hospital_vln.live import LivePublisher, publish_failure
from scripts.serve_hospital_dashboard import HospitalDashboardSession
from simple_room_vln.core import Pose2D


class LivePublisherTest(unittest.TestCase):
    def test_state_is_atomic_and_trajectory_is_distance_sampled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = LivePublisher(
                root,
                command="请带我到候诊区",
                task="waiting_area",
                map_source="formal",
                path=[(0.0, -1.5), (1.0, 0.0)],
            )
            for frame, x in enumerate((0.0, 0.01, 0.04)):
                publisher.publish_state(
                    state="running",
                    message="running",
                    frame=frame,
                    action="follow",
                    pose=Pose2D(x, -1.5, 0.0),
                    linear=0.2,
                    angular=0.0,
                    waypoint=1,
                    waypoint_count=1,
                )

            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["sequence"], 2)
            self.assertEqual(state["task"], "waiting_area")
            self.assertEqual(len(state["planned_trajectory"]), 2)
            self.assertEqual(len(state["trajectory"]), 2)
            self.assertFalse((root / ".state.json.tmp").exists())

    def test_failure_state_has_dashboard_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish_failure(
                root,
                command="go to reception",
                message="RuntimeError: failed",
                pose=Pose2D(0.0, -1.5, 1.57),
            )

            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["linear_velocity_mps"], 0.0)
            self.assertEqual(state["trajectory"], [])


class HospitalDashboardSessionTest(unittest.TestCase):
    class FakeIntentResolver:
        name = "deepseek"

        def resolve(self, command):
            if "坐着等医生" in command:
                return HospitalPlaceResolution(
                    "waiting_area",
                    "候诊区",
                    "deepseek",
                    0.93,
                    "navigate",
                )
            return HospitalPlaceResolution(
                "reception",
                "医院前台",
                "deepseek",
                0.91,
                "navigate",
            )

    def test_config_and_plan_use_relocatable_map_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            preview = artifacts / "map_preview"
            map_dir = artifacts / "lingbot_map"
            preview.mkdir(parents=True)
            map_dir.mkdir(parents=True)
            (preview / "rgb_pointcloud.png").write_bytes(b"png-pointcloud")
            (preview / "occupancy.png").write_bytes(b"png-occupancy")
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
            places = {
                "schema_version": 2,
                "places": [
                    {
                        "id": "reception",
                        "name": "医院前台",
                        "aliases": ["前台", "reception"],
                        "status": "approved",
                        "entrance_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0},
                        "metadata": {
                            "typical_requests": ["我想找工作人员问点事情"]
                        },
                    },
                    {
                        "id": "waiting_area",
                        "name": "候诊区",
                        "aliases": ["等候区", "waiting area"],
                        "status": "approved",
                        "entrance_pose": {"x": -1.0, "y": 0.0, "yaw": 0.0},
                        "metadata": {
                            "typical_requests": ["找个能坐着等医生的地方"]
                        },
                    }
                ],
                "map": {
                    "id": "test-map",
                    "sha256": "0" * 64,
                    "frame_id": "map",
                },
            }
            (artifacts / "places_formal.json").write_text(
                json.dumps(places, ensure_ascii=False), encoding="utf-8"
            )
            mapping = {
                "schema_version": 1,
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
                        {"id": "rgb_pointcloud", "asset": "/old/rgb_pointcloud.png"},
                        {"id": "occupancy", "asset": "/old/occupancy.png"},
                    ],
                },
                "assets": {
                    "rgb_pointcloud": "/old/rgb_pointcloud.png",
                    "occupancy": "/old/occupancy.png",
                },
            }
            (artifacts / "mapping_summary.json").write_text(
                json.dumps(mapping), encoding="utf-8"
            )
            live = root / "web-output/live"
            live.mkdir(parents=True)
            (live / "state.json").write_text(
                json.dumps(
                    {
                        "state": "succeeded",
                        "command": "找个能坐着等医生的地方",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "web-output/intent_resolution.json").write_text(
                json.dumps(
                    {
                        "command": "找个能坐着等医生的地方",
                        "intent_resolution": {
                            "place_id": "waiting_area",
                            "place_name": "候诊区",
                            "parser": "deepseek",
                            "confidence": 0.93,
                            "intent": "navigate",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            session = HospitalDashboardSession(
                argparse.Namespace(
                    artifacts=artifacts,
                    output=root / "web-output",
                    intent_resolver=self.FakeIntentResolver(),
                )
            )

            config = session.config()
            target, path, resolution = session.plan("找个能坐着等医生的地方")

            self.assertEqual(target.place_id, "waiting_area")
            self.assertEqual(resolution["parser"], "deepseek")
            self.assertEqual(resolution["confidence"], 0.93)
            self.assertEqual(resolution["docking"]["mode"], "formal_fixed_pose")
            self.assertEqual(target.pose.x, -1.0)
            self.assertGreaterEqual(len(path), 2)
            self.assertEqual(config["intent_parser"], "deepseek")
            self.assertEqual(
                session.snapshot()["intent_resolution"]["place_id"],
                "waiting_area",
            )
            self.assertEqual(
                config["places"][1]["examples"],
                ["找个能坐着等医生的地方"],
            )
            self.assertEqual(
                config["map"]["layers"][0]["asset"],
                "/asset/map/rgb_pointcloud.png",
            )
            self.assertEqual(
                session.map_asset("occupancy"),
                (preview / "occupancy.png").resolve(),
            )

            docking_path = root / "dynamic_candidates.json"
            docking_path.write_text(
                json.dumps(
                    {
                        "activation": "explicit_opt_in_only",
                        "map": {"sha256": "0" * 64},
                        "candidates": [
                            {
                                "candidate_id": "chair_right_south",
                                "chair_instance_id": "chair_right",
                                "eligible": True,
                                "score": 0.8,
                                "clearance_m": 0.7,
                                "path_length_m": 2.0,
                                "pose": {"x": -1.5, "y": 0.5, "yaw": 1.57},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dynamic_session = HospitalDashboardSession(
                argparse.Namespace(
                    artifacts=artifacts,
                    output=root / "dynamic-web-output",
                    intent_resolver=self.FakeIntentResolver(),
                    dynamic_docking=True,
                    docking_candidates=docking_path,
                    blocked_candidate=[],
                )
            )
            dynamic_target, _, dynamic_resolution = dynamic_session.plan(
                "找个能坐着等医生的地方"
            )

            self.assertEqual(dynamic_target.pose.x, -1.5)
            self.assertEqual(
                dynamic_resolution["docking"]["candidate_id"],
                "chair_right_south",
            )
            self.assertEqual(
                dynamic_session.config()["docking_mode"],
                "experimental_dynamic_candidate",
            )


if __name__ == "__main__":
    unittest.main()
