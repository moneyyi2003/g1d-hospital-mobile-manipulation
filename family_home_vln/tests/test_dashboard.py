import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.serve_family_home_dashboard import FamilyHomeDashboardSession


class FamilyHomeDashboardSessionTest(unittest.TestCase):
    def make_session(self, root: Path) -> FamilyHomeDashboardSession:
        return FamilyHomeDashboardSession(
            argparse.Namespace(
                artifacts=root / "artifacts",
                output=root / "web-output",
            )
        )

    def test_config_exposes_all_requested_layers_and_truth_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory))
            config = session.config()
            data = session.map_data()

            self.assertEqual(
                {layer["id"] for layer in config["layers"]},
                {"pointcloud", "semantic", "occupancy", "region"},
            )
            self.assertEqual(config["map"]["source_status"], "bootstrap")
            self.assertIn("geometry proxy", data["truth_boundary"])
            self.assertGreater(len(data["pointcloud"]), 100)
            self.assertEqual(
                {region["region_id"] for region in data["regions"]},
                {"bedroom", "living_room", "dining_area", "kitchen", "transition"},
            )

    def test_plan_uses_only_reviewed_family_place_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory))

            target, path = session.plan("请带我到厨房操作台")

            self.assertEqual(target.place_id, "kitchen_counter")
            self.assertGreaterEqual(len(path), 2)
            with self.assertRaises(ValueError):
                session.plan("请带我去阳台")

    def test_idle_snapshot_is_dashboard_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory))

            state = session.snapshot()

            self.assertEqual(state["state"], "idle")
            self.assertFalse(state["process_running"])
            self.assertEqual(state["pose"]["x"], 0.0)

    def test_incomplete_formal_ui_bundle_never_drops_bootstrap_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            (artifacts / "lingbot_map").mkdir(parents=True)
            (artifacts / "lingbot_map/map.yaml").write_text("image: map.pgm\n")
            (artifacts / "places_formal.json").write_text("{}")
            (artifacts / "mapping_summary.json").write_text("{}")

            session = self.make_session(root)
            config = session.config()

            self.assertTrue(config["map"]["formal_bundle_detected"])
            self.assertEqual(config["map"]["source_status"], "bootstrap")


if __name__ == "__main__":
    unittest.main()
