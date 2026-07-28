from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from simple_room_vln.core import PathFollower, Pose2D, path_length, resolve_place
from warehouse_vln.artifacts import (
    CollisionBounds,
    build_bootstrap_artifacts,
    build_collision_grid,
    build_survey_path,
    snap_pose_to_free,
)
from warehouse_vln.kinematics import (
    navigation_twist_to_wheel_speeds,
    navigation_yaw_to_root_yaw,
    root_yaw_to_navigation_yaw,
)
from warehouse_vln.physics import (
    PhysicsLimits,
    PhysicsTelemetry,
    evaluate_physics_acceptance,
)


def measured_warehouse_obstacles() -> list[CollisionBounds]:
    return [
        CollisionBounds(
            "/Shelf_0",
            (-10.00252, -4.489313, -0.000069),
            (-8.606841, 13.129315, 6.000001),
        ),
        CollisionBounds(
            "/Shelf_1",
            (-1.002521, -4.489313, -0.000069),
            (0.393158, 13.129315, 6.000001),
        ),
        CollisionBounds(
            "/Shelf_2",
            (7.997479, -4.489313, -0.000069),
            (11.413317, 13.129315, 6.000001),
        ),
        CollisionBounds(
            "/PalletBin_01",
            (-7.062774, 12.901321, 0.05),
            (-5.194026, 14.362494, 1.17047),
        ),
        CollisionBounds(
            "/PalletBin_02",
            (3.515124, 12.901321, 0.05),
            (5.383872, 14.362494, 1.17047),
        ),
        CollisionBounds(
            "/Ceiling",
            (-12.0, -18.0, 8.5),
            (12.0, 20.5, 9.3),
        ),
    ]


class WarehouseArtifactsTest(unittest.TestCase):
    def test_warehouse_waypoint_tolerance_avoids_near_point_deadlock(self) -> None:
        path = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]
        strict = PathFollower(
            path,
            goal_yaw=0.0,
            waypoint_tolerance=0.18,
        )
        warehouse = PathFollower(
            path,
            goal_yaw=0.0,
            waypoint_tolerance=0.30,
        )
        pose = Pose2D(0.775, 0.0, 0.0)

        strict.command(pose)
        warehouse.command(pose)

        self.assertEqual(strict.index, 1)
        self.assertEqual(warehouse.index, 2)

    def test_terminal_alignment_latches_and_overcomes_static_friction(self) -> None:
        follower = PathFollower(
            [(0.0, 0.0), (1.0, 0.0)],
            goal_yaw=math.pi / 2.0,
            position_tolerance=0.20,
            yaw_tolerance=0.20,
            min_align_angular=1.05,
        )

        _, angular, state = follower.command(Pose2D(0.82, 0.0, 1.14))
        _, drifted_angular, drifted_state = follower.command(
            Pose2D(0.79, 0.0, 1.14)
        )

        self.assertEqual(state, "align")
        self.assertAlmostEqual(angular, 1.05)
        self.assertEqual(drifted_state, "align")
        self.assertAlmostEqual(drifted_angular, 1.05)

    def test_collision_grid_routes_around_the_middle_shelf(self) -> None:
        grid, used = build_collision_grid(measured_warehouse_obstacles())
        start = (-5.0, -10.0)
        goal = (4.0, 9.0)

        path = grid.plan(start, goal)

        self.assertEqual(len(used), 5)
        self.assertGreaterEqual(len(path), 3)
        self.assertGreater(path_length(path), math.dist(start, goal))
        self.assertTrue(any(y < -4.9 for _, y in path[1:-1]))

    def test_snap_pose_is_bounded_and_deterministic(self) -> None:
        grid, _ = build_collision_grid(measured_warehouse_obstacles())

        snapped = snap_pose_to_free(grid, Pose2D(-0.5, 0.0, 1.0))

        self.assertLessEqual(math.dist((-0.5, 0.0), (snapped.x, snapped.y)), 1.0)
        self.assertEqual(snapped.yaw, 1.0)
        self.assertTrue(grid.is_free(grid.world_to_cell(snapped.x, snapped.y)))

    def test_bootstrap_artifacts_are_explicit_and_resolvable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            grid, places, start, used = build_bootstrap_artifacts(
                output,
                measured_warehouse_obstacles(),
            )

            target = resolve_place("请带我到东侧货架通道", places)
            route = grid.plan((start.x, start.y), (target.pose.x, target.pose.y))
            payload = json.loads(
                (output / "bootstrap_occupancy.json").read_text(encoding="utf-8")
            )

        self.assertEqual(target.place_id, "east_shelf_aisle")
        self.assertGreater(path_length(route), 20.0)
        self.assertEqual(payload["source"], "isaac_collision_aabb_bootstrap")
        self.assertEqual(payload["source_collision_count"], 6)
        self.assertEqual(payload["used_collision_count"], len(used))

    def test_invalid_collision_bounds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "inverted"):
            CollisionBounds("/bad", (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))

    def test_survey_visits_both_aisles_without_redundant_return(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            grid, places, start, _ = build_bootstrap_artifacts(
                Path(temporary),
                measured_warehouse_obstacles(),
            )

            route = build_survey_path(grid, start, places)

        self.assertEqual(route[0], (start.x, start.y))
        self.assertNotEqual(route[-1], (start.x, start.y))
        self.assertTrue(any(x < -4.5 and y > 8.5 for x, y in route))
        self.assertTrue(any(x > 3.5 and y > 8.5 for x, y in route))
        self.assertLess(path_length(route), 60.0)

    def test_g1d_usd_wheel_and_heading_conventions_are_explicit(self) -> None:
        left, right = navigation_twist_to_wheel_speeds(0.5, 0.0)
        turn_left, turn_right = navigation_twist_to_wheel_speeds(0.0, 1.0)

        self.assertGreater(left, 0.0)
        self.assertLess(right, 0.0)
        self.assertGreater(turn_left, 0.0)
        self.assertGreater(turn_right, 0.0)
        root_yaw = navigation_yaw_to_root_yaw(0.4)
        self.assertAlmostEqual(root_yaw_to_navigation_yaw(root_yaw), 0.4)

    def test_physics_acceptance_includes_braking_and_tilt(self) -> None:
        accepted, failures = evaluate_physics_acceptance(
            navigation_done=True,
            position_error_m=0.19,
            yaw_error_rad=0.15,
            position_tolerance_m=0.20,
            yaw_tolerance_rad=0.20,
            max_abs_roll_rad=0.03,
            max_abs_pitch_rad=0.04,
            brake_drift_m=0.01,
            stopped_linear_mps=0.005,
            stopped_angular_radps=0.01,
            limits=PhysicsLimits(),
        )
        rejected, rejected_failures = evaluate_physics_acceptance(
            navigation_done=True,
            position_error_m=0.19,
            yaw_error_rad=0.15,
            position_tolerance_m=0.20,
            yaw_tolerance_rad=0.20,
            max_abs_roll_rad=0.03,
            max_abs_pitch_rad=0.04,
            brake_drift_m=0.08,
            stopped_linear_mps=0.005,
            stopped_angular_radps=0.01,
            limits=PhysicsLimits(),
        )

        self.assertTrue(accepted)
        self.assertEqual(failures, [])
        self.assertFalse(rejected)
        self.assertIn("brake_drift_exceeded", rejected_failures)

    def test_physics_telemetry_records_real_travel(self) -> None:
        telemetry = PhysicsTelemetry(sample_period_frames=2)
        telemetry.observe(
            frame=0,
            pose=Pose2D(0.0, 0.0, 0.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            commanded_twist=(0.5, 0.0),
            wheel_targets_radps=(5.0, -5.0),
            wheel_actual_radps=(4.8, -4.7),
            linear_velocity_mps=(0.48, 0.0, 0.0),
            angular_velocity_radps=(0.0, 0.0, 0.0),
        )
        telemetry.observe(
            frame=1,
            pose=Pose2D(0.3, 0.4, 0.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            commanded_twist=(0.0, 0.0),
            wheel_targets_radps=(0.0, 0.0),
            wheel_actual_radps=(0.0, 0.0),
            linear_velocity_mps=(0.0, 0.0, 0.0),
            angular_velocity_radps=(0.0, 0.0, 0.0),
            force_sample=True,
        )

        self.assertAlmostEqual(telemetry.distance_m, 0.5)
        self.assertEqual(len(telemetry.samples), 2)


if __name__ == "__main__":
    unittest.main()
