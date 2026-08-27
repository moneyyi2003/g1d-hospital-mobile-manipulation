"""Headless articulation smoke test for the custom G1_D USD asset."""

import argparse
import os
import sys
import threading
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="Absolute path to the converted G1_D USD.")
parser.add_argument("--steps", type=int, default=20, help="Number of physics steps to run.")
parser.add_argument("--wheel-speed", type=float, default=1.0, help="Wheel target speed in rad/s.")
parser.add_argument(
    "--with-ground",
    action="store_true",
    help="Enable gravity and collisions with a ground plane (slower; default only validates articulation/control).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


EXPECTED_JOINTS = {
    "Left_Wheel_Joint",
    "Right_Wheel_Joint",
    "LZ_mt_Joint",
    "LZ_it_Joint",
    "right_shoulder_pitch_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
}


def make_robot_cfg() -> ArticulationCfg:
    """Return a non-overlapping actuator configuration for every movable joint."""
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=not args_cli.with_ground,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["Left_Wheel_Joint", "Right_Wheel_Joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=20.0,
                stiffness=0.0,
                damping=40.0,
            ),
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["LZ_mt_Joint", "LZ_it_Joint"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=500.0,
                damping=50.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["Yaw_Joint", "torso_Joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=3.0,
                stiffness=200.0,
                damping=20.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_(shoulder|elbow|wrist)_.*joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=4.0,
                stiffness=120.0,
                damping=12.0,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_(thumb|middle|index)_.*joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=6.0,
                stiffness=30.0,
                damping=3.0,
            ),
        },
    )


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0))
    if args_cli.with_ground:
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(make_robot_cfg())
    sim.reset()

    print(f"[INFO] Bodies ({robot.num_bodies}): {robot.body_names}")
    print(f"[INFO] Joints ({robot.num_joints}): {robot.joint_names}")
    missing = sorted(EXPECTED_JOINTS.difference(robot.joint_names))
    if missing:
        raise RuntimeError(f"Converted articulation is missing expected joints: {missing}")

    wheel_ids, wheel_names = robot.find_joints(["Left_Wheel_Joint", "Right_Wheel_Joint"])
    if len(wheel_ids) != 2:
        raise RuntimeError(f"Expected two wheel joints, got {wheel_names}")
    exercise_ids, exercise_names = robot.find_joints(
        ["right_shoulder_pitch_joint", "right_elbow_joint", "right_hand_index_0_joint"]
    )
    if len(exercise_ids) != 3:
        raise RuntimeError(f"Expected right arm/hand exercise joints, got {exercise_names}")
    hand_body_ids, hand_body_names = robot.find_bodies(
        ["right_hand_palm_link", "right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]
    )

    default_root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    start_root = robot.data.root_pos_w.clone()
    position_targets = robot.data.default_joint_pos.clone()
    position_targets[:, exercise_ids] += torch.tensor([0.2, 0.3, 0.4], device=robot.device)
    # The two URDF joint axes point in opposite Y directions, so forward motion uses opposite joint signs.
    wheel_targets = torch.tensor(
        [[-args_cli.wheel_speed, args_cli.wheel_speed]], device=robot.device
    ).repeat(robot.num_instances, 1)

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            robot.set_joint_position_target(position_targets)
            robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

    tensors = (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("G1_D produced NaN/Inf state during the smoke test.")

    root_delta = robot.data.root_pos_w - start_root
    wheel_velocity = robot.data.joint_vel[:, wheel_ids]
    exercise_delta = robot.data.joint_pos[:, exercise_ids] - robot.data.default_joint_pos[:, exercise_ids]
    print(f"[INFO] Root delta (m): {root_delta[0].tolist()}")
    print(f"[INFO] Wheel velocity (rad/s): {wheel_velocity[0].tolist()}")
    print(f"[INFO] Right arm/hand joint delta (rad): {exercise_delta[0].tolist()}")
    hand_positions = robot.data.body_pos_w[0, hand_body_ids]
    print(f"[INFO] Right hand body names: {hand_body_names}")
    print(f"[INFO] Right hand body positions (m): {hand_positions.tolist()}")
    if args_cli.steps >= 20 and float(torch.max(torch.abs(exercise_delta))) < 0.01:
        raise RuntimeError("Right arm/hand position command produced no measurable joint response.")
    print(f"[OK] Loaded G1_D and completed {args_cli.steps} finite simulation steps.")


def close_app(exit_code: int) -> None:
    """Close Kit, with a guard for Isaac 4.5's occasional stage-loading shutdown stall."""
    sys.stdout.flush()
    sys.stderr.flush()
    watchdog = threading.Timer(10.0, lambda: os._exit(exit_code))
    watchdog.start()
    simulation_app.close(wait_for_replicator=False)
    watchdog.cancel()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # make sure a failed headless run also terminates Kit
        traceback.print_exc()
        code = 1
    close_app(code)
    raise SystemExit(code)
