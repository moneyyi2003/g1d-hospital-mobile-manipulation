"""A deterministic language-to-point navigation baseline for the custom G1_D."""

import argparse
import math
import os
import sys
import threading
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="Absolute path to the converted G1_D USD.")
parser.add_argument("--instruction", default="前往红色目标", help="Chinese or English color-goal instruction.")
parser.add_argument("--suite", action="store_true", help="Run red, blue, and green instructions in sequence.")
parser.add_argument("--max-steps", type=int, default=1800, help="Maximum physics steps per episode.")
parser.add_argument("--success-distance", type=float, default=0.12, help="Goal success radius in metres.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


GOALS = {
    "red": ((0.55, 0.0), (0.85, 0.1, 0.1), "前往红色目标"),
    "blue": ((0.0, 0.55), (0.1, 0.2, 0.9), "go to the blue target"),
    "green": ((0.0, -0.55), (0.1, 0.8, 0.2), "前往绿色目标"),
}


def parse_instruction(text: str) -> str:
    lowered = text.lower()
    aliases = {"red": ("red", "红"), "blue": ("blue", "蓝"), "green": ("green", "绿")}
    matches = [name for name, words in aliases.items() if any(word in lowered for word in words)]
    if len(matches) != 1:
        raise ValueError(f"Instruction must contain exactly one supported color (red/blue/green): {text!r}")
    return matches[0]


def make_robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
                effort_limit_sim=400.0,
                velocity_limit_sim=20.0,
                stiffness=0.0,
                damping=150.0,
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


def quaternion_yaw(quaternion: torch.Tensor) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def set_targets(
    robot: Articulation,
    position_targets: torch.Tensor,
    wheel_ids: list[int],
    left: float,
    right: float,
) -> None:
    robot.set_joint_position_target(position_targets)
    wheel_targets = torch.tensor([[left, right]], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_ids)
    robot.write_data_to_sim()


@torch.inference_mode()
def reset_robot(sim: SimulationContext, robot: Articulation, wheel_ids: list[int]) -> None:
    root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    positions = robot.data.default_joint_pos.clone()
    with torch.inference_mode():
        for _ in range(120):
            set_targets(robot, positions, wheel_ids, 0.0, 0.0)
            sim.step()
            robot.update(sim.get_physics_dt())


@torch.inference_mode()
def run_episode(
    sim: SimulationContext,
    robot: Articulation,
    wheel_ids: list[int],
    goal_name: str,
    instruction: str,
) -> dict[str, float | int | str | bool]:
    reset_robot(sim, robot, wheel_ids)
    offset = GOALS[goal_name][0]
    start = robot.data.root_pos_w[0, :2].clone()
    goal = start + torch.tensor(offset, device=robot.device)
    positions = robot.data.default_joint_pos.clone()
    path_length = 0.0
    previous = start.clone()
    success = False
    final_distance = float("inf")

    # The imported wheel radius is about 8.5 cm and the joint centers are 40.62 cm apart.
    wheel_radius = 0.0848
    axle_track = 0.4062
    print(f"[VLN] instruction={instruction!r} parsed_goal={goal_name} goal_xy={goal.tolist()}")

    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            position = robot.data.root_pos_w[0, :2]
            delta = goal - position
            distance = float(torch.linalg.vector_norm(delta))
            final_distance = distance
            path_length += float(torch.linalg.vector_norm(position - previous))
            previous = position.clone()

            if distance < args_cli.success_distance:
                success = True
                set_targets(robot, positions, wheel_ids, 0.0, 0.0)
                for _ in range(30):
                    sim.step()
                    robot.update(sim.get_physics_dt())
                break

            desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
            yaw_error = wrap_angle(desired_yaw - quaternion_yaw(robot.data.root_quat_w[0]))
            angular_speed = max(-0.9, min(0.9, 2.2 * yaw_error))
            linear_speed = 0.18 * max(0.0, 1.0 - abs(yaw_error) / 0.8)
            linear_speed = min(linear_speed, 0.6 * distance)

            left_physical = linear_speed - angular_speed * axle_track / 2.0
            right_physical = linear_speed + angular_speed * axle_track / 2.0
            # Left and right joint axes are -Y and +Y respectively in the URDF.
            # The imported collision geometry has considerable rolling resistance.  A calibrated
            # command gain turns desired chassis velocity into an effective joint target.
            command_gain = 3.0
            left_joint = max(-12.0, min(12.0, -command_gain * left_physical / wheel_radius))
            right_joint = max(-12.0, min(12.0, command_gain * right_physical / wheel_radius))
            # Below roughly 5 rad/s the high-fidelity convex wheel/base contacts stick.
            # Preserve direction while supplying enough target speed to break static resistance.
            if 0.05 < abs(left_joint) < 5.0:
                left_joint = math.copysign(5.0, left_joint)
            if 0.05 < abs(right_joint) < 5.0:
                right_joint = math.copysign(5.0, right_joint)
            set_targets(robot, positions, wheel_ids, left_joint, right_joint)
            sim.step()
            robot.update(sim.get_physics_dt())

            if step % 240 == 0:
                print(
                    f"[VLN] step={step} distance={distance:.3f} yaw_error={yaw_error:.3f} "
                    f"wheel=({left_joint:.2f},{right_joint:.2f})"
                )

    status = "SUCCESS" if success else "FAIL"
    print(
        f"[VLN][{status}] goal={goal_name} steps={step + 1} "
        f"final_distance={final_distance:.3f}m path_length={path_length:.3f}m"
    )
    return {
        "goal": goal_name,
        "success": success,
        "steps": step + 1,
        "final_distance": final_distance,
        "path_length": path_length,
    }


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0)
    sim_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.2,
        dynamic_friction=1.0,
        friction_combine_mode="max",
    )
    sim = SimulationContext(sim_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=sim_cfg.physics_material)
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    for name, (xy, color, _) in GOALS.items():
        marker = sim_utils.CylinderCfg(
            radius=0.12,
            height=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        marker.func(f"/World/Goals/{name}", marker, translation=(xy[0], xy[1], 0.008))

    robot = Articulation(make_robot_cfg())
    sim.reset()
    wheel_ids, _ = robot.find_joints(["Left_Wheel_Joint", "Right_Wheel_Joint"])
    if len(wheel_ids) != 2:
        raise RuntimeError("G1_D wheel joint mapping failed.")

    if args_cli.suite:
        requests = [(name, data[2]) for name, data in GOALS.items()]
    else:
        name = parse_instruction(args_cli.instruction)
        requests = [(name, args_cli.instruction)]

    results = [run_episode(sim, robot, wheel_ids, name, instruction) for name, instruction in requests]
    successes = sum(bool(item["success"]) for item in results)
    print(f"[VLN][SUMMARY] success_rate={successes}/{len(results)}")
    if successes != len(results):
        raise RuntimeError("One or more VLN episodes did not reach the success radius.")


def close_app(exit_code: int) -> None:
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
    except BaseException:
        traceback.print_exc()
        code = 1
    close_app(code)
    raise SystemExit(code)
