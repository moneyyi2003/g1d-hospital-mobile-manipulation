"""Physical G1-D Nav2 bringup with a fail-closed vendor-driver boundary."""

from pathlib import Path
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


class G1DNav2Params(RewrittenYaml):
    """Rewrite stock Humble Nav2 parameters for G1-D and the formal map."""

    def perform(self, context):
        target = super().perform(context)
        with open(target, encoding="utf-8") as stream:
            params = yaml.safe_load(stream)

        params["amcl"]["ros__parameters"].update(
            {
                "base_frame_id": "AGV_link",
                "odom_frame_id": "odom",
                "global_frame_id": "map",
                "scan_topic": "/scan",
                "robot_model_type": "nav2_amcl::DifferentialMotionModel",
                "update_min_d": 0.10,
                "update_min_a": 0.10,
            }
        )
        params["bt_navigator"]["ros__parameters"].update(
            {
                "global_frame": "map",
                "robot_base_frame": "AGV_link",
                "odom_topic": "/odom",
            }
        )
        controller = params["controller_server"]["ros__parameters"]
        controller.update(
            {
                "controller_frequency": 20.0,
                "controller_plugins": ["FollowPath"],
            }
        )
        controller["progress_checker"].update(
            {"required_movement_radius": 0.10, "movement_time_allowance": 20.0}
        )
        controller["general_goal_checker"].update(
            {"xy_goal_tolerance": 0.20, "yaw_goal_tolerance": 0.20}
        )
        controller["FollowPath"] = {
            "plugin": (
                "nav2_regulated_pure_pursuit_controller::"
                "RegulatedPurePursuitController"
            ),
            "desired_linear_vel": 0.30,
            "lookahead_dist": 0.55,
            "min_lookahead_dist": 0.30,
            "max_lookahead_dist": 0.75,
            "lookahead_time": 1.5,
            "rotate_to_heading_angular_vel": 0.60,
            "transform_tolerance": 0.20,
            "use_velocity_scaled_lookahead_dist": True,
            "min_approach_linear_velocity": 0.04,
            "approach_velocity_scaling_dist": 0.80,
            "use_collision_detection": True,
            "max_allowed_time_to_collision_up_to_carrot": 1.0,
            "use_regulated_linear_velocity_scaling": True,
            "use_cost_regulated_linear_velocity_scaling": True,
            "regulated_linear_scaling_min_radius": 0.75,
            "regulated_linear_scaling_min_speed": 0.05,
            "cost_scaling_dist": 0.60,
            "cost_scaling_gain": 1.0,
            "inflation_cost_scaling_factor": 3.0,
            "use_rotate_to_heading": True,
            "allow_reversing": False,
            "rotate_to_heading_min_angle": 0.50,
            "max_angular_accel": 1.20,
        }

        for group, global_frame, rolling in (
            ("local_costmap", "odom", True),
            ("global_costmap", "map", False),
        ):
            costmap = params[group][group]["ros__parameters"]
            costmap.update(
                {
                    "global_frame": global_frame,
                    "robot_base_frame": "AGV_link",
                    "robot_radius": 0.42,
                    "resolution": 0.05,
                    "track_unknown_space": True,
                    "rolling_window": rolling,
                    "plugins": ["obstacle_layer", "inflation_layer"]
                    if rolling
                    else ["static_layer", "obstacle_layer", "inflation_layer"],
                }
            )
            costmap.pop("voxel_layer", None)
            costmap["obstacle_layer"] = {
                "plugin": "nav2_costmap_2d::ObstacleLayer",
                "enabled": True,
                "observation_sources": "scan",
                "scan": {
                    "topic": "/scan",
                    "max_obstacle_height": 1.8,
                    "clearing": True,
                    "marking": True,
                    "data_type": "LaserScan",
                    "raytrace_max_range": 8.0,
                    "raytrace_min_range": 0.15,
                    "obstacle_max_range": 6.0,
                    "obstacle_min_range": 0.15,
                },
            }
            costmap["inflation_layer"] = {
                "plugin": "nav2_costmap_2d::InflationLayer",
                "cost_scaling_factor": 3.0,
                "inflation_radius": 0.60,
            }
        params["local_costmap"]["local_costmap"]["ros__parameters"].update(
            {"width": 6, "height": 6}
        )
        params["planner_server"]["ros__parameters"]["GridBased"].update(
            {"tolerance": 0.20, "allow_unknown": False, "use_astar": True}
        )
        params["behavior_server"]["ros__parameters"].update(
            {
                "global_frame": "odom",
                "robot_base_frame": "AGV_link",
                "max_rotational_vel": 0.80,
                "min_rotational_vel": 0.10,
                "rotational_acc_lim": 1.20,
            }
        )
        params["velocity_smoother"]["ros__parameters"].update(
            {
                "feedback": "CLOSED_LOOP",
                "max_velocity": [0.35, 0.0, 0.80],
                "min_velocity": [-0.10, 0.0, -0.80],
                "max_accel": [0.50, 0.0, 1.20],
                "max_decel": [-0.80, 0.0, -1.60],
                "odom_topic": "/odom",
                "velocity_timeout": 0.25,
            }
        )
        with open(target, "w", encoding="utf-8") as stream:
            yaml.safe_dump(params, stream, sort_keys=False)
        return target


def _validated_nodes(context):
    robot_urdf = Path(LaunchConfiguration("robot_urdf").perform(context)).resolve()
    map_file = Path(LaunchConfiguration("map").perform(context)).resolve()
    places_file = Path(LaunchConfiguration("places").perform(context)).resolve()
    for label, path in (
        ("robot_urdf", robot_urdf),
        ("map", map_file),
        ("places", places_file),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} does not exist: {path}")
    description = robot_urdf.read_text(encoding="utf-8")
    if 'link name="AGV_link"' not in description:
        raise RuntimeError("G1-D URDF must use AGV_link as its root frame")
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="g1d_robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": description, "use_sim_time": False}],
        ),
    ]


def generate_launch_description():
    package_share = get_package_share_directory("lingbot_semantic_nav_ros")
    nav2_share = get_package_share_directory("nav2_bringup")
    map_file = LaunchConfiguration("map")
    places_file = LaunchConfiguration("places")
    allow_hardware = LaunchConfiguration("allow_hardware_output")
    base_params = LaunchConfiguration("base_params")
    nav2_params = G1DNav2Params(
        source_file=LaunchConfiguration("nav2_params"),
        root_key="",
        param_rewrites={"use_sim_time": "False"},
        convert_types=True,
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_share, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map": map_file,
            "use_sim_time": "False",
            "autostart": "True",
            "use_composition": "False",
            "params_file": nav2_params,
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", description="Formal LingBot occupancy YAML"),
            DeclareLaunchArgument("places", description="Reviewed schema-v2 places"),
            DeclareLaunchArgument(
                "robot_urdf",
                default_value="/root/autodl-tmp/Assets/g1_d_robot/source/g1_d.urdf",
            ),
            DeclareLaunchArgument(
                "base_params",
                default_value=os.path.join(package_share, "params", "g1d_base.yaml"),
            ),
            DeclareLaunchArgument(
                "nav2_params",
                default_value=os.path.join(nav2_share, "params", "nav2_params.yaml"),
            ),
            DeclareLaunchArgument("allow_hardware_output", default_value="False"),
            DeclareLaunchArgument(
                "lingbot_pythonpath",
                default_value="/root/autodl-tmp/lingbot_semantic_nav/src",
                description="Directory containing the shared lingbot_nav package",
            ),
            DeclareLaunchArgument("provider", default_value="rule"),
            DeclareLaunchArgument("allow_rule_fallback", default_value="False"),
            DeclareLaunchArgument("initial_x", default_value="-5.0"),
            DeclareLaunchArgument("initial_y", default_value="-10.0"),
            DeclareLaunchArgument("initial_yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "audit_log",
                default_value="/root/autodl-tmp/outputs/g1d_real_navigation_audit.jsonl",
            ),
            SetEnvironmentVariable(
                "PYTHONPATH",
                [
                    LaunchConfiguration("lingbot_pythonpath"),
                    os.pathsep,
                    EnvironmentVariable("PYTHONPATH", default_value=""),
                ],
            ),
            OpaqueFunction(function=_validated_nodes),
            Node(
                package="lingbot_semantic_nav_ros",
                executable="g1d_base_bridge",
                name="g1d_base_bridge",
                output="screen",
                parameters=[
                    base_params,
                    {
                        "allow_hardware_output": ParameterValue(
                            allow_hardware, value_type=bool
                        )
                    },
                ],
            ),
            navigation,
            Node(
                package="lingbot_semantic_nav_ros",
                executable="initial_pose_publisher",
                name="g1d_initial_pose",
                output="screen",
                parameters=[
                    {
                        "x": ParameterValue(
                            LaunchConfiguration("initial_x"), value_type=float
                        ),
                        "y": ParameterValue(
                            LaunchConfiguration("initial_y"), value_type=float
                        ),
                        "yaw": ParameterValue(
                            LaunchConfiguration("initial_yaw"), value_type=float
                        ),
                    }
                ],
            ),
            Node(
                package="lingbot_semantic_nav_ros",
                executable="language_goal_node",
                name="g1d_semantic_language_goal",
                output="screen",
                parameters=[
                    {
                        "places": places_file,
                        "map_yaml": map_file,
                        "robot_frame": "AGV_link",
                        "arrival_xy_tolerance": 0.20,
                        "arrival_yaw_tolerance": 0.20,
                        "provider": LaunchConfiguration("provider"),
                        "allow_rule_fallback": ParameterValue(
                            LaunchConfiguration("allow_rule_fallback"),
                            value_type=bool,
                        ),
                        "audit_log": LaunchConfiguration("audit_log"),
                    }
                ],
            ),
        ]
    )
