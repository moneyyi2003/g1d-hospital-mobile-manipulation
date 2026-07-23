"""Legacy dashboard bridge; use gazebo_wheel_nav2.launch.py for formal tests."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from launch_ros.parameter_descriptions import ParameterValue


class LingBotWheelParams(RewrittenYaml):
    """Configure Nav2 for a small wheel robot and a static RGB-derived map."""

    def perform(self, context):
        target = super().perform(context)
        with open(target, encoding="utf-8") as stream:
            params = yaml.safe_load(stream)

        local_costmap = params["local_costmap"]["local_costmap"]["ros__parameters"]
        # The 8083 wheel simulator has exact odometry and no laser.  Both
        # global and local collision geometry therefore come from the LingBot
        # occupancy map; no ReplicaCAD navmesh or scene geometry is queried.
        local_costmap["plugins"] = ["static_layer", "inflation_layer"]

        controller = params["controller_server"]["ros__parameters"]
        controller["controller_plugins"] = ["FollowPath"]
        controller["FollowPath"] = {
            "plugin": (
                "nav2_regulated_pure_pursuit_controller::"
                "RegulatedPurePursuitController"
            ),
            "desired_linear_vel": 0.12,
            "lookahead_dist": 0.09,
            "min_lookahead_dist": 0.06,
            "max_lookahead_dist": 0.14,
            "lookahead_time": 1.0,
            "rotate_to_heading_angular_vel": 0.55,
            "transform_tolerance": 0.20,
            "use_velocity_scaled_lookahead_dist": True,
            "min_approach_linear_velocity": 0.025,
            "approach_velocity_scaling_dist": 0.18,
            # The differential-wheel bridge checks every 0.05 s integration
            # step against the LingBot occupancy inflated by robot_radius.
            # RPP's additional arc projection produced false positives at the
            # reconstructed bedroom bend even though the circular footprint
            # was still in free space.  Keep Nav2 cost regulation and the
            # bridge's authoritative footprint check, but avoid rejecting the
            # same safe bend twice with different discretizations.
            "use_collision_detection": False,
            "max_allowed_time_to_collision_up_to_carrot": 0.6,
            "use_regulated_linear_velocity_scaling": True,
            "use_cost_regulated_linear_velocity_scaling": True,
            "regulated_linear_scaling_min_radius": 0.14,
            "regulated_linear_scaling_min_speed": 0.025,
            "cost_scaling_dist": 0.08,
            "cost_scaling_gain": 1.0,
            "inflation_cost_scaling_factor": 4.0,
            "use_rotate_to_heading": True,
            "allow_reversing": False,
            "rotate_to_heading_min_angle": 0.45,
            "max_angular_accel": 2.5,
        }
        with open(target, "w", encoding="utf-8") as stream:
            yaml.safe_dump(params, stream)
        return target


def generate_launch_description():
    nav2_share = get_package_share_directory("nav2_bringup")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    robot_radius = LaunchConfiguration("robot_radius")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_yaw = LaunchConfiguration("initial_yaw")
    configured_params = LingBotWheelParams(
        source_file=params_file,
        root_key="",
        param_rewrites={
            "use_sim_time": "False",
            "robot_radius": robot_radius,
            "inflation_radius": "0.10",
            "cost_scaling_factor": "4.0",
            "allow_unknown": "False",
            "track_unknown_space": "True",
            "required_movement_radius": "0.01",
            "movement_time_allowance": "30.0",
            "xy_goal_tolerance": "0.05",
            "yaw_goal_tolerance": "0.30",
        },
        convert_types=True,
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_share, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": "False",
            "autostart": "True",
            "use_composition": "False",
            "params_file": configured_params,
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument("map", description="LingBot RGB-only ROS map YAML"),
        DeclareLaunchArgument("robot_radius", default_value="0.06"),
        DeclareLaunchArgument("initial_x", default_value="0.0"),
        DeclareLaunchArgument("initial_y", default_value="0.0"),
        DeclareLaunchArgument("initial_yaw", default_value="0.0"),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(nav2_share, "params", "nav2_params.yaml"),
        ),
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[{"yaml_filename": map_file, "use_sim_time": False}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_map",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "autostart": True,
                "node_names": ["map_server"],
            }],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="lingbot_map_to_odom",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "map", "--child-frame-id", "odom",
            ],
        ),
        Node(
            package="lingbot_semantic_nav_ros",
            executable="wheel_initial_odom",
            name="lingbot_8083_initial_odom",
            output="screen",
            parameters=[{
                "x": ParameterValue(initial_x, value_type=float),
                "y": ParameterValue(initial_y, value_type=float),
                "yaw": ParameterValue(initial_yaw, value_type=float),
            }],
        ),
        navigation,
    ])
