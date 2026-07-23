"""Official TurtleBot3 Gazebo physics + AMCL/Nav2 + semantic goal bridge.

This launch is the formal wheel-simulation path. Unlike the historical
dashboard bridge it never publishes exact simulator odometry, never draws a
trajectory by changing poses, and never inserts semantic waypoints.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    turtlebot3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    nav2_bringup = get_package_share_directory("nav2_bringup")
    map_file = LaunchConfiguration("map")
    places_file = LaunchConfiguration("places")
    params_file = LaunchConfiguration("params_file")
    model = LaunchConfiguration("model")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_yaw = LaunchConfiguration("initial_yaw")

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo, "launch", "turtlebot3_world.launch.py")
        ),
        launch_arguments={"x_pose": initial_x, "y_pose": initial_y}.items(),
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map": map_file,
            "use_sim_time": "True",
            "autostart": "True",
            "use_composition": "False",
            "params_file": params_file,
        }.items(),
    )
    semantic_goal = Node(
        package="lingbot_semantic_nav_ros",
        executable="language_goal_node",
        name="semantic_language_goal",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "places": places_file,
            "map_yaml": map_file,
            "provider": LaunchConfiguration("provider"),
            "allow_rule_fallback": ParameterValue(
                LaunchConfiguration("allow_rule_fallback"), value_type=bool
            ),
            "audit_log": LaunchConfiguration("audit_log"),
        }],
    )
    initial_pose = Node(
        package="lingbot_semantic_nav_ros",
        executable="initial_pose_publisher",
        name="semantic_nav_initial_pose",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "x": ParameterValue(initial_x, value_type=float),
            "y": ParameterValue(initial_y, value_type=float),
            "yaw": ParameterValue(initial_yaw, value_type=float),
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument("map", description="LingBot-Map RGB-only occupancy YAML"),
        DeclareLaunchArgument("places", description="Reviewed place catalog schema v2"),
        DeclareLaunchArgument("model", default_value="waffle_pi"),
        DeclareLaunchArgument("initial_x", default_value="-2.0"),
        DeclareLaunchArgument("initial_y", default_value="-0.5"),
        DeclareLaunchArgument("initial_yaw", default_value="0.0"),
        DeclareLaunchArgument("provider", default_value="deepseek"),
        DeclareLaunchArgument("allow_rule_fallback", default_value="False"),
        DeclareLaunchArgument("audit_log", default_value="outputs/navigation_audit.jsonl"),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(nav2_bringup, "params", "nav2_params.yaml"),
        ),
        SetEnvironmentVariable("TURTLEBOT3_MODEL", model),
        simulator,
        navigation,
        initial_pose,
        semantic_goal,
    ])
