"""Legacy dashboard reproduction; not a formal wheel-physics validation path.

Use gazebo_wheel_nav2.launch.py for the official differential-drive physics,
LiDAR/IMU/odometry, AMCL, and Nav2 closed loop.
"""

import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


class HabitatNav2Params(RewrittenYaml):
    """Apply Habitat-specific nested/list overrides after scalar rewrites."""

    def perform(self, context):
        target = super().perform(context)
        with open(target, encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
        local_costmap = params["local_costmap"]["local_costmap"]["ros__parameters"]
        # There is no laser scan in the Habitat bridge.  Use the RGB-only
        # LingBot occupancy as DWB's obstacle source while leaving the local
        # costmap's default track_unknown_space=False policy unchanged.
        local_costmap["plugins"] = ["static_layer", "inflation_layer"]
        controller = params["controller_server"]["ros__parameters"]
        # DWB's endpoint critics can get trapped by the apartment's dining-room
        # entrance: the Navfn path must first travel west around a wall before
        # turning north, while GoalAlign / GoalDist reward turning directly
        # toward the final goal.  The result is a zero-linear-velocity
        # left/right oscillation in front of the wall.  Regulated Pure Pursuit
        # follows Nav2's global path instead of optimizing toward the endpoint,
        # and retains footprint/costmap collision checking.
        controller["controller_plugins"] = ["FollowPath"]
        controller["FollowPath"] = {
            "plugin": (
                "nav2_regulated_pure_pursuit_controller::"
                "RegulatedPurePursuitController"
            ),
            "desired_linear_vel": 0.26,
            "lookahead_dist": 0.30,
            "min_lookahead_dist": 0.15,
            "max_lookahead_dist": 0.40,
            "lookahead_time": 1.0,
            "rotate_to_heading_angular_vel": 0.60,
            "transform_tolerance": 0.20,
            "use_velocity_scaled_lookahead_dist": True,
            "min_approach_linear_velocity": 0.05,
            "approach_velocity_scaling_dist": 0.40,
            "use_collision_detection": True,
            "max_allowed_time_to_collision_up_to_carrot": 1.0,
            "use_regulated_linear_velocity_scaling": True,
            "use_cost_regulated_linear_velocity_scaling": True,
            "regulated_linear_scaling_min_radius": 0.40,
            "regulated_linear_scaling_min_speed": 0.05,
            "cost_scaling_dist": 0.30,
            "cost_scaling_gain": 1.0,
            "inflation_cost_scaling_factor": 3.0,
            "use_rotate_to_heading": True,
            "allow_reversing": False,
            "rotate_to_heading_min_angle": 0.50,
            "max_angular_accel": 3.2,
        }
        with open(target, "w", encoding="utf-8") as stream:
            yaml.safe_dump(params, stream)
        return target


def generate_launch_description():
    nav2_share = get_package_share_directory("nav2_bringup")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_yaw = LaunchConfiguration("initial_yaw")
    configured_params = HabitatNav2Params(
        source_file=params_file,
        root_key="",
        param_rewrites={
            "use_sim_time": "False",
            "robot_radius": "0.14",
            # Leave a meaningful safety corridor around reconstructed walls.
            # With 0.16 m inflation and a 0.14 m robot, Navfn can select a path
            # with only ~1 cm footprint margin at the dining-room doorway.
            "inflation_radius": "0.30",
            "cost_scaling_factor": "3.0",
            # The RGB-only reconstruction keeps unobserved cells as unknown.
            # Nav2 may traverse unknown space, but reconstructed obstacles and
            # inflation remain authoritative.  This avoids a Habitat-navmesh
            # fallback for targets outside the observed camera footprint.
            "allow_unknown": "True",
            "track_unknown_space": "True",
            # Keep the local controller on the same reconstructed geometry as
            # the global planner.  The stock local costmap only enables its voxel layer,
            # but this RGB-only demo has no laser scan source.  Without the
            # static layer DWB can cut through a reconstructed obstacle; the
            # next global replan then starts inside inflated cost and fails.
            # The local costmap's default track_unknown_space=False preserves
            # the existing policy that unobserved RGB-map cells are traversable.
            # Default Nav2 expects 0.5 m progress every 10 s.  That is too
            # coarse for this compact map and marks legitimate in-place turns
            # as failures.  Use a small translation threshold and allow time
            # for final heading alignment.
            "required_movement_radius": "0.02",
            "movement_time_allowance": "30.0",
            # Keep the bridge close to the RGB-supported scan corridor.  The
            # stock 0.25 m goal tolerance can stop a short waypoint segment on
            # an inflated cell, after which the controller has no safe continuation.
            "xy_goal_tolerance": "0.08",
            "yaw_goal_tolerance": "0.35",
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
        DeclareLaunchArgument("map"),
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
            name="map_to_odom",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "map", "--child-frame-id", "odom",
            ],
        ),
        Node(
            package="lingbot_semantic_nav_ros",
            executable="habitat_initial_odom",
            name="habitat_initial_odom",
            parameters=[{"x": initial_x, "y": initial_y, "yaw": initial_yaw}],
        ),
        navigation,
    ])
