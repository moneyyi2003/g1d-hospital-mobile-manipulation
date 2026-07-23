from glob import glob
import os
from setuptools import find_packages, setup


package_name = "lingbot_semantic_nav_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "data"), glob("data/*.json")),
        (os.path.join("share", package_name, "params"), glob("params/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLN workspace",
    maintainer_email="maintainer@example.com",
    description="Ordered semantic-place routes executed as Nav2 NavigateToPose goals",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "habitat_initial_odom = lingbot_semantic_nav_ros.habitat_initial_odom:main",
            "wheel_initial_odom = lingbot_semantic_nav_ros.wheel_initial_odom:main",
            "language_goal_node = lingbot_semantic_nav_ros.language_goal_node:main",
            "initial_pose_publisher = lingbot_semantic_nav_ros.initial_pose_publisher:main",
        ],
    },
)
