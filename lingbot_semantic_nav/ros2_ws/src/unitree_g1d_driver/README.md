# Unitree G1-D ROS 2 adapter

This package converts the existing fail-closed G1-D hardware topics into the
official Unitree SDK2 `g1::AgvClient::Move(vx, 0, wz)` call.

It deliberately does **not** synthesize wheel encoder feedback. The checked SDK
revision provides G1-D AGV motion, lift-column and arm examples, but no confirmed
left/right wheel encoder topic. Until real encoder feedback is mapped to
`/joint_states`, the upstream `g1d_base_bridge` feedback watchdog prevents arm.

The SDK exposes no dedicated G1-D mechanical-brake or hard-e-stop call in this
revision. A brake request therefore has only `Move(0,0,0)` semantics. A physical
hard-e-stop circuit and its independently monitored ROS heartbeat remain
mandatory.

Default parameters keep DDS disconnected and non-zero motion disabled. Never
enable the driver against a robot that is not on stands and inside an isolated
test area.

```bash
cd /root/autodl-tmp/lingbot_semantic_nav/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select unitree_g1d_driver lingbot_semantic_nav_ros
```

Dry-run, with no SDK network activity:

```bash
source install/setup.bash
ros2 run unitree_g1d_driver unitree_g1d_driver_node \
  --ros-args --params-file \
  install/unitree_g1d_driver/share/unitree_g1d_driver/params/unitree_g1d_driver.yaml
```

The two SDK gates are `connect_sdk` and `allow_sdk_motion`. The upstream safety
bridge has a third, independent `allow_hardware_output` gate. Opening the gates
is not sufficient for physical operation: fresh hardware e-stop and real wheel
feedback are also required.
