"""Publish startup odometry until the Habitat Nav2 bridge becomes active."""

import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class HabitatInitialOdom(Node):
    def __init__(self):
        super().__init__("habitat_initial_odom")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("yaw", 0.0)
        self.x = float(self.get_parameter("x").value)
        self.y = float(self.get_parameter("y").value)
        self.yaw = float(self.get_parameter("yaw").value)
        self.active = True
        self.odom = self.create_publisher(Odometry, "/odom", 20)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Bool, "/habitat_bridge_active", self._on_active, 10)
        self.create_timer(0.05, self._tick)

    def _on_active(self, message):
        self.active = not message.data

    def _tick(self):
        if not self.active:
            return
        stamp = self.get_clock().now().to_msg()
        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom.publish(odom)
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = HabitatInitialOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
