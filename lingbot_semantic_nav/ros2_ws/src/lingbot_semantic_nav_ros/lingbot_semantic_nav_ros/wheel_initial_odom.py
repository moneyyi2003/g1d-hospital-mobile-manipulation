"""Hand initial odometry to the 8083 bridge without resetting between tasks."""

import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class WheelInitialOdom(Node):
    def __init__(self):
        super().__init__("lingbot_8083_initial_odom")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("yaw", 0.0)
        self.x = float(self.get_parameter("x").value)
        self.y = float(self.get_parameter("y").value)
        self.yaw = float(self.get_parameter("yaw").value)
        self.active = True
        self.odom = self.create_publisher(Odometry, "/odom", 20)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/odom", self._remember_bridge_pose, 20)
        self.create_subscription(
            Bool, "/lingbot_wheel_bridge_active", self._on_active, 10
        )
        self.create_timer(0.05, self._tick)

    def _on_active(self, message):
        self.active = not message.data

    def _remember_bridge_pose(self, message):
        # While the long-lived bridge owns /odom, retain its newest pose.  If
        # it ever releases ownership, the fallback publisher continues from
        # that pose instead of jumping back to the dashboard's initial pose.
        if self.active or message.child_frame_id != "base_link":
            return
        self.x = float(message.pose.pose.position.x)
        self.y = float(message.pose.pose.position.y)
        orientation = message.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )

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
    node = WheelInitialOdom()
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
