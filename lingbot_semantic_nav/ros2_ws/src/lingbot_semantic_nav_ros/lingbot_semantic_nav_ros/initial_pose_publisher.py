"""Publish a configured AMCL initial pose; never publishes continuous truth pose."""

import math

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__("semantic_nav_initial_pose")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("publish_count", 10)
        self.x = float(self.get_parameter("x").value)
        self.y = float(self.get_parameter("y").value)
        self.yaw = float(self.get_parameter("yaw").value)
        self.remaining = int(self.get_parameter("publish_count").value)
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", qos)
        self.timer = self.create_timer(0.5, self._publish)

    def _publish(self):
        if self.remaining <= 0:
            self.timer.cancel()
            return
        message = PoseWithCovarianceStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.pose.pose.position.x = self.x
        message.pose.pose.position.y = self.y
        message.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        message.pose.covariance[0] = 0.05
        message.pose.covariance[7] = 0.05
        message.pose.covariance[35] = 0.10
        self.publisher.publish(message)
        self.remaining -= 1


def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
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

