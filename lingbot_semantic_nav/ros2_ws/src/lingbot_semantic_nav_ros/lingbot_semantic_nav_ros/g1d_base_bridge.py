"""Fail-closed ROS 2 command, wheel odometry, brake and e-stop bridge for G1-D."""

from __future__ import annotations

import json
import math

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from .g1d_base_core import SafetyController, SafetyLimits, WheelOdometry


class G1DBaseBridge(Node):
    """Final software safety boundary between Nav2 and a vendor base driver."""

    def __init__(self):
        super().__init__("g1d_base_bridge")
        self.declare_parameter("allow_hardware_output", False)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("driver_ready_topic", "/g1d/hardware/driver_ready")
        self.declare_parameter("estop_input_topic", "/g1d/hardware/estop")
        self.declare_parameter("hardware_command_topic", "/g1d/hardware/cmd_vel")
        self.declare_parameter("hardware_brake_topic", "/g1d/hardware/brake")
        self.declare_parameter("safe_command_topic", "/g1d/safety/safe_cmd_vel")
        self.declare_parameter("brake_state_topic", "/g1d/safety/brake")
        self.declare_parameter("status_topic", "/g1d/safety/status")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "AGV_link")
        self.declare_parameter("left_wheel_joint", "Left_Wheel_Joint")
        self.declare_parameter("right_wheel_joint", "Right_Wheel_Joint")
        self.declare_parameter("wheel_radius_m", 0.0848)
        self.declare_parameter("wheel_base_m", 0.4062)
        self.declare_parameter("left_encoder_sign", 1.0)
        self.declare_parameter("right_encoder_sign", -1.0)
        self.declare_parameter("max_linear_mps", 0.35)
        self.declare_parameter("max_angular_radps", 0.80)
        self.declare_parameter("max_linear_accel_mps2", 0.50)
        self.declare_parameter("max_angular_accel_radps2", 1.20)
        self.declare_parameter("command_timeout_s", 0.25)
        self.declare_parameter("feedback_timeout_s", 0.50)
        self.declare_parameter("driver_timeout_s", 0.50)
        self.declare_parameter("estop_timeout_s", 0.50)
        self.declare_parameter("control_rate_hz", 50.0)

        self.allow_hardware_output = bool(
            self.get_parameter("allow_hardware_output").value
        )
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.left_joint = str(self.get_parameter("left_wheel_joint").value)
        self.right_joint = str(self.get_parameter("right_wheel_joint").value)
        limits = SafetyLimits(
            max_linear_mps=float(self.get_parameter("max_linear_mps").value),
            max_angular_radps=float(self.get_parameter("max_angular_radps").value),
            max_linear_accel_mps2=float(
                self.get_parameter("max_linear_accel_mps2").value
            ),
            max_angular_accel_radps2=float(
                self.get_parameter("max_angular_accel_radps2").value
            ),
            command_timeout_s=float(self.get_parameter("command_timeout_s").value),
            feedback_timeout_s=float(
                self.get_parameter("feedback_timeout_s").value
            ),
            driver_timeout_s=float(self.get_parameter("driver_timeout_s").value),
            estop_timeout_s=float(self.get_parameter("estop_timeout_s").value),
        )
        self.safety = SafetyController(limits)
        self.odometer = WheelOdometry(
            wheel_radius_m=float(self.get_parameter("wheel_radius_m").value),
            wheel_base_m=float(self.get_parameter("wheel_base_m").value),
            left_sign=float(self.get_parameter("left_encoder_sign").value),
            right_sign=float(self.get_parameter("right_encoder_sign").value),
        )

        self.safe_command = self.create_publisher(
            Twist, str(self.get_parameter("safe_command_topic").value), 10
        )
        self.brake_state = self.create_publisher(
            Bool, str(self.get_parameter("brake_state_topic").value), 10
        )
        self.status = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.diagnostics = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10
        )
        self.odom_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("odom_topic").value), 20
        )
        self.hardware_command = self.create_publisher(
            Twist, str(self.get_parameter("hardware_command_topic").value), 10
        )
        self.hardware_brake = self.create_publisher(
            Bool, str(self.get_parameter("hardware_brake_topic").value), 10
        )
        self.tf = TransformBroadcaster(self)
        self.create_subscription(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            self._on_command,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("driver_ready_topic").value),
            self._on_driver_ready,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("estop_input_topic").value),
            self._on_estop_input,
            10,
        )
        self.create_service(Trigger, "/g1d/safety/arm", self._arm)
        self.create_service(Trigger, "/g1d/safety/disarm", self._disarm)
        self.create_service(Trigger, "/g1d/safety/brake_now", self._brake)
        self.create_service(Trigger, "/g1d/safety/estop", self._estop)
        self.create_service(Trigger, "/g1d/safety/clear_estop", self._clear_estop)
        rate = float(self.get_parameter("control_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        self.last_status_time = -math.inf
        self.last_output = self.safety.step(self._now())
        self.create_timer(1.0 / rate, self._control_tick)
        self.get_logger().warning(
            "G1-D bridge started disarmed with e-stop latched; hardware_output=%s"
            % self.allow_hardware_output
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_command(self, message: Twist) -> None:
        self.safety.set_command(
            float(message.linear.x), float(message.angular.z), self._now()
        )

    def _on_driver_ready(self, message: Bool) -> None:
        self.safety.update_driver_ready(bool(message.data), self._now())

    def _on_estop_input(self, message: Bool) -> None:
        self.safety.set_estop_input(bool(message.data), self._now())

    def _wheel_values(self, message: JointState):
        names = list(message.name)
        try:
            left_index = names.index(self.left_joint)
            right_index = names.index(self.right_joint)
            left_position = float(message.position[left_index])
            right_position = float(message.position[right_index])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"joint_states must contain positions for "
                f"{self.left_joint}/{self.right_joint}"
            ) from exc
        left_velocity = None
        right_velocity = None
        if len(message.velocity) > max(left_index, right_index):
            left_velocity = float(message.velocity[left_index])
            right_velocity = float(message.velocity[right_index])
        return left_position, right_position, left_velocity, right_velocity

    def _on_joint_state(self, message: JointState) -> None:
        now = self._now()
        try:
            left, right, left_velocity, right_velocity = self._wheel_values(message)
            sample = self.odometer.update(
                left,
                right,
                now,
                left_velocity_radps=left_velocity,
                right_velocity_radps=right_velocity,
            )
        except ValueError as exc:
            self.safety.emergency_stop("invalid_wheel_feedback")
            self.get_logger().error(str(exc))
            return
        self.safety.update_feedback(now)
        if sample is not None:
            self._publish_odom(sample)

    def _publish_odom(self, sample) -> None:
        stamp = self.get_clock().now().to_msg()
        qz = math.sin(sample.yaw / 2.0)
        qw = math.cos(sample.yaw / 2.0)
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = sample.x
        message.pose.pose.position.y = sample.y
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw
        message.twist.twist.linear.x = sample.linear_mps
        message.twist.twist.angular.z = sample.angular_radps
        message.pose.covariance[0] = 0.02
        message.pose.covariance[7] = 0.02
        message.pose.covariance[35] = 0.05
        message.twist.covariance[0] = 0.03
        message.twist.covariance[35] = 0.08
        self.odom_publisher.publish(message)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = sample.x
        transform.transform.translation.y = sample.y
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf.sendTransform(transform)

    def _control_tick(self) -> None:
        now = self._now()
        output = self.safety.step(now)
        self.last_output = output
        command = Twist()
        command.linear.x = output.linear_mps
        command.angular.z = output.angular_radps
        brake = Bool(data=output.brake)
        self.safe_command.publish(command)
        self.brake_state.publish(brake)
        if self.allow_hardware_output:
            self.hardware_command.publish(command)
            self.hardware_brake.publish(brake)
        if now - self.last_status_time >= 0.20:
            self._publish_status(now)
            self.last_status_time = now

    def _publish_status(self, now: float) -> None:
        output = self.last_output
        payload = {
            "timestamp": now,
            "allow_hardware_output": self.allow_hardware_output,
            "armed": output.armed,
            "estop_latched": output.estop_latched,
            "estop_input": self.safety.estop_input,
            "estop_input_fresh": self.safety.estop_input_fresh(now),
            "driver_ready": self.safety.driver_ready,
            "brake": output.brake,
            "reason": output.reason,
            "safe_command": {
                "linear_mps": output.linear_mps,
                "angular_radps": output.angular_radps,
            },
            "frames": {"odom": self.odom_frame, "base": self.base_frame},
        }
        status = String(data=json.dumps(payload, ensure_ascii=False))
        self.status.publish(status)
        diagnostic = DiagnosticStatus()
        diagnostic.name = "g1d_base_safety"
        diagnostic.hardware_id = "g1_d"
        if output.estop_latched:
            diagnostic.level = DiagnosticStatus.ERROR
        elif output.brake or not output.armed:
            diagnostic.level = DiagnosticStatus.WARN
        else:
            diagnostic.level = DiagnosticStatus.OK
        diagnostic.message = output.reason
        diagnostic.values = [
            KeyValue(key="hardware_output", value=str(self.allow_hardware_output)),
            KeyValue(key="armed", value=str(output.armed)),
            KeyValue(key="brake", value=str(output.brake)),
            KeyValue(key="driver_ready", value=str(self.safety.driver_ready)),
            KeyValue(
                key="estop_input_fresh",
                value=str(self.safety.estop_input_fresh(now)),
            ),
        ]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [diagnostic]
        self.diagnostics.publish(array)

    def _arm(self, _request, response):
        response.success, response.message = self.safety.arm(
            self._now(), hardware_output_enabled=self.allow_hardware_output
        )
        return response

    def _disarm(self, _request, response):
        self.safety.disarm("operator_disarm")
        response.success = True
        response.message = self.safety.reason
        return response

    def _brake(self, _request, response):
        self.safety.disarm("operator_brake")
        response.success = True
        response.message = self.safety.reason
        return response

    def _estop(self, _request, response):
        self.safety.emergency_stop("operator_estop")
        response.success = True
        response.message = self.safety.reason
        return response

    def _clear_estop(self, _request, response):
        response.success, response.message = self.safety.clear_estop(self._now())
        return response


def main(args=None):
    rclpy.init(args=args)
    node = G1DBaseBridge()
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
