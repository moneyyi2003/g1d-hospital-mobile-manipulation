"""Resolve constrained place IDs and execute them as ordered Nav2 goals."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from lingbot_nav.intent import create_intent_parser
from lingbot_nav.mission import MissionResolver
from lingbot_nav.models import RouteAction
from lingbot_nav.place_db import PlaceDatabase, normalize_label
from lingbot_nav.place_catalog_builder import map_bundle_sha256


def _yaw_from_quaternion(value) -> float:
    return math.atan2(
        2.0 * (value.w * value.z + value.x * value.y),
        1.0 - 2.0 * (value.y * value.y + value.z * value.z),
    )


def _angle_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


class LanguageGoalNode(Node):
    """The only online component allowed to turn place IDs into poses."""

    def __init__(self):
        super().__init__("semantic_language_goal")
        self.declare_parameter("places", "")
        self.declare_parameter("provider", "deepseek")
        self.declare_parameter("allow_rule_fallback", False)
        self.declare_parameter("expected_map_id", "")
        self.declare_parameter("expected_map_sha256", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("command_topic", "/semantic_nav/command")
        self.declare_parameter("status_topic", "/semantic_nav/status")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("arrival_xy_tolerance", 0.30)
        self.declare_parameter("arrival_yaw_tolerance", 0.50)
        self.declare_parameter("verify_arrival_tf", True)
        self.declare_parameter("audit_log", "")

        places_path = str(self.get_parameter("places").value)
        if not places_path:
            raise RuntimeError("The places parameter is required")
        expected_map_sha256 = str(self.get_parameter("expected_map_sha256").value)
        map_yaml = str(self.get_parameter("map_yaml").value)
        if map_yaml:
            active_map_hash = map_bundle_sha256(map_yaml)
            if expected_map_sha256 and expected_map_sha256 != active_map_hash:
                raise RuntimeError("Configured expected map hash disagrees with map_yaml")
            expected_map_sha256 = active_map_hash
        self.places = PlaceDatabase.load(
            places_path,
            allow_legacy=False,
            expected_map_id=str(self.get_parameter("expected_map_id").value),
            expected_map_sha256=expected_map_sha256,
        )
        parser = create_intent_parser(
            str(self.get_parameter("provider").value),
            self.places,
            allow_rule_fallback=bool(self.get_parameter("allow_rule_fallback").value),
        )
        self.resolver = MissionResolver(parser, self.places)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.xy_tolerance = float(self.get_parameter("arrival_xy_tolerance").value)
        self.yaw_tolerance = float(self.get_parameter("arrival_yaw_tolerance").value)
        self.verify_arrival_tf = bool(self.get_parameter("verify_arrival_tf").value)
        self.audit_log = str(self.get_parameter("audit_log").value)

        self.status = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 20
        )
        self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
        )
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.mission = None
        self.step_index = -1
        self.pose_index = -1
        self.active_poses = ()
        self.goal_handle = None
        self.cancel_requested = False
        self.mission_id = ""
        self._publish("ready", map_id=self.places.map_id, place_count=len(self.places.places))

    def _audit(self, payload: dict) -> None:
        if not self.audit_log:
            return
        target = Path(self.audit_log).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _publish(self, event: str, **fields) -> None:
        payload = {
            "timestamp": time.time(),
            "event": event,
            "mission_id": self.mission_id,
            **fields,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.status.publish(message)
        self._audit(payload)

    def _on_command(self, message: String) -> None:
        command = message.data.strip()
        normalized = normalize_label(command)
        if any(token in normalized for token in ("取消", "停止", "cancel", "stop")):
            self._cancel_active()
            return
        if self.mission is not None:
            self._publish("rejected", reason="busy", command=command)
            return
        try:
            mission = self.resolver.resolve(command)
        except Exception as exc:  # ROS boundary: convert domain/provider errors to status.
            self._publish("rejected", reason=str(exc), command=command)
            return
        self.mission = mission
        self.cancel_requested = False
        self.step_index = -1
        self.pose_index = -1
        self.active_poses = ()
        self.mission_id = f"mission-{time.time_ns()}"
        requested = [
            {"place_id": step.place.place_id, "action": step.action.value}
            for step in mission.steps
        ]
        self._publish(
            "accepted",
            command=command,
            parser=mission.intent.parser,
            requested_steps=requested,
            # This exact list is the anti-cheating resolution contract. Actual
            # submissions are separately recorded as goal_submitted events,
            # so B is not falsely marked submitted when A fails.
            resolved_place_ids=[item["place_id"] for item in requested],
        )
        self._send_next_step()

    def _cancel_active(self) -> None:
        if self.mission is None:
            self._publish("cancel_ignored", reason="no_active_mission")
            return
        self._publish("canceling")
        self.cancel_requested = True
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def _send_next_step(self) -> None:
        assert self.mission is not None
        self.step_index += 1
        if self.step_index >= len(self.mission.steps):
            self._finish("succeeded")
            return
        step = self.mission.steps[self.step_index]
        self.active_poses = step.place.navigation_poses(step.action)
        self.pose_index = 0
        self._send_active_pose()

    def _send_active_pose(self) -> None:
        assert self.mission is not None
        if not self.nav.wait_for_server(timeout_sec=2.0):
            self._finish("failed", reason="navigate_to_pose action server unavailable")
            return
        step = self.mission.steps[self.step_index]
        pose = self.active_poses[self.pose_index]
        phase = (
            "approach"
            if self.pose_index < len(self.active_poses) - 1
            else "destination"
        )
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = pose.frame_id
        goal.pose.pose.position.x = pose.x
        goal.pose.pose.position.y = pose.y
        goal.pose.pose.orientation.z = math.sin(pose.yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(pose.yaw / 2.0)
        self._publish(
            "goal_submitted",
            step_index=self.step_index,
            place_id=step.place.place_id,
            action=step.action.value,
            phase=phase,
            pose=pose.to_dict(),
        )
        future = self.nav.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        try:
            self.goal_handle = future.result()
        except Exception as exc:
            self._finish("failed", reason=f"goal submission failed: {exc}")
            return
        if not self.goal_handle.accepted:
            self._finish("failed", reason="Nav2 rejected goal")
            return
        if self.cancel_requested:
            self.goal_handle.cancel_goal_async()
        self.goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_feedback(self, feedback) -> None:
        remaining = getattr(feedback.feedback, "distance_remaining", None)
        self._publish(
            "navigating",
            step_index=self.step_index,
            distance_remaining=float(remaining) if remaining is not None else None,
        )

    def _arrival_verified(self, target, action: RouteAction) -> tuple[bool, str]:
        if not self.verify_arrival_tf:
            return True, "tf verification disabled"
        try:
            transform = self.tf_buffer.lookup_transform(
                target.frame_id, self.robot_frame, Time()
            )
        except TransformException as exc:
            return False, f"arrival TF unavailable: {exc}"
        actual_x = float(transform.transform.translation.x)
        actual_y = float(transform.transform.translation.y)
        actual_yaw = _yaw_from_quaternion(transform.transform.rotation)
        xy_error = math.hypot(actual_x - target.x, actual_y - target.y)
        yaw_error = _angle_error(actual_yaw, target.yaw)
        # A pass-through waypoint has no orientation obligation. A requested
        # arrival does, including intermediate A in "先到 A，再到 B".
        yaw_ok = action == RouteAction.PASS or yaw_error <= self.yaw_tolerance
        if xy_error <= self.xy_tolerance and yaw_ok:
            return True, f"xy_error={xy_error:.3f}, yaw_error={yaw_error:.3f}"
        return False, f"arrival outside tolerance: xy={xy_error:.3f}, yaw={yaw_error:.3f}"

    def _on_result(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._finish("failed", reason=f"Nav2 result failed: {exc}")
            return
        if self.cancel_requested or result.status == GoalStatus.STATUS_CANCELED:
            self._finish("canceled")
            return
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._finish("failed", reason=f"Nav2 terminal status {result.status}")
            return
        assert self.mission is not None
        step = self.mission.steps[self.step_index]
        target = self.active_poses[self.pose_index]
        is_approach = self.pose_index < len(self.active_poses) - 1
        verified, detail = self._arrival_verified(
            target,
            RouteAction.ARRIVE if is_approach else step.action,
        )
        if not verified:
            self._finish("failed", reason=detail)
            return
        if is_approach:
            self._publish(
                "docking_approach_reached",
                step_index=self.step_index,
                place_id=step.place.place_id,
                verification=detail,
            )
            self.goal_handle = None
            self.pose_index += 1
            self._send_active_pose()
            return
        self._publish(
            "step_reached",
            step_index=self.step_index,
            place_id=step.place.place_id,
            action=step.action.value,
            verification=detail,
        )
        self.goal_handle = None
        self.cancel_requested = False
        self._send_next_step()

    def _finish(self, event: str, **fields) -> None:
        self._publish(event, **fields)
        self.mission = None
        self.step_index = -1
        self.pose_index = -1
        self.active_poses = ()
        self.goal_handle = None
        self.cancel_requested = False
        self.mission_id = ""


def main(args=None):
    rclpy.init(args=args)
    node = LanguageGoalNode()
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
