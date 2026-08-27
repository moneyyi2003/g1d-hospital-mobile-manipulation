"""Dependency-light live state publishing for the Family Home dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Sequence

from simple_room_vln.core import Pose2D


def _atomic_bytes(root: Path, name: str, payload: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


class LivePublisher:
    """Atomically publish simulator frames and navigation state for HTTP readers."""

    def __init__(
        self,
        root: Path,
        *,
        command: str,
        task: str,
        map_source: str,
        path: Sequence[tuple[float, float]],
        mission_mode: str = "vln_navigation",
        mission_steps: Sequence[str] = (),
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.command = command
        self.task = task
        self.map_source = map_source
        self.mission_mode = mission_mode
        self.mission_steps = list(mission_steps)
        self.planned = [{"x": x, "y": y} for x, y in path]
        self.trajectory: list[dict[str, float]] = []
        self.sequence = 0
        self.last_pose: Pose2D | None = None

    def publish_state(
        self,
        *,
        state: str,
        message: str,
        frame: int,
        action: str,
        pose: Pose2D,
        linear: float,
        angular: float,
        waypoint: int,
        waypoint_count: int,
        result: dict | None = None,
    ) -> None:
        if self.last_pose is None or math.dist(
            (pose.x, pose.y), (self.last_pose.x, self.last_pose.y)
        ) >= 0.025:
            self.trajectory.append({"x": pose.x, "y": pose.y, "yaw": pose.yaw})
            self.last_pose = pose
        payload = {
            "schema_version": 1,
            "sequence": self.sequence,
            "updated_at": time.time(),
            "state": state,
            "message": message,
            "command": self.command,
            "task": self.task,
            "map_source": self.map_source,
            "mission_mode": self.mission_mode,
            "mission_steps": self.mission_steps,
            "frame": frame,
            "action": action,
            "pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
            "linear_velocity_mps": linear,
            "angular_velocity_rps": angular,
            "waypoint": waypoint,
            "waypoint_count": waypoint_count,
            "planned_trajectory": self.planned,
            "trajectory": self.trajectory,
            # Keep the historical overview stream for route supervision and
            # publish the robot-mounted RGB stream independently.  A browser
            # can subscribe to either one without delaying the other.
            "camera_url": "/stream/camera.mjpg",
            "robot_camera_url": "/stream/robot-camera.mjpg",
            "result": result,
        }
        self.sequence += 1
        _atomic_bytes(
            self.root,
            "state.json",
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )

    def publish_image(self, image, *, stream: str = "overview") -> None:
        """Publish the newest JPEG for one named live camera stream.

        ``overview`` remains the backwards-compatible dashboard camera;
        ``robot`` is the calibrated ego-centric camera rigidly carried by the
        G1-D during navigation.
        """
        from io import BytesIO
        from PIL import Image

        filenames = {"overview": "camera.jpg", "robot": "robot_camera.jpg"}
        try:
            filename = filenames[stream]
        except KeyError as exc:
            raise ValueError(f"unsupported live camera stream: {stream}") from exc
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=82)
        _atomic_bytes(self.root, filename, stream.getvalue())


def publish_failure(root: Path, *, command: str, message: str, pose: Pose2D) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": time.time(),
        "state": "failed",
        "message": message,
        "command": command,
        "task": None,
        "frame": 0,
        "action": "error",
        "pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
        "linear_velocity_mps": 0.0,
        "angular_velocity_rps": 0.0,
        "waypoint": 0,
        "waypoint_count": 0,
        "planned_trajectory": [],
        "trajectory": [],
        "result": None,
    }
    _atomic_bytes(
        root.resolve(),
        "state.json",
        (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )


__all__ = ["LivePublisher", "publish_failure"]
