"""Dependency-light G1-D differential-drive safety and odometry core."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class SafetyLimits:
    max_linear_mps: float = 0.35
    max_angular_radps: float = 0.80
    max_linear_accel_mps2: float = 0.50
    max_angular_accel_radps2: float = 1.20
    command_timeout_s: float = 0.25
    feedback_timeout_s: float = 0.50
    driver_timeout_s: float = 0.50
    estop_timeout_s: float = 0.50

    def validate(self) -> None:
        if min(
            self.max_linear_mps,
            self.max_angular_radps,
            self.max_linear_accel_mps2,
            self.max_angular_accel_radps2,
            self.command_timeout_s,
            self.feedback_timeout_s,
            self.driver_timeout_s,
            self.estop_timeout_s,
        ) <= 0.0:
            raise ValueError("all G1-D safety limits must be positive")


@dataclass(frozen=True)
class SafeOutput:
    linear_mps: float
    angular_radps: float
    brake: bool
    armed: bool
    estop_latched: bool
    reason: str


class SafetyController:
    """Fail-closed command gate; startup is e-stopped and disarmed."""

    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self.limits.validate()
        self.armed = False
        self.estop_latched = True
        self.estop_input = False
        self.last_estop_time: float | None = None
        self.driver_ready = False
        self.last_driver_time: float | None = None
        self.last_feedback_time: float | None = None
        self.last_command_time: float | None = None
        self.command_linear = 0.0
        self.command_angular = 0.0
        self.output_linear = 0.0
        self.output_angular = 0.0
        self.last_step_time: float | None = None
        self.reason = "startup_estop_latched"

    def set_command(self, linear_mps: float, angular_radps: float, now: float) -> None:
        if not all(math.isfinite(value) for value in (linear_mps, angular_radps, now)):
            self.emergency_stop("non_finite_command")
            return
        self.command_linear = _clamp(
            linear_mps, -self.limits.max_linear_mps, self.limits.max_linear_mps
        )
        self.command_angular = _clamp(
            angular_radps,
            -self.limits.max_angular_radps,
            self.limits.max_angular_radps,
        )
        self.last_command_time = now

    def update_feedback(self, now: float) -> None:
        if math.isfinite(now):
            self.last_feedback_time = now

    def update_driver_ready(self, ready: bool, now: float) -> None:
        self.driver_ready = bool(ready)
        self.last_driver_time = now
        if not self.driver_ready:
            self.disarm("driver_not_ready")

    def set_estop_input(self, active: bool, now: float) -> None:
        self.estop_input = bool(active)
        if math.isfinite(now):
            self.last_estop_time = now
        else:
            self.emergency_stop("non_finite_estop_timestamp")
            return
        if self.estop_input:
            self.emergency_stop("hardware_estop_input")

    def emergency_stop(self, reason: str = "emergency_stop") -> None:
        self.estop_latched = True
        self.armed = False
        self.output_linear = 0.0
        self.output_angular = 0.0
        self.reason = reason

    def disarm(self, reason: str = "disarmed") -> None:
        self.armed = False
        self.output_linear = 0.0
        self.output_angular = 0.0
        self.reason = reason

    def _fresh(self, timestamp: float | None, now: float, timeout: float) -> bool:
        return timestamp is not None and 0.0 <= now - timestamp <= timeout

    def estop_input_fresh(self, now: float) -> bool:
        return self._fresh(
            self.last_estop_time, now, self.limits.estop_timeout_s
        )

    def clear_estop(self, now: float) -> tuple[bool, str]:
        if not self.estop_input_fresh(now):
            return False, "hardware estop heartbeat is not fresh"
        if self.estop_input:
            return False, "hardware estop input is still active"
        if not self.driver_ready or not self._fresh(
            self.last_driver_time, now, self.limits.driver_timeout_s
        ):
            return False, "vendor driver is not ready/fresh"
        if not self._fresh(
            self.last_feedback_time, now, self.limits.feedback_timeout_s
        ):
            return False, "wheel feedback is not fresh"
        self.estop_latched = False
        self.reason = "estop_cleared_disarmed"
        return True, self.reason

    def arm(self, now: float, *, hardware_output_enabled: bool) -> tuple[bool, str]:
        if not hardware_output_enabled:
            return False, "hardware output is disabled by configuration"
        if not self.estop_input_fresh(now):
            return False, "hardware estop heartbeat is not fresh"
        if self.estop_latched or self.estop_input:
            return False, "estop is latched or physically active"
        if not self.driver_ready or not self._fresh(
            self.last_driver_time, now, self.limits.driver_timeout_s
        ):
            return False, "vendor driver is not ready/fresh"
        if not self._fresh(
            self.last_feedback_time, now, self.limits.feedback_timeout_s
        ):
            return False, "wheel feedback is not fresh"
        self.armed = True
        self.reason = "armed_waiting_for_command"
        return True, self.reason

    def step(self, now: float) -> SafeOutput:
        if self.last_step_time is None:
            dt = 0.0
        else:
            dt = _clamp(now - self.last_step_time, 0.0, 0.10)
        self.last_step_time = now
        feedback_fresh = self._fresh(
            self.last_feedback_time, now, self.limits.feedback_timeout_s
        )
        driver_fresh = self.driver_ready and self._fresh(
            self.last_driver_time, now, self.limits.driver_timeout_s
        )
        command_fresh = self._fresh(
            self.last_command_time, now, self.limits.command_timeout_s
        )
        estop_fresh = self.estop_input_fresh(now)

        if not estop_fresh and not self.estop_latched:
            self.emergency_stop("estop_input_watchdog_timeout")
            reason = self.reason
        elif self.estop_latched or self.estop_input:
            reason = self.reason or "estop_latched"
        elif not self.armed:
            reason = self.reason or "disarmed"
        elif not driver_fresh:
            self.disarm("driver_watchdog_timeout")
            reason = self.reason
        elif not feedback_fresh:
            self.disarm("feedback_watchdog_timeout")
            reason = self.reason
        elif not command_fresh:
            self.output_linear = 0.0
            self.output_angular = 0.0
            reason = "command_watchdog_brake"
        else:
            linear_delta = self.limits.max_linear_accel_mps2 * dt
            angular_delta = self.limits.max_angular_accel_radps2 * dt
            self.output_linear += _clamp(
                self.command_linear - self.output_linear,
                -linear_delta,
                linear_delta,
            )
            self.output_angular += _clamp(
                self.command_angular - self.output_angular,
                -angular_delta,
                angular_delta,
            )
            reason = "command_active"

        brake = reason != "command_active"
        if brake:
            self.output_linear = 0.0
            self.output_angular = 0.0
        self.reason = reason
        return SafeOutput(
            self.output_linear,
            self.output_angular,
            brake,
            self.armed,
            self.estop_latched,
            reason,
        )


@dataclass(frozen=True)
class OdomSample:
    x: float
    y: float
    yaw: float
    linear_mps: float
    angular_radps: float


class WheelOdometry:
    """Integrate the opposed-axis G1-D wheel encoders in the odom frame."""

    def __init__(
        self,
        *,
        wheel_radius_m: float = 0.0848,
        wheel_base_m: float = 0.4062,
        left_sign: float = 1.0,
        right_sign: float = -1.0,
    ):
        if wheel_radius_m <= 0.0 or wheel_base_m <= 0.0:
            raise ValueError("wheel radius and wheel base must be positive")
        self.radius = wheel_radius_m
        self.base = wheel_base_m
        self.left_sign = left_sign
        self.right_sign = right_sign
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.previous_left: float | None = None
        self.previous_right: float | None = None
        self.previous_time: float | None = None

    def update(
        self,
        left_position_rad: float,
        right_position_rad: float,
        now: float,
        *,
        left_velocity_radps: float | None = None,
        right_velocity_radps: float | None = None,
    ) -> OdomSample | None:
        values = (left_position_rad, right_position_rad, now)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("wheel positions/time must be finite")
        if self.previous_left is None:
            self.previous_left = left_position_rad
            self.previous_right = right_position_rad
            self.previous_time = now
            return None
        assert self.previous_right is not None and self.previous_time is not None
        dt = now - self.previous_time
        if dt <= 0.0:
            return None
        delta_left = self.left_sign * (left_position_rad - self.previous_left)
        delta_right = self.right_sign * (right_position_rad - self.previous_right)
        distance_left = self.radius * delta_left
        distance_right = self.radius * delta_right
        distance = 0.5 * (distance_left + distance_right)
        delta_yaw = (distance_right - distance_left) / self.base
        self.x += distance * math.cos(self.yaw + 0.5 * delta_yaw)
        self.y += distance * math.sin(self.yaw + 0.5 * delta_yaw)
        self.yaw = math.atan2(
            math.sin(self.yaw + delta_yaw), math.cos(self.yaw + delta_yaw)
        )
        if (
            left_velocity_radps is not None
            and right_velocity_radps is not None
            and math.isfinite(left_velocity_radps)
            and math.isfinite(right_velocity_radps)
        ):
            velocity_left = self.radius * self.left_sign * left_velocity_radps
            velocity_right = self.radius * self.right_sign * right_velocity_radps
        else:
            velocity_left = distance_left / dt
            velocity_right = distance_right / dt
        self.previous_left = left_position_rad
        self.previous_right = right_position_rad
        self.previous_time = now
        return OdomSample(
            self.x,
            self.y,
            self.yaw,
            0.5 * (velocity_left + velocity_right),
            (velocity_right - velocity_left) / self.base,
        )


__all__ = [
    "OdomSample",
    "SafeOutput",
    "SafetyController",
    "SafetyLimits",
    "WheelOdometry",
]
