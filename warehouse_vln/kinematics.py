"""G1-D navigation-frame to imported-USD wheel conventions."""

from __future__ import annotations

import math


WHEEL_RADIUS_M = 0.0848
WHEEL_BASE_M = 0.4062

# The imported G1-D chassis rolls toward local -X for its positive forward
# wheel command.  Navigation uses the conventional local +X forward axis.
ROOT_FROM_NAVIGATION_YAW_RAD = math.pi

# The imported wheel joint axes make a positive navigation yaw require the
# opposite angular sign at the differential-drive command boundary.
USD_ANGULAR_SIGN = -1.0


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def navigation_yaw_to_root_yaw(navigation_yaw: float) -> float:
    return wrap_angle(navigation_yaw + ROOT_FROM_NAVIGATION_YAW_RAD)


def root_yaw_to_navigation_yaw(root_yaw: float) -> float:
    return wrap_angle(root_yaw - ROOT_FROM_NAVIGATION_YAW_RAD)


def navigation_twist_to_wheel_speeds(
    linear_mps: float,
    angular_radps: float,
    *,
    wheel_radius_m: float = WHEEL_RADIUS_M,
    wheel_base_m: float = WHEEL_BASE_M,
) -> tuple[float, float]:
    if wheel_radius_m <= 0.0 or wheel_base_m <= 0.0:
        raise ValueError("wheel radius and base must be positive")
    usd_angular = USD_ANGULAR_SIGN * angular_radps
    left = (
        linear_mps - usd_angular * wheel_base_m / 2.0
    ) / wheel_radius_m
    right = -(
        linear_mps + usd_angular * wheel_base_m / 2.0
    ) / wheel_radius_m
    return left, right


__all__ = [
    "ROOT_FROM_NAVIGATION_YAW_RAD",
    "USD_ANGULAR_SIGN",
    "WHEEL_BASE_M",
    "WHEEL_RADIUS_M",
    "navigation_twist_to_wheel_speeds",
    "navigation_yaw_to_root_yaw",
    "root_yaw_to_navigation_yaw",
    "wrap_angle",
]
