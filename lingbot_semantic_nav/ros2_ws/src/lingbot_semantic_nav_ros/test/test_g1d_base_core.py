import math

from lingbot_semantic_nav_ros.g1d_base_core import SafetyController, WheelOdometry


def test_startup_and_disabled_hardware_are_fail_closed():
    gate = SafetyController()
    gate.update_driver_ready(True, 1.0)
    gate.update_feedback(1.0)
    gate.set_estop_input(False, 1.0)

    cleared, _ = gate.clear_estop(1.0)
    armed, reason = gate.arm(1.0, hardware_output_enabled=False)
    output = gate.step(1.01)

    assert cleared
    assert not armed
    assert "disabled" in reason
    assert output.brake
    assert output.linear_mps == 0.0


def test_command_watchdog_and_estop_zero_immediately():
    gate = SafetyController()
    gate.update_driver_ready(True, 1.0)
    gate.update_feedback(1.0)
    gate.set_estop_input(False, 1.0)
    assert gate.clear_estop(1.0)[0]
    assert gate.arm(1.0, hardware_output_enabled=True)[0]
    gate.set_command(0.3, 0.2, 1.0)
    gate.step(1.01)
    active = gate.step(1.1)
    timed_out = gate.step(1.4)
    gate.emergency_stop()
    stopped = gate.step(1.41)

    assert not active.brake
    assert active.linear_mps > 0.0
    assert timed_out.brake
    assert timed_out.linear_mps == 0.0
    assert stopped.estop_latched
    assert stopped.brake


def test_estop_heartbeat_is_required_and_loss_latches_stop():
    gate = SafetyController()
    gate.update_driver_ready(True, 1.0)
    gate.update_feedback(1.0)
    assert not gate.clear_estop(1.0)[0]

    gate.set_estop_input(False, 1.0)
    assert gate.clear_estop(1.0)[0]
    assert gate.arm(1.0, hardware_output_enabled=True)[0]
    gate.set_command(0.2, 0.0, 1.0)
    assert not gate.step(1.1).brake

    stopped = gate.step(1.6)
    assert stopped.brake
    assert stopped.estop_latched
    assert stopped.reason == "estop_input_watchdog_timeout"


def test_opposed_wheel_axes_integrate_forward_and_turn():
    odom = WheelOdometry()
    assert odom.update(0.0, 0.0, 0.0) is None
    forward = odom.update(1.0, -1.0, 1.0)
    turning = odom.update(2.0, -0.5, 2.0)

    assert forward is not None
    assert math.isclose(forward.x, 0.0848, rel_tol=1e-6)
    assert math.isclose(forward.y, 0.0, abs_tol=1e-9)
    assert math.isclose(forward.yaw, 0.0, abs_tol=1e-9)
    assert turning is not None
    assert turning.angular_radps < 0.0
