#include <limits>

#include <gtest/gtest.h>

#include "unitree_g1d_driver/driver_core.hpp"

using unitree_g1d_driver::DriverCore;

TEST(DriverCore, StartsStoppedAndRequiresBrakeHeartbeat)
{
  DriverCore core(0.25, 0.25, 0.35, 0.60);
  const auto output = core.step(1.0, true);
  EXPECT_TRUE(output.stop);
  EXPECT_EQ(output.reason, "brake_heartbeat_timeout");
}

TEST(DriverCore, BrakeAlwaysOverridesFreshCommand)
{
  DriverCore core(0.25, 0.25, 0.35, 0.60);
  core.set_command(0.2, 0.3, 1.0);
  core.set_brake(true, 1.0);
  const auto output = core.step(1.1, true);
  EXPECT_TRUE(output.stop);
  EXPECT_DOUBLE_EQ(output.linear_mps, 0.0);
  EXPECT_EQ(output.reason, "brake_requested");
}

TEST(DriverCore, MotionGateAndTimeoutAreFailClosed)
{
  DriverCore core(0.25, 0.25, 0.35, 0.60);
  core.set_command(0.2, 0.3, 1.0);
  core.set_brake(false, 1.0);
  EXPECT_EQ(core.step(1.1, false).reason, "sdk_motion_disabled");
  EXPECT_FALSE(core.step(1.1, true).stop);
  core.set_brake(false, 1.3);
  EXPECT_EQ(core.step(1.3, true).reason, "command_timeout");
}

TEST(DriverCore, ClampsCommandsAndRejectsNonFiniteValues)
{
  DriverCore core(0.25, 0.25, 0.35, 0.60);
  core.set_brake(false, 1.0);
  core.set_command(4.0, -4.0, 1.0);
  auto output = core.step(1.1, true);
  EXPECT_FALSE(output.stop);
  EXPECT_DOUBLE_EQ(output.linear_mps, 0.35);
  EXPECT_DOUBLE_EQ(output.angular_radps, -0.60);

  core.set_command(
    std::numeric_limits<double>::quiet_NaN(), 0.0, 1.2);
  output = core.step(1.2, true);
  EXPECT_TRUE(output.stop);
  EXPECT_EQ(output.reason, "invalid_command");
}

TEST(DriverCore, StaleBrakeHeartbeatStopsEvenWithFreshCommands)
{
  DriverCore core(0.25, 0.25, 0.35, 0.60);
  core.set_brake(false, 1.0);
  core.set_command(0.1, 0.0, 1.3);
  const auto output = core.step(1.3, true);
  EXPECT_TRUE(output.stop);
  EXPECT_EQ(output.reason, "brake_heartbeat_timeout");
}
