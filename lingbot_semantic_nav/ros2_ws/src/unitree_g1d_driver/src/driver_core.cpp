#include "unitree_g1d_driver/driver_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace unitree_g1d_driver
{

DriverCore::DriverCore(
  const double command_timeout_s,
  const double brake_timeout_s,
  const double max_linear_mps,
  const double max_angular_radps)
: command_timeout_s_(command_timeout_s),
  brake_timeout_s_(brake_timeout_s),
  max_linear_mps_(max_linear_mps),
  max_angular_radps_(max_angular_radps)
{
  if (
    !std::isfinite(command_timeout_s_) || command_timeout_s_ <= 0.0 ||
    !std::isfinite(brake_timeout_s_) || brake_timeout_s_ <= 0.0 ||
    !std::isfinite(max_linear_mps_) || max_linear_mps_ <= 0.0 ||
    !std::isfinite(max_angular_radps_) || max_angular_radps_ <= 0.0)
  {
    throw std::invalid_argument("driver limits and timeouts must be finite and positive");
  }
}

void DriverCore::set_command(
  const double linear_mps,
  const double angular_radps,
  const double now_s)
{
  command_received_ = true;
  last_command_s_ = now_s;
  command_valid_ =
    std::isfinite(linear_mps) && std::isfinite(angular_radps) && std::isfinite(now_s);
  if (!command_valid_) {
    linear_mps_ = 0.0;
    angular_radps_ = 0.0;
    return;
  }
  linear_mps_ = std::clamp(linear_mps, -max_linear_mps_, max_linear_mps_);
  angular_radps_ = std::clamp(angular_radps, -max_angular_radps_, max_angular_radps_);
}

void DriverCore::set_brake(const bool active, const double now_s)
{
  if (!std::isfinite(now_s)) {
    brake_received_ = false;
    brake_active_ = true;
    return;
  }
  brake_received_ = true;
  brake_active_ = active;
  last_brake_s_ = now_s;
}

DriverCommand DriverCore::step(const double now_s, const bool motion_enabled) const
{
  DriverCommand output;
  if (!std::isfinite(now_s)) {
    output.reason = "invalid_clock";
    return output;
  }
  if (!brake_received_ || now_s < last_brake_s_ ||
    now_s - last_brake_s_ > brake_timeout_s_)
  {
    output.reason = "brake_heartbeat_timeout";
    return output;
  }
  if (brake_active_) {
    output.reason = "brake_requested";
    return output;
  }
  if (!motion_enabled) {
    output.reason = "sdk_motion_disabled";
    return output;
  }
  if (!command_received_) {
    output.reason = "command_not_received";
    return output;
  }
  if (!command_valid_) {
    output.reason = "invalid_command";
    return output;
  }
  if (now_s < last_command_s_ || now_s - last_command_s_ > command_timeout_s_) {
    output.reason = "command_timeout";
    return output;
  }
  output.linear_mps = linear_mps_;
  output.angular_radps = angular_radps_;
  output.stop = false;
  output.reason = "command_active";
  return output;
}

}  // namespace unitree_g1d_driver
