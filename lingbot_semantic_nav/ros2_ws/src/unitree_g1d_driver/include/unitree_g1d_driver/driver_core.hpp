#ifndef UNITREE_G1D_DRIVER__DRIVER_CORE_HPP_
#define UNITREE_G1D_DRIVER__DRIVER_CORE_HPP_

#include <string>

namespace unitree_g1d_driver
{

struct DriverCommand
{
  double linear_mps{0.0};
  double angular_radps{0.0};
  bool stop{true};
  std::string reason{"not_initialized"};
};

class DriverCore
{
public:
  DriverCore(
    double command_timeout_s,
    double brake_timeout_s,
    double max_linear_mps,
    double max_angular_radps);

  void set_command(double linear_mps, double angular_radps, double now_s);
  void set_brake(bool active, double now_s);
  DriverCommand step(double now_s, bool motion_enabled) const;

private:
  double command_timeout_s_;
  double brake_timeout_s_;
  double max_linear_mps_;
  double max_angular_radps_;
  double linear_mps_{0.0};
  double angular_radps_{0.0};
  double last_command_s_{0.0};
  double last_brake_s_{0.0};
  bool command_received_{false};
  bool command_valid_{false};
  bool brake_received_{false};
  bool brake_active_{true};
};

}  // namespace unitree_g1d_driver

#endif  // UNITREE_G1D_DRIVER__DRIVER_CORE_HPP_
