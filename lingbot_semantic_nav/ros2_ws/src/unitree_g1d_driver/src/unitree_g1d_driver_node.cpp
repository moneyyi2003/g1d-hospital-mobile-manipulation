#include <net/if.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/agv/g1_agv_client.hpp>

#include "unitree_g1d_driver/driver_core.hpp"

namespace unitree_g1d_driver
{

class UnitreeG1DDriverNode : public rclcpp::Node
{
public:
  UnitreeG1DDriverNode()
  : Node("unitree_g1d_driver"),
    connect_sdk_(declare_parameter<bool>("connect_sdk", false)),
    allow_motion_(declare_parameter<bool>("allow_sdk_motion", false)),
    network_interface_(declare_parameter<std::string>("network_interface", "")),
    sdk_timeout_s_(declare_parameter<double>("sdk_timeout_s", 0.20)),
    ready_timeout_s_(declare_parameter<double>("ready_timeout_s", 0.50)),
    command_rate_hz_(declare_parameter<double>("command_rate_hz", 20.0)),
    core_(
      declare_parameter<double>("command_timeout_s", 0.25),
      declare_parameter<double>("brake_timeout_s", 0.25),
      declare_parameter<double>("max_linear_mps", 0.35),
      declare_parameter<double>("max_angular_radps", 0.60))
  {
    if (!std::isfinite(sdk_timeout_s_) || sdk_timeout_s_ <= 0.0) {
      throw std::invalid_argument("sdk_timeout_s must be finite and positive");
    }
    if (!std::isfinite(ready_timeout_s_) || ready_timeout_s_ <= 0.0) {
      throw std::invalid_argument("ready_timeout_s must be finite and positive");
    }
    if (!std::isfinite(command_rate_hz_) || command_rate_hz_ <= 0.0) {
      throw std::invalid_argument("command_rate_hz must be finite and positive");
    }
    if (allow_motion_ && !connect_sdk_) {
      throw std::invalid_argument(
              "allow_sdk_motion=True requires connect_sdk=True");
    }

    const auto command_topic =
      declare_parameter<std::string>("command_topic", "/g1d/hardware/cmd_vel");
    const auto brake_topic =
      declare_parameter<std::string>("brake_topic", "/g1d/hardware/brake");
    const auto ready_topic =
      declare_parameter<std::string>(
      "driver_ready_topic", "/g1d/hardware/driver_ready");
    const auto status_topic =
      declare_parameter<std::string>(
      "status_topic", "/g1d/hardware/sdk_status");

    ready_publisher_ = create_publisher<std_msgs::msg::Bool>(ready_topic, 10);
    status_publisher_ = create_publisher<std_msgs::msg::String>(status_topic, 10);
    diagnostics_publisher_ =
      create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    command_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      command_topic, 10,
      [this](const geometry_msgs::msg::Twist::SharedPtr message) {
        core_.set_command(
          message->linear.x, message->angular.z, now_seconds());
      });
    brake_subscription_ = create_subscription<std_msgs::msg::Bool>(
      brake_topic, 10,
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        core_.set_brake(message->data, now_seconds());
      });

    if (connect_sdk_) {
      initialize_sdk();
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Unitree SDK transport is disconnected; no DDS traffic or motion output is possible");
    }
    if (!allow_motion_) {
      RCLCPP_WARN(
        get_logger(),
        "Unitree SDK non-zero motion is disabled; driver_ready will remain false");
    }

    const auto period = std::chrono::duration<double>(1.0 / command_rate_hz_);
    command_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&UnitreeG1DDriverNode::command_tick, this));
  }

  ~UnitreeG1DDriverNode() override
  {
    if (sdk_initialized_) {
      const int32_t result = agv_client_->Move(0.0F, 0.0F, 0.0F);
      if (result != 0) {
        RCLCPP_ERROR(
          get_logger(), "Final Unitree zero-velocity command failed: %d", result);
      }
    }
  }

private:
  double now_seconds()
  {
    return get_clock()->now().seconds();
  }

  void initialize_sdk()
  {
    if (network_interface_.empty()) {
      throw std::invalid_argument(
              "network_interface is required when connect_sdk=True");
    }
    if (if_nametoindex(network_interface_.c_str()) == 0U) {
      throw std::invalid_argument(
              "network_interface does not exist: " + network_interface_);
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    agv_client_ = std::make_unique<unitree::robot::g1::AgvClient>();
    agv_client_->SetTimeout(static_cast<float>(sdk_timeout_s_));
    agv_client_->Init();
    sdk_initialized_ = true;
    RCLCPP_WARN(
      get_logger(),
      "Unitree SDK DDS initialized on '%s'; startup remains zero velocity until all gates clear",
      network_interface_.c_str());
  }

  void command_tick()
  {
    const double now_s = now_seconds();
    const DriverCommand command = core_.step(now_s, allow_motion_);
    bool rpc_ok = false;
    if (sdk_initialized_) {
      const int32_t result = agv_client_->Move(
        static_cast<float>(command.linear_mps),
        0.0F,
        static_cast<float>(command.angular_radps));
      last_rpc_result_ = result;
      if (result == 0) {
        rpc_ok = true;
        last_rpc_success_s_ = now_s;
      }
    }

    const bool ready =
      sdk_initialized_ && allow_motion_ && rpc_ok &&
      now_s - last_rpc_success_s_ <= ready_timeout_s_;
    ready_publisher_->publish(std_msgs::msg::Bool().set__data(ready));
    publish_status(now_s, command, ready);
  }

  void publish_status(
    const double now_s,
    const DriverCommand & command,
    const bool ready)
  {
    std::ostringstream json;
    json << std::boolalpha
         << "{\"timestamp\":" << now_s
         << ",\"connect_sdk\":" << connect_sdk_
         << ",\"sdk_initialized\":" << sdk_initialized_
         << ",\"allow_sdk_motion\":" << allow_motion_
         << ",\"driver_ready\":" << ready
         << ",\"last_rpc_result\":" << last_rpc_result_
         << ",\"command_reason\":\"" << command.reason << "\""
         << ",\"linear_mps\":" << command.linear_mps
         << ",\"angular_radps\":" << command.angular_radps
         << ",\"brake_semantics\":\"Move(0,0,0)_only\""
         << ",\"wheel_feedback_available\":false"
         << ",\"hardware_estop_available\":false}";
    status_publisher_->publish(std_msgs::msg::String().set__data(json.str()));

    diagnostic_msgs::msg::DiagnosticStatus diagnostic;
    diagnostic.name = "unitree_g1d_sdk_driver";
    diagnostic.hardware_id = "g1_d";
    if (connect_sdk_ && last_rpc_result_ != 0) {
      diagnostic.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      diagnostic.message = "unitree_sdk_rpc_failed";
    } else if (!ready) {
      diagnostic.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      diagnostic.message = command.reason;
    } else {
      diagnostic.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      diagnostic.message = "unitree_sdk_ready";
    }
    diagnostic.values = {
      diagnostic_msgs::msg::KeyValue().set__key("network_interface").set__value(
        network_interface_),
      diagnostic_msgs::msg::KeyValue().set__key("connect_sdk").set__value(
        connect_sdk_ ? "true" : "false"),
      diagnostic_msgs::msg::KeyValue().set__key("allow_sdk_motion").set__value(
        allow_motion_ ? "true" : "false"),
      diagnostic_msgs::msg::KeyValue().set__key("driver_ready").set__value(
        ready ? "true" : "false"),
      diagnostic_msgs::msg::KeyValue().set__key("brake_semantics").set__value(
        "zero_velocity_only"),
      diagnostic_msgs::msg::KeyValue().set__key("wheel_feedback").set__value(
        "unavailable"),
      diagnostic_msgs::msg::KeyValue().set__key("hardware_estop").set__value(
        "external_required")
    };
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = get_clock()->now();
    array.status.push_back(diagnostic);
    diagnostics_publisher_->publish(array);
  }

  bool connect_sdk_;
  bool allow_motion_;
  std::string network_interface_;
  double sdk_timeout_s_;
  double ready_timeout_s_;
  double command_rate_hz_;
  DriverCore core_;
  bool sdk_initialized_{false};
  int32_t last_rpc_result_{-1};
  double last_rpc_success_s_{0.0};
  std::unique_ptr<unitree::robot::g1::AgvClient> agv_client_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr brake_subscription_;
  rclcpp::TimerBase::SharedPtr command_timer_;
};

}  // namespace unitree_g1d_driver

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<unitree_g1d_driver::UnitreeG1DDriverNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      rclcpp::get_logger("unitree_g1d_driver"), "Driver initialization failed: %s",
      error.what());
    rclcpp::shutdown();
    return 2;
  }
  rclcpp::shutdown();
  return 0;
}
