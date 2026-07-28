# 物理 G1-D 的 ROS 2 / Nav2 安全接入

更新时间：2026-07-28（UTC）

## 当前结论

仓库已经提供物理 G1-D 所需的 ROS 2 接口层和 Nav2 bringup，但默认、以及当前实际状态，
都不会向机器人发送驱动命令。原因不是软件链缺失，而是本机尚无已确认的 G1-D 厂商底盘
驱动、机器人网络配置、硬急停回路和真实传感器数据。把这些外部前置条件补齐并完成隔离场
验收前，`allow_hardware_output` 必须保持 `False`。

已实现并通过零输出集成测试的接口：

- Nav2 `/cmd_vel` 最终限幅、加速度限制和 0.25 s 看门狗；
- `Left_Wheel_Joint` / `Right_Wheel_Joint` 编码器里程计；
- `/odom` 和 `odom -> AGV_link` TF；
- 软件制动、锁存急停、clear-estop、arm 和 disarm；
- 厂商驱动 ready、硬急停输入心跳、轮反馈时效检查和诊断状态；
- 正式 LingBot occupancy map、AMCL、Nav2 RPP、地点库和语言目标节点；
- 审核 `approach_pose -> docking pose` 的双阶段 Nav2 目标提交与 0.20 m/rad 到达复核；
- 启动时“急停锁存 + 未使能”，且 `allow_hardware_output=False` 时 arm 必定失败。

## ROS 图和坐标系

```text
正式 map.yaml
    |
    v
map_server -> AMCL -- map -> odom
                         |
轮编码器 -> g1d_base_bridge -- odom -> AGV_link
                         |
                       /odom

语言指令 -> language_goal_node -> Nav2 -> /cmd_vel
                                          |
                                          v
                                  g1d_base_bridge
                                   |           |
                             safe_cmd_vel    brake/e-stop
                                   |           |
                                   +-- 仅在显式使能后 --> 厂商驱动适配器
```

物理 G1-D URDF 的真实根 link 是 `AGV_link`，因此 Nav2、AMCL 和到达验证都使用
`AGV_link`，不伪造 `base_link`。完整 TF 应为：

```text
map -> odom -> AGV_link -> G1-D 各关节/link
```

`map -> odom` 由 AMCL 发布；`odom -> AGV_link` 只能由轮编码器里程计发布；
`robot_state_publisher` 消费同一份 `/joint_states` 发布机器人内部 TF。

正式 Warehouse 地点库会在选中的 docking candidate 内保存 `approach_pose`。语言目标
节点对 `ARRIVE` 任务先提交预停靠目标，验证位置和朝向后再提交最终 docking pose；
`PASS` 路由不注入预停靠。这样仿真 runner 与物理 Nav2 不会在末段采用两套停靠策略。

## 厂商驱动适配器契约

当前项目没有 G1-D 厂商驱动，后续适配器必须实现以下 topic：

| 方向 | Topic | 类型 | 要求 |
|---|---|---|---|
| 驱动 -> 项目 | `/joint_states` | `sensor_msgs/JointState` | 必含左右轮 position，建议含 velocity |
| 驱动 -> 项目 | `/g1d/hardware/driver_ready` | `std_msgs/Bool` | 连续心跳，不是只发一次 |
| 硬件 -> 项目 | `/g1d/hardware/estop` | `std_msgs/Bool` | 持续心跳；`True` 立即锁存，`False` 不自动复位 |
| 项目 -> 驱动 | `/g1d/hardware/cmd_vel` | `geometry_msgs/Twist` | 厂商适配器转换为底盘协议 |
| 项目 -> 驱动 | `/g1d/hardware/brake` | `std_msgs/Bool` | `True` 必须优先制动并忽略速度 |

左右轮轴方向相反：左编码器符号 `+1`，右编码器符号 `-1`；轮半径
`0.0848 m`，轮距 `0.4062 m`。厂商适配器若已经把两轮统一为前进正方向，应把
`right_encoder_sign` 改为 `+1`，不得重复反号。

## 安全状态机

启动后必须依次满足：

1. 硬急停输入持续发送释放心跳，且实体回路确实为释放状态；
2. 厂商驱动持续发布 ready 心跳；
3. 左右轮反馈持续且时间未过期；
4. 调用 `/g1d/safety/clear_estop`；
5. 现场检查完成后调用 `/g1d/safety/arm`；
6. Nav2 命令仍要经过速度、加速度和命令看门狗。

可用服务：

```text
/g1d/safety/arm
/g1d/safety/disarm
/g1d/safety/brake_now
/g1d/safety/estop
/g1d/safety/clear_estop
```

任一急停、硬急停输入心跳超时、驱动心跳超时、轮反馈超时或非有限数都会输出零速度并
制动。急停是锁存的，硬件输入恢复为 `False` 后仍需显式 clear；clear 不会自动 arm。

## 构建与默认 dry-run

```bash
cd /root/autodl-tmp/lingbot_semantic_nav/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select lingbot_semantic_nav_ros --symlink-install
source install/setup.bash

cd /root/autodl-tmp
./mobilemanibench.sh g1d-real-nav
```

默认命令加载正式 Warehouse 地图和地点库，但保持
`allow_hardware_output:=False`。此时即使发布 `/cmd_vel`，`safe_cmd_vel` 也保持零，
`/g1d/safety/brake=True`，arm 服务返回失败。

Nav2/AMCL 还要求真实、带正确时间戳和 TF 的 `/scan`。可由 2D LiDAR 直接提供，或由
经过标定和时间同步的 RGB-D/depth-to-scan 节点提供；只有 RGB 图像而没有可定位的深度/
激光观测，不能宣称 AMCL 已完成物理定位。LingBot RGB-only 地图负责离线建图，不会自动
替代在线定位传感器。

## 真机使能前的强制验收

1. 核对机器人网络、驱动固件、速度单位和符号，不在仓库文档保存凭据。
2. 确认实体硬急停能切断驱动力，软件急停不能代替硬急停。
3. 架空轮子验证左右轮方向、编码器方向、速度限幅和 brake 优先级。
4. 落地低速验证直行、原地转向、松手制动和 0.25 s 通信丢失制动。
5. 用实测 `/scan`、TF 和 AMCL 验证定位，不使用静态假 TF。
6. 在隔离场把 `allow_hardware_output:=True`，clear 后人工 arm；先把最大线速度降至
   `0.10 m/s`。
7. 连续三次完成短路线和急停测试后，才逐步开放正式 Warehouse 路线。

当前没有完成上述实体机验收，因此本文描述的是已经实现并测试的接口，不是“物理机器人
已经运动”的声明。
