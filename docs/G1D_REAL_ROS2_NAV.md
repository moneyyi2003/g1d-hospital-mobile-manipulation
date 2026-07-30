# 物理 G1-D 的 ROS 2 / Nav2 安全接入

更新时间：2026-07-30（UTC）

## 当前结论

仓库已经把本机 `/root/autodl-tmp/unitree_sdk2` 中的官方 G1-D
`g1::AgvClient` 接入 ROS 2/Nav2。新增 `unitree_g1d_driver` 把安全桥输出的
`Twist` 转成 `AgvClient::Move(vx, 0, wz)`，并以独立开关控制 DDS 初始化和非零速度。

默认、以及当前实际状态，仍不会向机器人发送驱动命令。SDK 当前没有给出已确认的左右轮
编码器 topic、独立机械制动 API 或硬急停 API，本机也没有机器人网络、硬急停回路和真实
`/scan`。这些条件补齐并完成隔离场验收前，三个输出门必须全部保持关闭。

已实现并通过零输出集成测试的接口：

- Nav2 `/cmd_vel` 最终限幅、加速度限制和 0.25 s 看门狗；
- `Left_Wheel_Joint` / `Right_Wheel_Joint` 编码器里程计；
- `/odom` 和 `odom -> AGV_link` TF；
- 软件制动、锁存急停、clear-estop、arm 和 disarm；
- 厂商驱动 ready、硬急停输入心跳、轮反馈时效检查和诊断状态；
- 正式 LingBot occupancy map、AMCL、Nav2 RPP、地点库和语言目标节点；
- 审核 `approach_pose -> docking pose` 的双阶段 Nav2 目标提交与 0.20 m/rad 到达复核；
- 启动时“急停锁存 + 未使能”，且 `allow_hardware_output=False` 时 arm 必定失败。
- Unitree SDK2 G1-D AGV 命令适配、SDK RPC 状态、二次命令/制动心跳看门狗和诊断；
- SDK、ROS 2 安全桥和 Nav2 在同一 bringup 中启动，默认 DDS 不连接且非零速度禁止。

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
                                   v           v
                             unitree_g1d_driver
                                   |
                                   v
                       Unitree AgvClient::Move(vx, 0, wz)
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

## Unitree SDK 适配状态

当前命令侧已经实现，反馈和实体安全侧仍待真机确认：

| 方向 | Topic/API | 状态 |
|---|---|---|---|
| 项目 -> SDK | `/g1d/hardware/cmd_vel` -> `AgvClient::Move` | 已实现；`vy` 固定为 0 |
| 项目 -> SDK | `/g1d/hardware/brake` -> `Move(0,0,0)` | 已实现零速度优先级；不等于机械制动 |
| SDK -> 项目 | `/g1d/hardware/driver_ready` | 已实现；仅 SDK 非零输出已允许且最近 RPC 成功时为真 |
| SDK -> 项目 | `/g1d/hardware/sdk_status` | 已实现；报告开关、RPC、命令原因及缺失能力 |
| 驱动 -> 项目 | `/joint_states` 左右轮 position/velocity | 未实现；SDK 示例没有确认轮编码器 topic |
| 硬件 -> 项目 | `/g1d/hardware/estop` | 未实现；必须接实体独立硬急停回路 |
| 传感器 -> Nav2 | `/scan` 与对应 TF | 未实现；必须接真实 LiDAR 或标定深度传感器 |

SDK 官方上限为线速度 `1.5 m/s`、角速度 `0.6 rad/s`；项目当前进一步限为
`0.35 m/s` 和 `0.6 rad/s`。SDK 适配节点还要求 `/g1d/hardware/brake` 心跳新鲜，
制动心跳、速度命令或 RPC 任一超时都会发送零速度。

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
colcon build --symlink-install \
  --packages-select unitree_g1d_driver lingbot_semantic_nav_ros \
  --cmake-args \
  -DUNITREE_SDK2_ROOT=/root/autodl-tmp/unitree_sdk2 \
  -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

cd /root/autodl-tmp
./mobilemanibench.sh g1d-real-nav
```

构建命令必须从 `lingbot_semantic_nav/ros2_ws` 执行，不能从项目根目录让 `colcon`
扫描 Isaac/Conda 第三方目录。

默认命令加载正式 Warehouse 地图和地点库，并启动 Unitree 适配节点，但保持：

```text
allow_hardware_output=False
unitree_connect_sdk=False
unitree_allow_sdk_motion=False
```

因此不会初始化 Unitree DDS，也不会产生 SDK 运动调用；`driver_ready=False`，arm 服务
返回失败。三个开关是独立门，不能以打开其中一个代替实体反馈和急停验收。

重建后的家庭正式图使用：

```bash
./mobilemanibench.sh g1d-home-real-nav
```

该入口只更换为家庭 `map.yaml` 和审核地点库，底盘、TF、里程计、制动和急停合同不变，
同样强制三个输出门关闭。SDK 适配完成不等于实体安全条件完成，项目不提供“跳过检查
直接开电机”的命令。

Nav2/AMCL 还要求真实、带正确时间戳和 TF 的 `/scan`。可由 2D LiDAR 直接提供，或由
经过标定和时间同步的 RGB-D/depth-to-scan 节点提供；只有 RGB 图像而没有可定位的深度/
激光观测，不能宣称 AMCL 已完成物理定位。LingBot RGB-only 地图负责离线建图，不会自动
替代在线定位传感器。

## 真机使能前的强制验收

1. 在机器人旁的工控机确认机器人侧网卡名、IP/固件和 SDK API 版本；先以只发零速度的
   `unitree_connect_sdk=True` 验证 RPC，不能在当前远程服务器凭空假设网络可达。
2. 确认实体硬急停能切断驱动力，软件急停不能代替硬急停。
3. 架空轮子验证左右轮方向、编码器方向、速度限幅和 brake 优先级。
4. 落地低速验证直行、原地转向、松手制动和 0.25 s 通信丢失制动。
5. 用实测 `/scan`、TF 和 AMCL 验证定位，不使用静态假 TF。
6. 在架空轮和实体急停监护下依次开启
   `unitree_connect_sdk`、`unitree_allow_sdk_motion` 和 `allow_hardware_output`，
   clear 后人工 arm；先把最大线速度降至 `0.10 m/s`。
7. 连续三次完成短路线和急停测试后，才逐步开放正式 Warehouse 路线。

当前没有完成上述实体机验收，因此本文描述的是已经实现并测试的接口，不是“物理机器人
已经运动”的声明。
