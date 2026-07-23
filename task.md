# Isaac Sim + G1_D + MobileManiBench 实施清单

更新时间：2026-07-23（UTC）

## 最终目标

在 Isaac Sim 4.5 / Isaac Lab 中加载 `g1_d_description` 的 G1_D 机器人和
MobileManiBench 场景/物体，至少完成：

1. 一个可复现的简易 VLN（语言指令到目标点导航）任务；
2. 一个可复现的简易移动操作任务（优先为导航到物体后抓取/抬起）；
3. 无界面 smoke test、成功判据和统一启动命令。

## Hospital VLN 专项进度（2026-07-23 UTC）

- [x] G1-D 加载 NVIDIA Hospital 前台/候诊区 ROI。
- [x] 头部 RGB 巡检、LingBot-Map RGB-only 推理、米制对齐和 ROS occupancy 地图。
- [x] 生成经审核的正式地点库，当前开放 `reception` 和 `waiting_area`。
- [x] 中文指令“请带我到候诊区”解析、规划和确定性导航成功。
- [x] Hospital dashboard 接入 DeepSeek 模糊地点解析；“带我去找个能坐着等医生的地方”
  在不出现地点名时选择正式 `waiting_area` 并完成真实 Isaac 导航。
- [x] 端到端命令 `hospital-vln --headless --test --no-camera` 返回 0；路径
  7.376 m，1092 帧，终点位置误差 0.119 m，航向误差 0.117 rad。
- [ ] 纯轮地接触物理模式（`--wheel-physics-only`）的摩擦/驱动调参与稳定回归；
  这是下一阶段，不影响当前 `stable_assisted` Hospital 语义导航 MVP 验收。

状态说明：`[x]` 已执行并验证；`[~]` 已有部分成果但尚未通过最终验收；`[ ]` 等待完成。

## 已完成并验证

- [x] MobileManiBench 上游源码位于 `/root/autodl-tmp/MobileManiBench`。
- [x] 独立 Python 3.10 环境位于 `/root/autodl-tmp/envs/mobilemanibench`。
- [x] Isaac Sim `4.5.0.0`、PyTorch `2.5.1+cu121` 可用。
- [x] RTX 4090（24 GB）及 NVIDIA 驱动 `580.105.08` 可被 Isaac Sim 识别。
- [x] MobileManiBench 已 editable 安装为 `unimanip 1.0`。
- [x] 已将 Isaac Lab 核心包、assets、tasks、RL、mimic 从当前源码 editable 安装进专用环境。
- [x] G1_D URDF 和全部本地 STL 网格齐全；URDF 可解析为 41 个 link、40 个 joint（30 revolute、2 continuous、2 prismatic、6 fixed）。
- [x] G1_D 已转换为分层 USD：`Assets/g1_d/g1_d.usd` 及 `configuration/` 下 base/physics/sensor 层。
- [x] 转换结果不是空占位文件：主 USD 为 crate 格式，base 层约 37 MB。
- [x] 已有统一入口 `mobilemanibench.sh` 和官方环境零动作测试 `unimanip/rsl_ppo/smoke_env.py`。

## 正在进行 / 部分完成

- [~] MobileManiBench 官方资产下载。
  - 当前仓库里的旧 `Assets.zip` 是不完整的稀疏文件，无法通过 ZIP 校验。
  - 上游当前文件长度为 `8,507,213,644` 字节，与旧下载记录的长度不同。
  - 正在重新下载为 `Assets/Assets.zip.download`；下载完成后必须校验并解压，不能把“文件存在”视为完成。
- [~] Python 依赖完整性。
  - 运行任务所需的 Isaac Lab 核心导入已恢复。
  - 可选的 Pink IK / dex-retargeting 等完整依赖仍需固定兼容版本并通过 `pip check`；基础 smoke 不应依赖这些可选模块。
- [~] G1_D Isaac 导入。
  - URDF -> USD 已成功。
  - 尚需在物理仿真中实例化 Articulation，确认 root、body/joint 名称、默认姿态、碰撞和关节驱动均有效。

## 待完成（按执行顺序）

### P0：恢复可复现的基础环境

- [ ] 完成当前官方 `Assets.zip` 下载，校验 ZIP，解压到 `/root/autodl-tmp/Assets`。
  - 验收：存在 `Assets/g1_robot/g1_robot.usd`、至少一个 YCB USD 和至少一个 room USD。
- [ ] 运行官方 MobileManiBench G1 + YCB 无界面 smoke。
  - 命令：`./mobilemanibench.sh smoke --headless --device cuda:0 --steps 8`
  - 验收：环境 reset 并完成 8 个仿真 step，进程返回 0。
- [ ] 增加 `doctor` 命令，对 Python 包、GPU、资产、G1_D USD 和关键文件给出明确 PASS/FAIL。

### P1：让自定义 G1_D 在 Isaac Lab 中可控

- [ ] 增加 G1_D 独立 smoke/inspect 脚本，在 headless 模式加载 USD 并打印实际 body/joint 列表。
  - 验收：仿真至少运行 20 step，无 missing prim / invalid articulation 错误。
- [ ] 建立 G1_D `ArticulationCfg`。
  - 底盘：`Left_Wheel_Joint`、`Right_Wheel_Joint` 速度控制。
  - 升降：`LZ_mt_Joint`、`LZ_it_Joint` 位置控制。
  - 上身/双臂：肩、肘、腕位置控制。
  - 手指：拇指/食指/中指关节位置控制。
  - 验收：底盘能前进/转向，右臂和右手能执行小幅关节动作且状态有限、无 NaN。
- [ ] 为 G1_D 增加头部 RGB-D 相机（必要时增加腕部相机）。
  - 验收：headless 模式能取得非空 RGB 和 depth tensor。

### P1：简易 VLN

- [ ] 在 MobileManiBench/Isaac Lab 内新增最小 VLN 任务和 Gym 注册。
  - 输入：中英文模板指令，例如“前往红色目标 / go to the red target”。
  - 观测：机器人基座位姿、目标相对方向/距离；相机观测作为可选项保留。
  - 动作：左右轮速度或线速度/角速度。
  - 成功：基座到目标点距离小于阈值并停止。
- [ ] 提供确定性的 scripted baseline（不要求先训练 RL/VLA）。
  - 验收：固定种子下至少 3 个目标点全部成功，并输出路径长度、最终距离、成功率。
- [ ] 增加 VLN headless smoke 命令。

### P1/P2：简易抓取与移动操作

- [ ] 先用简单立方体完成右手靠近、闭合、抬起的 scripted grasp。
  - 验收：物体离地高度增加至少 5 cm，并保持若干仿真 step。
- [ ] 再替换为一个 MobileManiBench YCB 物体。
  - 验收：YCB 物体可生成、接触、抓起；记录成功/失败和最终高度。
- [ ] 组合“语言目标 -> 导航 -> 停靠 -> 抓取”的单回合演示。
  - 验收：一条命令可在 headless 模式跑完，阶段日志包含 NAVIGATE、ALIGN、GRASP、LIFT、SUCCESS/FAIL。

### P2：场景与质量收尾

- [ ] 把最小任务从简单地面迁移到至少一个 MobileManiBench room 场景，校正出生点和碰撞。
- [ ] 检查 G1_D 自碰撞、轮地摩擦、质量/惯量、关节限位和驱动增益。
- [ ] 增加运行文档、固定随机种子、错误诊断和关键回归测试。
- [ ] 如需学习策略，再单独确定训练路线（RL、VLN policy 或 VLA 权重）；这不是 scripted MVP 的前置条件。

## 当前已知风险

- MobileManiBench 自带的 G1 与 G1_D 的 link/joint、底盘和夹爪结构完全不同，不能只替换 USD 路径。
- G1_D 是轮式底盘 + 多指手；可靠抓取需要实际关节映射、接触参数及可能的 IK，不应沿用官方平行夹爪的单关节动作空间。
- 官方资产约 8.5 GB；下载未完成前无法对官方 room/YCB 路径做最终 smoke。
- “能打开 USD”不等于“机器人任务可用”；必须以 Articulation 实例化、动作响应和任务成功判据作为验收。
