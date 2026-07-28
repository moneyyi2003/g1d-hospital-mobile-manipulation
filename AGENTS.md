# 项目长期维护指南

更新时间：2026-07-28（UTC）

## 项目目标

本项目在 NVIDIA Isaac Sim 中加载轮式双臂 G1-D 机器人，结合
MobileManiBench、LingBot-Map 和语义地点库，构建可复现的语言导航与移动操作流程。
当前阶段的主成果是 SimpleRoom、Hospital 和多货架 Warehouse 场景中的中文/英文语言
目标导航；Warehouse 已完成正式 RGB-only 建图、语义地点审核和纯轮地接触路线验收。
后续重点是把 fail-closed ROS 2/Nav2 接口接入实体 G1-D 厂商驱动与硬急停，以及
“导航—停靠—抓取—抬升”的移动操作闭环。

## 新对话恢复顺序

1. 先阅读本文件和根目录 `TODO.md`。
2. 阅读 `CHANGELOG.md` 最近一条记录，并运行 `git status --short --branch`。
3. 根据任务类型阅读 `task.md`、`docs/HOSPITAL_SEMANTIC_NAV.md`、
   `docs/WAREHOUSE_G1D_NAV.md` 或 `MOBILEMANIBENCH_SETUP.md`。
4. 检查 `MobileManiBench` 子模块状态；该目录有独立 Git 历史。
5. 只在确认运行时、资产和输出存在后执行耗时仿真，不要把“文件存在”当成验收成功。

## 当前环境

- 工作区：`/root/autodl-tmp`
- 系统：Ubuntu 22.04.1 LTS，Linux 5.15.0-78-generic，x86_64
- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，97,887 MiB 显存，
  compute capability 12.0
- 驱动：590.44.01；`nvidia-smi` 显示最高 CUDA 13.1
- 当前主 Isaac Sim：
  `/root/autodl-tmp/isaacsim`，版本 `6.0.1-rc.7+release.42383.32955d8d.gl`，
  内置 Python 3.12.13
- MobileManiBench 环境：
  `/root/autodl-tmp/envs/mobilemanibench`，Python 3.10.20，
  PyTorch 2.5.1+cu121；该环境保留早期 Isaac Sim 4.5 / Isaac Lab 兼容链
- LingBot-Map 环境：
  `/root/autodl-tmp/envs/lingbot-map`，Python 3.10.8，
  PyTorch 2.8.0+cu128，用于 Blackwell 上的 RGB-only 建图
- 其他 Conda 环境：`/root/miniconda3/envs/habitat` 和
  `/root/miniconda3/envs/vln`，均为 Python 3.10.20
- Isaac/Omniverse 命令必须设置 `OMNI_KIT_ACCEPT_EULA=YES`
- 2026-07-23 检查时 Isaac Sim WebRTC streaming 和本地 Vite 客户端正在运行；进程不是
  项目状态的一部分，重启会消失。

历史文档中的 RTX 4090、驱动 580.105.08 和 Isaac Sim 4.5 是此前环境记录，不代表当前
主运行时。涉及复现时必须明确使用“主 Isaac Sim 6.0.1”还是
“MobileManiBench Python 3.10 锁定环境”，不可混用其 Python 包。

## 代码和目录职责

- `run_g1d_simple_room_vln.py`：SimpleRoom 语言导航仿真入口。
- `run_g1d_hospital_vln.py`：Hospital 巡检、相机采集和导航入口。
- `run_g1d_warehouse_vln.py`：多货架 Warehouse 场景审计、RGB 巡检和导航入口。
- `simple_room_vln/`：地点解析、地图加载、规划及路径跟随公共逻辑。
- `hospital_vln/`：Hospital 路径、正式地点库及相关测试。
- `warehouse_vln/`：Warehouse bootstrap/正式制品边界、语义地点和 G1-D 轮子约定。
- `g1d_agent/`：VLN/VLA 任务路由、物体—技能交互配置、VLA 启动门控和外部 backend
  接口；没有实时观测或 VLA 时必须 fail-closed。
- `scripts/build_hospital_map.py`：LingBot 推理结果对齐、地图和预览构建。
- `mobilemanibench.sh`：统一命令入口。
- `MobileManiBench/`：上游仓库子模块，包含本项目增加的 G1-D smoke、VLN 和 doctor
  工具；修改后应在子模块内单独提交，再提交根仓库中的子模块指针。
- `lingbot_semantic_nav/`：语义导航、Habitat/ROS 2 集成代码；数据集、输出和第三方源码
  不进入根仓库。
- `Assets/`、`checkpoints/`、`envs/`、`isaacsim/`、`outputs/`：本机大文件或生成物，
  已被 `.gitignore` 排除，不得强制加入 Git。

## 常用运行与验证

```bash
cd /root/autodl-tmp

# 环境诊断
./mobilemanibench.sh doctor

# SimpleRoom 确定性无界面回归
./mobilemanibench.sh simple-room-vln --headless --test --no-camera

# Hospital 正式地图候诊区回归
./mobilemanibench.sh hospital-vln --headless --test --no-camera \
  --command '请带我到候诊区'

# Hospital RGB 巡检和建图
./mobilemanibench.sh hospital-survey --headless --resolution 640x360
./mobilemanibench.sh hospital-map

# 多货架 Warehouse G1-D 确定性回归
./mobilemanibench.sh warehouse-vln --headless --test --no-camera \
  --command '请带我到东侧货架通道'

# Warehouse RGB-only 地图流水线和正式纯轮导航
./mobilemanibench.sh warehouse-map --stage all
./mobilemanibench.sh warehouse-vln-formal --headless --no-camera \
  --command '请带我到东侧货架通道'

# 实体 G1-D ROS 2/Nav2（默认禁止硬件输出）
cd /root/autodl-tmp/lingbot_semantic_nav/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select lingbot_semantic_nav_ros
cd /root/autodl-tmp
./mobilemanibench.sh g1d-real-nav

# 有桌面/VNC 时运行第三人称演示
./mobilemanibench.sh hospital-demo

# 轻量 Python 测试（不启动 Isaac）
/root/autodl-tmp/isaacsim/python.sh -m unittest discover \
  -s /root/autodl-tmp/hospital_vln/tests -v
```

详细参数以 `./mobilemanibench.sh help` 和对应脚本 `--help` 为准。大型仿真执行前先检查
GPU 上是否已有 Isaac 进程，避免同时启动多个 Kit 实例。

## 已验证基线

- SimpleRoom：指令“请带我到沙发旁边”，`stable_assisted` 成功，657 帧，
  路径 2.733 m，位置误差 0.119 m。
- Hospital：完成 G1-D RGB 巡检、LingBot-Map RGB-only 推理、米制对齐、ROS occupancy
  地图和正式地点库；已审核地点为 `reception`、`waiting_area`。
- Hospital 候诊区：正式地图、`stable_assisted` 模式成功，1092 帧，
  路径 7.376 m，位置误差 0.119 m，航向误差 0.117 rad。
- Warehouse：完成 188 帧 G1-D RGB 巡检、LingBot RGB-only 推理、SAM3.1 语义投影、
  372 x 617 正式 occupancy 和地点审核；开放 `east_shelf_aisle`、
  `west_shelf_aisle`，`loading_zone` 因覆盖不足保持拒绝。
- Warehouse 东侧货架正式定向路线：`--wheel-physics-only` 连续三次成功；每次
  10,306 帧，规划 32.538 m、物理路程 32.516 m，位置误差 0.190 m，航向误差
  0.051 rad，最大 roll/pitch 0.026/0.147 rad，2 秒制动漂移 0.0086 m。
- 物理 G1-D ROS 2/Nav2 接口已完成本机 fail-closed 联调；实体机器人尚未接入厂商驱动、
  网络、硬急停回路和真实 `/scan`，不得把仿真结果表述为真机运动验收。

以上数值来自本机 `outputs/` 中的运行摘要；输出不进 Git，重跑可能覆盖它们。

## 重要约束

- 默认 `stable_assisted` 会写轮速，同时用确定性平面位姿更新保证高层链路稳定；只有
  `--wheel-physics-only` 才验证真实轮地接触。
- 正式导航必须使用经过审核的地点 ID 和 docking pose；语言模型只选择地点，不直接
  生成任意坐标。bootstrap 几何仅用于联调。
- LingBot 正式输入只有 RGB；Isaac 相机位姿仅可在推理后用于米制 Sim(3) 对齐或明确标注
  的 pose-anchored 融合，不得冒充纯视觉全局建图。
- G1-D 与 MobileManiBench 自带 G1 的 link、joint、底盘和手部结构不同，不能只替换 USD。
- 不提交模型、数据集、仿真安装、虚拟环境、USD/STL/GLB、生成地图、录屏或日志。
- 不删除或覆盖用户已有资产与输出。不要用破坏性 Git 命令清理工作区。
- 不在文档或提交中保存 token、`.env`、私有 npm 配置或其他凭据。

## 每个重要步骤的维护协议

完成一个可独立验收的重要步骤时，必须按以下顺序收尾：

1. 运行与改动风险相称的测试，记录实际命令和结果。
2. 更新 `TODO.md`：移动完成项、补充失败证据、明确唯一的近期下一步。
3. 更新 `CHANGELOG.md`：写日期、变更内容、验证结果和已知限制。
4. 运行 `git status --short` 与 `git diff --check`，确认无大文件和凭据。
5. 创建聚焦的 Git commit；若改动 `MobileManiBench/`，先在子模块提交，再提交根仓库指针。

不要为微小的日志刷新制造提交；一个 commit 应对应一个能够解释和恢复的工程状态。
