# MobileManiBench / Isaac Sim 本机入口

更新时间：2026-07-28（UTC）

当前主导航仿真使用 `/root/autodl-tmp/isaacsim` 的 Isaac Sim 6.0.1 和内置
Python 3.12；`/root/autodl-tmp/envs/mobilemanibench` 的 Python 3.10 环境只保留上游
MobileManiBench/Isaac Lab 旧依赖链。以下命令由 `mobilemanibench.sh` 明确选择运行时，
不要在两个 Python 中交叉导入 Isaac 包。

## 目录

- 官方源码：`/root/autodl-tmp/MobileManiBench`
- Python 环境：`/root/autodl-tmp/envs/mobilemanibench`
- 官方运行资产：`/root/autodl-tmp/Assets`
- 自定义机器人：`/root/autodl-tmp/g1_d_description/g1_d.urdf`
- 统一入口：`/root/autodl-tmp/mobilemanibench.sh`

## 验证官方 MobileManiBench 环境

运行一个 G1 + YCB 物体的无界面环境，复位后执行若干个零动作：

```bash
cd /root/autodl-tmp
./mobilemanibench.sh smoke --headless --device cuda:0 --steps 8
```

## 转换并加载自定义 G1_D

先将 URDF 转换为 USD：

```bash
cd /root/autodl-tmp
./mobilemanibench.sh convert-urdf --headless
```

输出文件是 `/root/autodl-tmp/Assets/g1_d/g1_d.usd`。启动有图形桌面的
Isaac Sim 后，可通过 `File -> Open` 打开该 USD：

```bash
cd /root/autodl-tmp
./mobilemanibench.sh isaacsim
```

`G1_D` 与 MobileManiBench 自带 G1 的 link/joint 命名和执行器结构不同，不能仅替换
`Assets/g1_robot/g1_robot.usd`。后续接入任务时，需要为它单独配置
`ArticulationCfg`、关节映射、末端执行器 link、头部/腕部相机和动作空间。

## 官方训练入口示例

```bash
cd /root/autodl-tmp/MobileManiBench/unimanip/rsl_ppo
/root/autodl-tmp/envs/mobilemanibench/bin/python train.py \
  --headless --device cuda:0 --num_envs 64 \
  --task Isaac-G1-Robot-Direct-v0 \
  --config train_g1_robot_open_best_0.yaml \
  --type ycb --group ycb --index 0
```

MobileManiBench 上游提供的是移动操作任务框架（open/close/pull/push/pick）。VLN
需要在此基础上另外定义语言目标、导航观测/动作和成功条件；VLA 推理还需要选定并下载
具体 VLA 权重，不能只靠仿真资产启动。

当前 `./mobilemanibench.sh doctor` 为 12/15：项目 G1-D 和已有房间资产可用，但完整
官方 G1/YCB 资产归档仍缺失。因此官方 smoke 不能仅凭脚本存在判定成功。

## SimpleRoom VLN 演示

在有桌面显示的终端中启动第三人称可视化导航：

```bash
cd /root/autodl-tmp
./mobilemanibench.sh simple-room-vln \
  --command "请带我到沙发旁边" \
  --start-hold-seconds 2 --arrival-hold-seconds 8
```

没有桌面显示时，可以在 headless 模式录制第三人称 GIF：

```bash
cd /root/autodl-tmp
./mobilemanibench.sh simple-room-vln --headless --no-camera \
  --command "go to the sofa" \
  --record-gif outputs/simple_room_vln/navigation.gif
```

红色圆盘是语言目标，绿色线段是规划路径。运行摘要写入
`outputs/simple_room_vln/run_summary.json`。

## 多货架 Warehouse G1-D 导航

当前选择 NVIDIA `warehouse_multiple_shelves.usd` 作为复杂场景。它包含三列长货架和
多条通道，主 Isaac Sim 6.0.1 实测可通过远端 reference 组合。运行：

```bash
cd /root/autodl-tmp

./mobilemanibench.sh warehouse-vln \
  --headless --test --no-camera \
  --command '请带我到东侧货架通道'

./mobilemanibench.sh agent \
  --navigation-scene warehouse \
  --command '请带我到东侧货架通道'
```

`warehouse-vln` 保留显式 collision bootstrap 基线；正式 RGB-only 地图使用：

```bash
./mobilemanibench.sh warehouse-map --stage all
./mobilemanibench.sh warehouse-vln-formal \
  --headless --no-camera --wheel-physics-only \
  --steps 12000 --position-tolerance 0.20 --yaw-tolerance 0.20 \
  --command '请带我到东侧货架通道'
```

188 帧 G1-D RGB 巡检、LingBot 推理、pose-anchored 米制融合、SAM3.1 货架跟踪和地点
审核已经完成；东/西通道批准，未覆盖的装卸区拒绝。东侧正式定向路线在纯轮地接触模式
连续三次通过：10,306 帧、物理路程 32.516 m、位置/朝向误差 0.190 m/0.051 rad。
完整证据和实体机边界详见
`docs/WAREHOUSE_G1D_NAV.md` 与 `docs/G1D_REAL_ROS2_NAV.md`。
