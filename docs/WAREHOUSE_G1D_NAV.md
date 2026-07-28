# MobileManiBench 多货架 Warehouse 的 G1-D 导航

更新时间：2026-07-28（UTC）

## 1. 选用场景和当前结论

MobileManiBench 的场景配置列出了 Corner、SimpleRoom、WareHouse、ShelveHouse、
Hospital 和 Office 等 Isaac 场景。当前机器没有完整的官方 `Assets.zip`，但
NVIDIA 远端资产中的
`Simple_Warehouse/warehouse_multiple_shelves.usd` 可以被主 Isaac Sim 6.0.1
作为 reference 正常组合，因此本项目选择它作为比 SimpleRoom 和 Hospital 前厅更复杂的
导航联调场景。

实际组合审计结果：

- 场景范围约为 `x=[-12, 12] m`、`y=[-18, 20.818] m`；
- 8,139 个 prim、1,878 个 mesh、1,878 个 collision prim；
- 三列长货架、托盘箱和装卸空间形成多条长通道；
- 加载的机器人是 `/root/autodl-tmp/Assets/g1_d_robot/g1_d.usd`，不是
  MobileManiBench 自带 G1。

Bootstrap 路线“请带我到东侧货架通道”已经在 `--wheel-physics-only` 下连续三次通过。
该模式只写左右轮目标，路径跟随和验收读取物理实体 G1-D 的实际 root 位姿，不进行
assisted 平面位姿写入。三次结果一致：物理路程 22.289 m、最终位置误差 0.192 m、
航向误差 0.169 rad、最大 roll/pitch 约 0.023/0.146 rad；零轮速制动 2 秒后的漂移
0.007 m，停止线/角速度约 0.0042 m/s、0.0115 rad/s。

正式视觉地图也已生成并替换默认正式插槽。它审核开放东/西货架通道；未被巡检路线覆盖的
装卸区保持 `rejected`。正式东侧路线采用“预停靠点 `(4,8)` → docking pose `(4,9)`”
的定向末段，并在纯轮地接触模式连续三次通过。物理真机接口已经实现，但尚无厂商驱动和
实体安全验收，因此不能表述为物理机器人已经运动。

## 2. 实现组成

- `run_g1d_warehouse_vln.py`：Isaac 场景组合、G1-D 加载、建图制品选择、规划、路径跟随、
  RGB 巡检和运行摘要。
- `warehouse_vln/artifacts.py`：Warehouse 场景常量、bootstrap occupancy、语义地点和
  巡检路径。
- `warehouse_vln/kinematics.py`：G1-D 导航坐标与导入 USD 的底盘朝向、轮子符号和几何
  参数。
- `warehouse_vln/physics.py`：物理路程、倾斜、轮速、制动漂移和停止速度验收。
- `warehouse_vln/formal_places.py`：正式 occupancy 上的 footprint、连通性和地点审核。
- `scripts/build_warehouse_map.py`：LingBot、对齐、occupancy、SAM3 投影、地点库和预览。
- `scripts/audit_mobile_scene.py`：组合任意 USD 并统计范围、mesh 和碰撞。
- `g1d_agent/adapters.py`：`WarehouseVlnAdapter`；Agent 的纯区域导航步骤可以选择
  Warehouse，物体预抓取在该场景尚未实现时会 fail-closed。

Warehouse runner 复用项目已有的 `GridMap`、`Place`、`resolve_place`、
`load_lingbot_artifacts`、A* 路径规划和 `PathFollower`，没有引入另一套语言导航模型。
DeepSeek/SAM3/LingBot 的职责边界也不变：语言模型只选择审核地点 ID，坐标来自地点库。

## 3. 立即可运行的命令

先确认没有第二个 Isaac Kit 进程，再从工作区运行：

```bash
cd /root/autodl-tmp

# 审计默认多货架场景
./mobilemanibench.sh warehouse-scene-audit \
  'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd' \
  --name-contains Shelf --name-contains PalletBin

# 只组合真实场景并规划，不加载机器人运动循环
./mobilemanibench.sh warehouse-vln \
  --headless --plan-only --no-camera \
  --command '请带我到东侧货架通道'

# 加载真实 G1-D USD 的确定性 Isaac 回归
./mobilemanibench.sh warehouse-vln \
  --headless --test --no-camera \
  --command '请带我到东侧货架通道'

# 使用正式 RGB-only 地图；不带 --allow-bootstrap
./mobilemanibench.sh warehouse-vln-formal \
  --headless --no-camera --wheel-physics-only \
  --steps 12000 --position-tolerance 0.20 --yaw-tolerance 0.20 \
  --command '请带我到东侧货架通道'

# 让 Agent 选择 Warehouse VLN；默认只输出计划
./mobilemanibench.sh agent \
  --navigation-scene warehouse \
  --command '请带我到东侧货架通道'
```

`mobilemanibench.sh warehouse-vln` 始终显式启用 bootstrap，只用于基线和碰撞图诊断；
正式回归必须使用 `warehouse-vln-formal`。输出位于 `outputs/warehouse_vln/`，不进入
Git。

## 4. Bootstrap 地图的真实性边界

首次联调会从组合后的 Isaac collision geometry 读取经过挑选的 `Shelf_*` 和
`PalletBin_*` 顶层包围盒，投影到 XY 平面，以 G1-D `0.42 m` footprint 半径膨胀后生成
栅格。墙和柱子的顶层 AABB 包含真实开口；直接栅格化会错误封死通道，所以它们没有被
纳入 bootstrap。

这张图只是 simulator ground-truth 联调图，不是 LingBot 视觉建图结果，也不能作为
物理真机正式地图。制品在自身 metadata 中记录：

- `source=isaac_collision_aabb_bootstrap`；
- 使用和忽略了哪些碰撞根；
- 场景 URL、边界、分辨率、机器人半径和数量；
- 地点审核状态为 `accepted_for_bootstrap_demo`。

正式导航不得把这个标记改名为 LingBot 或声称它来自 RGB。

## 5. 已完成的 LingBot/SAM3 正式地图

已按 Hospital 的真实性边界执行以下流水线：

1. 运行 Warehouse RGB 巡检，让 G1-D 头部相机覆盖两条长货架通道：

   ```bash
   ./mobilemanibench.sh warehouse-survey \
     --headless --resolution 640x360
   ```

2. 188 张 `640x360` RGB 覆盖 32.605 m 巡检路线。manifest 明确
   `rgb_is_only_model_input=true`。
3. LingBot-Map 只读取 RGB，完成 188 帧推理；全局 Sim(3) 因 0/188 对应点满足
   0.45 m 阈值而被拒绝，随后使用明确标注的离线 survey-pose anchored 深度融合。
4. 生成 372 x 617、0.05 m/cell 的 ROS occupancy；188 帧保留约 239 万个点。
5. 官方 SAM3.1 multiplex checkpoint 以 `warehouse shelf` 文本提示跟踪全部 188 帧，
   产生 548 次检测；546 条通过 LingBot 深度置信度过滤并投影到 `map` 坐标系。
6. 以 `0.42 m` G1-D footprint 审核地点，并要求 docking pose 朝向后方 `1.0 m`
   的预停靠点安全、可达且能直线进入。东/西通道原始 pose 无需 snap，定向规划路径约
   32.538 m 和 19.004 m；装卸区在 0.75 m 内没有正式 free cell，因此拒绝。
7. 正式制品位于：

   ```text
   outputs/warehouse_vln/lingbot_map/map.yaml
   outputs/warehouse_vln/places_formal.json
   ```

完整构建命令：

```bash
./mobilemanibench.sh warehouse-map --stage all
```

已有 LingBot/SAM3 制品时可分别运行 `--stage align|map|project|places|render`。SAM3 是
实际模型输出，不是根据场景真值伪造的标签；相机位姿没有进入 LingBot 或 SAM3 推理，只在
之后用于米制几何融合。

runner 在没有 `--allow-bootstrap` 时会要求两个正式文件都存在，并检查审核起点在
occupancy 中为可行区域；缺少制品会直接失败。

## 6. 轮子控制和实体机边界

实际 G1-D USD 探针确认了两个导入约定：

- 导航的机体 `+X` 前方对应 USD root 增加 `pi` 的朝向；
- 导航正角速度在差速轮边界需要乘 `-1`；
- 轮半径为 `0.0848 m`，轮距为 `0.4062 m`。

纯轮物理验收还要求导航完成、目标误差、roll/pitch、制动漂移和最终停止速度同时通过；
只到达坐标但没有可靠制动仍算失败。22 m bootstrap 路线和 32.538 m 正式 occupancy
定向路线均已分别连续三次通过。正式路线三次结果完全一致：

- 10,306 帧，物理实体实际行驶 32.516 m；
- 位置误差 0.190 m，朝向误差 0.051 rad；
- 最大 roll/pitch 0.026/0.147 rad；
- 零轮速制动 2 秒漂移 0.0086 m；
- 停止线/角速度 0.0034 m/s、0.0095 rad/s；
- `physics.accepted=true`，失败列表为空。

早期直接到 `(4,9,+pi/2)` 的路线从目标北侧向南接近，末端需要原地转 180°，会在位置
阈值边缘形成“对齐—重新靠近”极限环。正式地点审核现在保存 `(4,8,+pi/2)` 预停靠点，
runner 与 ROS 2 语言目标节点都先提交预停靠，再沿审核朝向进入最终点。

物理 G1-D 的 ROS 2 接口已经提供：

- `map -> odom -> AGV_link` TF 和轮编码器 `/odom`；
- Nav2 RPP、`0.42 m` footprint、速度/加速度和 `/scan` 障碍层；
- cmd_vel 看门狗、软件制动、锁存急停、硬急停心跳、driver ready 和反馈时效检查；
- 默认 `allow_hardware_output=False`，启动时急停锁存且未 arm。

本机仍没有 G1-D 厂商底盘驱动、真实 `/scan`、机器人网络和硬急停确认，所以实际硬件
输出没有开启，也没有执行实体路线。接口、topic/service 契约和真机使能顺序详见
`docs/G1D_REAL_ROS2_NAV.md`。
