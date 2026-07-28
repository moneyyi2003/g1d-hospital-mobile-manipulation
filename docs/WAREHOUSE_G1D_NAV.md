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

已完成的确定性回归为“请带我到东侧货架通道”：规划 3 个 waypoint、路径
22.029 m，`stable_assisted` 在 1,936 帧到达，位置误差 0.149 m、航向误差
0.146 rad。该结果验证场景、G1-D、语义目标、地图、规划和高层执行已经连通，不代表
纯轮地接触底盘或物理真机已经验收。

## 2. 实现组成

- `run_g1d_warehouse_vln.py`：Isaac 场景组合、G1-D 加载、建图制品选择、规划、路径跟随、
  RGB 巡检和运行摘要。
- `warehouse_vln/artifacts.py`：Warehouse 场景常量、bootstrap occupancy、语义地点和
  巡检路径。
- `warehouse_vln/kinematics.py`：G1-D 导航坐标与导入 USD 的底盘朝向、轮子符号和几何
  参数。
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

# 让 Agent 选择 Warehouse VLN；默认只输出计划
./mobilemanibench.sh agent \
  --navigation-scene warehouse \
  --command '请带我到东侧货架通道'
```

`mobilemanibench.sh warehouse-vln` 当前显式启用
`--allow-bootstrap`，用于在正式 LingBot 制品完成前保证场景联调可运行。输出位于
`outputs/warehouse_vln/`，不进入 Git。

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

## 5. 用现有 LingBot/SAM3 链替换 Bootstrap

建议按 Hospital 已验证流程执行：

1. 运行 Warehouse RGB 巡检，让 G1-D 头部相机覆盖两条长货架通道：

   ```bash
   ./mobilemanibench.sh warehouse-survey \
     --headless --resolution 640x360
   ```

2. 将 `outputs/warehouse_vln/survey/` 中的 RGB 和 manifest 送入现有
   LingBot-Map RGB-only 推理；相机位姿只能在推理后用于米制 Sim(3) 对齐，不能冒充
   纯视觉全局建图。
3. 用 SAM3 识别货架、托盘、装卸区和后续操作物体，把语义与 LingBot 点云/region 对齐。
4. 审核 occupancy map 的通道开口、连通性和尺度，输出 ROS map YAML/PGM。
5. 审核每个地点的 `place_id`、别名、区域和面向通道的 docking pose，生成
   `places_formal.json`。语言模型只能返回这些 ID。
6. 将正式制品放到默认位置：

   ```text
   outputs/warehouse_vln/lingbot_map/map.yaml
   outputs/warehouse_vln/places_formal.json
   ```

7. 不使用 shell 的 bootstrap 包装，直接执行正式模式：

   ```bash
   OMNI_KIT_ACCEPT_EULA=YES /root/autodl-tmp/isaacsim/python.sh \
     /root/autodl-tmp/run_g1d_warehouse_vln.py \
     --headless --no-camera \
     --map /root/autodl-tmp/outputs/warehouse_vln/lingbot_map/map.yaml \
     --places /root/autodl-tmp/outputs/warehouse_vln/places_formal.json \
     --command '请带我到东侧货架通道'
   ```

runner 在没有 `--allow-bootstrap` 时会要求两个正式文件都存在，并检查审核起点在
occupancy 中为可行区域；缺少制品会直接失败。

## 6. 轮子控制和实体机边界

实际 G1-D USD 探针确认了两个导入约定：

- 导航的机体 `+X` 前方对应 USD root 增加 `pi` 的朝向；
- 导航正角速度在差速轮边界需要乘 `-1`；
- 轮半径为 `0.0848 m`，轮距为 `0.4062 m`。

修正后，300 帧直行探针从 `x=-5.0 m` 正确移动到约 `x=-4.09 m`，转向探针也从正向
转弯进入路径跟随；但短探针没有完成 22 m 任务，尚未形成纯轮地接触导航验收。

当前“真实机器人”指在 Isaac 中加载项目真实 G1-D 结构的数字孪生。要把相同 VLN 跑到
物理 G1-D，还必须单独完成：

1. ROS 2 底盘驱动和速度/制动接口；
2. `map -> odom -> base_link` TF、轮速里程计和定位；
3. Nav2 footprint、速度/加速度、局部避障和恢复参数；
4. 相机标定、时间同步、LingBot/SAM3 的真实观测坐标转换；
5. 急停、碰撞限速、人工接管和低速隔离场验收；
6. 直行、原地转向、制动和至少三次固定路线重复性测试。

实体机执行器桥接和这些安全项尚未接入，因此不能将当前 Isaac 成功描述为物理真机已经
在 Warehouse 中完成导航。
