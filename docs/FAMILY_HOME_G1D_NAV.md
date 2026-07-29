# G1-D 稍复杂家庭场景导航

更新时间：2026-07-29（UTC）

## 1. 场景选择与边界

本场景面向家庭服务任务，不再把 Warehouse 作为最终演示环境。它复用本机
`Assets/room/IsaacSim/SimpleRoom_flat.usd` 的约 10 m × 10 m 房间壳体和
`Assets/room/GenieSim/scenes/iros/SofaTablePlant.usd` 的沙发、茶几与绿植，并增加带
碰撞的隔墙、床、餐桌、厨房操作台和电视柜，形成四个可导航区域：

- 客厅：`living_room_sofa`
- 卧室：`bedroom_bed`
- 餐区：`dining_area`
- 厨房：`kitchen_counter`

隔墙保留适合 G1-D 约 0.8 m 直径 footprint 通过的门洞。新增家具目前是有颜色、碰撞和
语义属性的基础几何体，重点用于导航、建图和任务链验证，不是最终写实美术资产。现有
SofaTablePlant 缺少部分相对路径贴图，因此沙发几何和碰撞可用，但渲染会报告材质缺失。

MobileManiBench 配置中虽有更完整的家庭资产名称，但本机缺少完整官方资产包。当前实现
只组合已有资产，没有下载约 TB 级数据仓库或覆盖用户资产。后续取得经许可的家庭 USD
后，可以保留同一 `family-home` profile、地点审核和 Agent 接口，只替换场景引用与重新
生成正式地图。

## 2. 运行方式

家庭 bootstrap 导航：

```bash
./mobilemanibench.sh home-vln \
  --headless --test --no-camera \
  --command '我困了，请带我到卧室床边'
```

家庭 RGB 巡检：

```bash
./mobilemanibench.sh home-survey --headless --resolution 640x360
```

从巡检 RGB 构建正式四层地图：

```bash
./mobilemanibench.sh home-map
```

家庭实时可视化控制台：

```bash
./mobilemanibench.sh home-web --host 0.0.0.0 --port 6012
```

浏览器打开 `http://服务器地址:6012`。页面输入指令后会启动一个独立 Isaac Sim 任务，并
同步显示：

- 960 × 540 RTX 跟随相机中的 G1-D 导航过程；
- Point Cloud：G1-D 自采 RGB 经 LingBot-Map 生成的 RGB 点云；
- Semantic：SAM3.1 mask 经 LingBot 深度投影并过滤后期跟踪漂移；
- Occupancy：LingBot RGB-only 深度点云生成的 ROS 栅格；
- Region：正式可通行空间按已通过的 SAM3 语义锚点做测地划分；
- 同一 `map` 坐标系中的规划路径、实际轨迹、机器人朝向、速度和航点。

网页只接受 Point Cloud、Semantic、Occupancy、Region 四层都存在的正式 bundle；任一
文件缺失即拒绝启动，不会退回 bootstrap。当前审核只开放“客厅沙发旁”，所以网页不会
显示或执行卧室、餐桌和操作台指令。

Agent 只生成任务计划：

```bash
./mobilemanibench.sh agent \
  --navigation-scene home \
  --command '请带我到客厅沙发旁'
```

`home-vln` 仍显式允许 bootstrap，只用于集成回归。Agent 和网页默认使用：

```bash
./mobilemanibench.sh home-vln-formal \
  --headless --test --no-camera \
  --command '请带我到客厅沙发旁'
```

`home-vln-formal` 不带 `--allow-bootstrap`。正式地图或地点文件不存在、哈希不匹配或地点
未审核时必须失败，不能回退到几何真值。

## 3. 已完成的 Isaac 验证

2026-07-29 在主 Isaac Sim 6.0.1 中实际加载项目 G1-D 数字孪生：

- 卧室导航：2 个 waypoint、规划路径 2.378 m、530 帧，位置误差 0.120 m、航向误差
  0.119 rad，成功；
- 全屋巡检：14 个 waypoint、路线 19.285 m、3750 帧，采集 215 张 640 × 360 RGB，
  最终位置误差 0.119 m、航向误差 0.119 rad，成功；
- 四个地点均通过 0.40 m footprint 安全、从起点可达和审核 catalog 约束测试；
- 原 SimpleRoom 回归仍为 657 帧、2.733 m、0.119 m/0.119 rad，成功。
- 初版 bootstrap 家庭网页 API 曾实际提交卧室任务：状态按
  `starting -> loading -> running -> succeeded` 更新，530 帧、55 个实时轨迹采样点；
  餐桌任务为 548 帧、73 个轨迹采样点。MJPEG 可读取，卧室/餐区最终帧已目视确认能看到
  G1-D、家庭家具和绿色规划路线。
- LingBot 实际处理 215 帧 RGB；全局 Sim(3) 只有 22/215 个对应点通过 0.45 m 门槛而被
  拒绝，随后以明确标注的 pose-anchored 离线融合生成 169 × 153、0.05 m/cell 正式图。
- SAM3.1 沙发提示得到 87 个检测；人工检查发现离开视野后有跟踪漂移，正式投影用提示帧
  起 36 帧窗口保留 36 条证据。床、餐桌、操作台在 0.50 和 0.20 阈值下均为 0，地点
  审核结果为 1 approved / 3 rejected。
- 正式扫描地图上的沙发导航成功：2 个 waypoint、1.838 m、523 帧，位置误差
  0.119 m、航向误差 0.120 rad。
- 正式网页 API 返回 169 × 153 地图、四层 `FORMAL` 资产和唯一获批沙发地点；对未批准
  卧室指令返回 HTTP 400，未启动 Isaac。

以上导航使用 `stable_assisted`，用于稳定验证语言目标、地图、规划、相机和 Agent
高层链路。它会写轮速，同时以确定性平面位姿更新保证回归，不等于家庭场景已经通过
`--wheel-physics-only` 纯轮地接触验收。

## 4. 从 RGB 巡检生成正式地图

`family_home_vln/layout.py` 生成的 bootstrap 仍只用于巡检路线和显式回归，不能作为
正式导航输入。正式步骤由 `scripts/build_family_home_map.py` 实现：

1. 用 `home-survey` 让 G1-D 头部相机覆盖卧室、客厅、餐区和厨房。
2. LingBot-Map 只读取 `survey/rgb/*.png`，输出视觉深度、相机运动和点云。
3. LingBot 推理完成后，再使用巡检相机位姿做米制 Sim(3) 对齐；若改用
   pose-anchored 融合，必须在制品中显式标注。
4. 用 SAM3 对床、沙发、餐桌和操作台做语义对齐，把检测通过深度和 TF 投影到
   `map`。
5. 从点云构建 ROS occupancy map，并按 G1-D footprint 膨胀障碍。
6. 审核每个地点的 docking pose、朝向、clearance、路径可达性和地图哈希，只开放通过
   的地点 ID。
7. 正式产物写到
   `outputs/family_home_vln/lingbot_map/map.yaml` 和
   `outputs/family_home_vln/places_formal.json`，再运行 `home-vln-formal`。
8. 网页还要求 `mapping_summary.json` 引用真实四层 PNG，缺失时 fail-closed。
9. 正式地图通过后，才在同一场景做 `--wheel-physics-only` 三次连续导航与制动验收。

相机位姿不能作为 LingBot 的模型输入，也不能把 Isaac 几何直接栅格化后标成
“RGB-only 正式地图”。

当前家具是低多边形基础几何，SAM3 未能把蓝色方块床、灰色餐桌和操作台识别为对应类别。
这是当前资产/视觉域限制，不应通过读取 `HOME_FIXTURES` 坐标绕过。下一步应替换为更真实、
具有稳定视觉外观的家庭 USD，重新巡检并复核提示帧，再重新审核这三个地点。

## 5. Agent、VLA 与真机接口

`FamilyHomeVlnAdapter` 让任务级 Agent 把家庭地点指令交给现有 VLN。语言只选择审核
`place_id`，不生成任意坐标。家庭物体预抓取和 VLA 尚未接好，因此操作指令在该 adapter
上会明确 `blocked`，不会把“到达房间”误报为“已经抓取”。

VLA 到货后，应在同一 Isaac SimulationApp 中保持场景、机器人、相机与时间连续：

```text
语言任务
  -> Agent 选择 VLN
  -> 正式 occupancy/Nav2 到审核地点
  -> RGB/深度/TF 计算物体相对位姿
  -> 局部停靠满足该物体和技能的距离、横向、朝向区间
  -> 底盘制动并通过 VLA readiness gate
  -> Agent 选择 VLA，仅接管手臂和手
  -> 独立成功判据确认抓取/放置
```

“真实机器人在仿真平台完成任务”在工程上指：Isaac 中使用与物理 G1-D 同构的 USD、关节、
TF、传感器和控制接口做数字孪生验证。当前代码没有驱动物理机器人。接实体 G1-D 时必须
继续使用现有 fail-closed ROS 2/Nav2 bridge，先接厂商轮驱、真实里程计、`/scan`、硬
急停和 driver-ready，再依次做架空轮、隔离场低速与通信丢失制动验收；仿真成功不能替代
这些真机安全步骤。
