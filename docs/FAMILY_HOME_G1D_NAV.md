# G1-D 稍复杂家庭场景导航

更新时间：2026-08-03（UTC）

## 1. 场景选择与边界

本场景面向家庭服务任务，不再把 Warehouse 作为最终演示环境。它复用本机
`Assets/room/IsaacSim/SimpleRoom_flat.usd` 的约 10 m × 10 m 房间壳体和
`Assets/room/GenieSim/scenes/iros/SofaTablePlant.usd` 的沙发、茶几与绿植，并增加带
碰撞的隔墙、床、餐桌、厨房操作台和电视柜，形成四个可导航区域：

- 客厅：`living_room_sofa`
- 卧室：`bedroom_bed`
- 餐区：`dining_area`
- 厨房：`kitchen_counter`

隔墙保留适合 G1-D 约 0.8 m 直径 footprint 通过的门洞。场景还从本机 ReplicaCAD
数据加入植物、台灯、显示器、篮子、杯、碗、包、书、遥控器和厨房小物件。这些物品有
可见网格和碰撞，但使用 `Item01...` 不透明 prim 名，不注册类别语义；源码里的
`evaluation_label` 只用于离线验收，绝不送给机器人。现有
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

首次建立地图，或者场景/相机发生变化后，才运行家庭 RGB 巡检：

```bash
./mobilemanibench.sh home-assets
./mobilemanibench.sh home-survey --headless --resolution 640x360
./mobilemanibench.sh home-discover
./mobilemanibench.sh home-map
```

`home-discover` 只给 Florence-2 `<OD>` / `<DENSE_REGION_CAPTION>` 任务 token 和
机器人 RGB；它不读取 USD 语义、物品坐标或类别清单。全图与重叠局部视图的模型输出先
经过跨帧一致性和结构/巡检线伪影门，再把模型自己生成的标签与首见帧交给 SAM3。人工
`--sam3-prompt` 只允许显式诊断 override，正式流程不会使用。

这些步骤是离线准备流程，耗时明显，不能把它们当成每次打开网页都要执行的启动命令。正式
四层制品已经存在且场景没有变化时，日常使用只运行：

```bash
./mobilemanibench.sh home-web --host 0.0.0.0 --port 6012
```

浏览器打开 `http://服务器地址:6012`。页面输入指令后会启动一个独立 Isaac Sim 任务，并
同步显示：

- 960 × 540 RTX 跟随相机中的 G1-D 导航过程；
- Point Cloud：G1-D 自采 RGB 经 LingBot-Map 生成的 RGB 点云；
- Semantic：自主发现标签驱动的 SAM3.1 mask 经 LingBot 深度投影并过滤后期跟踪漂移；
- Occupancy：LingBot RGB-only 深度点云生成的 ROS 栅格；
- Region：正式可通行空间按已通过的 SAM3 语义锚点做测地划分；
- 同一 `map` 坐标系中的规划路径、实际轨迹、机器人朝向、速度和航点。
- G1-D 巡检帧数、SAM3 原始检测数、通过过滤的 map-frame 证据数、语义锚点坐标；
- 客厅、卧室、餐区、厨房分别是“已确认”还是“巡检覆盖但语义未确认”；
- Florence-2 自己生成了哪些物体名称、跨多少帧、是否已由 SAM3 投影入图及是否可导航。

网页只接受 Point Cloud、Semantic、Occupancy、Region 四层都存在的正式 bundle；任一
文件缺失即拒绝启动，不会退回 bootstrap。当前审核开放“客厅沙发旁”和“餐区”，卧室与
厨房操作台仍因语义/覆盖证据不足保持拒绝。页面首屏、
状态和识别报告不等待 Isaac；只有提交导航指令后才冷启动 Isaac，因此 RTX 画面会晚于
网页本身出现。

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

同一 Isaac 会话的双脑链路：

```bash
./mobilemanibench.sh home-dual-agent \
  --headless --test --resolution 640x360 \
  --command '请带我到客厅沙发旁' \
  --target-object houseplant
```

可执行“去—拿—返回”完整原型：

```bash
./mobilemanibench.sh home-task \
  --headless --test --resolution 640x360 \
  --command '请带我去餐厅，拿杯子，再回到客厅沙发旁'
```

命令由 v2 Executive 依次取得底盘/右臂/右手控制租约并执行 `NAVIGATE`、实时
`SEARCH_OBJECT`、`APPROACH_AND_ALIGN`、`MANIPULATE`、`VERIFY`。搜索阶段在当前机器人
位置用机载 RGB 拍摄 9 个视角，Florence-2 仍只接收 RGB 和任务 token；目标名称只在
推理完成后用于匹配审核对象目录。Isaac 应用在外部 Python 3.10 感知 sidecar 推理时保持
存活和制动，不重载场景。杯子先在 1.11 m 的扫描可见方位做 3 航向 × 3 俯角 RGB
门控，再沿正式 occupancy 收近到约 0.68 m 的机械臂工作位。公开 OpenVLA 的动作未按
G1-D 标定，因此只作策略建议；仿真执行由审核三维锚点、有界单臂位置 IK、G1-D 多指手
目标和显式 PhysX 固定约束完成，并用实际抬升/稳定持有验证。该路径
`hardware_output=false`，不代表正式 VLA 或实体机器人抓取。

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
- 新物品巡检的 LingBot 实际处理 215 帧 RGB；全局 Sim(3) 只有 22/215 个对应点通过
  0.45 m 门槛而被拒绝，随后以明确标注的 pose-anchored 离线融合生成
  169 × 152、0.05 m/cell 正式图。
- 旧正式图的 SAM3.1 沙发提示得到 87 个检测；人工检查发现离开视野后有跟踪漂移，正式投影用提示帧
  起 36 帧窗口保留 36 条证据。床、餐桌、操作台在 0.50 和 0.20 阈值下均为 0，地点
  审核结果为 1 approved / 3 rejected。
- 正式扫描地图上的沙发导航成功：2 个 waypoint、1.838 m、523 帧，位置误差
  0.119 m、航向误差 0.120 rad。
- 正式网页 API 返回 169 × 153 地图、四层 `FORMAL` 资产和唯一获批沙发地点；对未批准
  卧室指令返回 HTTP 400，未启动 Isaac。
- 新增无语义家庭物品后的 G1-D 重巡检仍为 215 帧并成功。Florence-2 在 80 个均匀
  RGB 帧及其无类别局部视图上自主生成 621 条原始检测；跨帧质量门接受 14 个标签，
  包括新增实体 `houseplant`（14 帧）和 `monitor`（2 帧）。这一步没有读取资产名称或
  坐标。新图 SAM3 投影后有 7 类形成 map-frame 锚点和 7 个 region；人工审核批准
  `table`、`houseplant`、`couch`、`stool`、`coffee table` 进入对象搜索/停靠目录，
  拒绝 `box` 歧义、`sofa` 重复以及 `bathtub/board/whiteboard/curtain` 等错误标签。
  `monitor` 虽有 RGB 候选，但没有合格 map-frame 证据，因此保持拒绝。
- 同一 Isaac SimulationApp 端到端双脑验收成功到达 VLA 边界：正式客厅导航位置误差
  0.120 m；9 帧 live RGB 类别自由发现输出 12 类并确认 `houseplant`（4 帧）及
  `potted plant`（3 帧）；对象精停位置误差 0.030 m、底盘到对象锚点 0.749 m、朝向
  误差 0.049 rad。三步 `application_id` 相同，随后按设计
  `blocked/vla_unavailable`。
- 2026-08-03 新版近距离巡检采集 292 张 `1280x720` RGB；Florence 在 160 个抽样帧
  产生 1302 条原始检测并接受 29 个自主标签。LingBot/SAM3 正式流水线形成
  170 × 154 occupancy、399 条投影观察、18 个语义锚点/region、5 个审核对象和
  2 个审核地点。
- `coffee cup` 的正式三维锚点由 7 帧 reviewed SAM3 mask 与巡检相机射线三角化得到，
  使用 0.183 m 相机基线，中位射线误差 0.016 m；没有读取 USD 语义或物体坐标。
- 指令“请带我去餐厅，拿杯子，再回到客厅沙发旁”在同一
  `application_id=isaac-sim-122461` 成功。去程 3.581 m/904 帧；杯子实时 RGB bbox
  在 35° 俯角通过门控，底盘到扫描锚点 0.766 m；杯体实际抬升 0.293 m、稳定保持
  30 帧；回程 4.237 m/946 帧，掌心—杯体距离漂移 0.00019 m。

以上导航使用 `stable_assisted`，用于稳定验证语言目标、地图、规划、相机和 Agent
高层链路。它会写轮速，同时以确定性平面位姿更新保证回归，不等于家庭场景已经通过
`--wheel-physics-only` 纯轮地接触验收。

## 4. 从 RGB 巡检生成正式地图

`family_home_vln/layout.py` 生成的 bootstrap 仍只用于巡检路线和显式回归，不能作为
正式导航输入。正式步骤由 `scripts/build_family_home_map.py` 实现：

1. 用 `home-survey` 让 G1-D 头部相机覆盖卧室、客厅、餐区和厨房。
2. Florence-2 只读取 RGB，以任务 token 自主发现物体标签和框；至少跨两个巡检帧才进入
   后续候选，不把场景配置中的物品名称告诉模型。
3. LingBot-Map 只读取 `survey/rgb/*.png`，输出视觉深度、相机运动和点云。
4. LingBot 推理完成后，再使用巡检相机位姿做米制 Sim(3) 对齐；若改用
   pose-anchored 融合，必须在制品中显式标注。
5. 用自主发现的标签和首见帧启动 SAM3，把 mask 通过 LingBot 深度和 TF 投影到 `map`。
6. 从点云构建 ROS occupancy map，并按 G1-D footprint 膨胀障碍。
7. 发现完成后才把 `couch/sofa` 等模型词汇映射到导航地点 ontology；审核 docking
   pose、朝向、clearance、路径可达性和地图哈希，只开放通过
   的地点 ID。
8. 正式产物写到
   `outputs/family_home_vln/lingbot_map/map.yaml` 和
   `outputs/family_home_vln/places_formal.json`。自主物体另写入
   `objects_formal.json`；它保存审核状态、扫描锚点、对象专属停靠距离和
   `manipulation_ready`，拒绝项仍保留证据但不能被 Agent 解析。
9. 网页还要求 `mapping_summary.json` 引用真实四层 PNG，缺失时 fail-closed。
10. 正式地图通过后，才在同一场景做 `--wheel-physics-only` 三次连续导航与制动验收。

相机位姿不能作为 LingBot 的模型输入，也不能把 Isaac 几何直接栅格化后标成
“RGB-only 正式地图”。

当前家具是低多边形基础几何，SAM3 未能把床和操作台识别为对应类别；通用 `table`
也不足以在审核中自动等同于餐桌地点。
这是当前资产/视觉域限制，不应通过读取 `HOME_FIXTURES` 坐标绕过。下一步应替换为更真实、
具有稳定视觉外观的家庭 USD，重新巡检并复核提示帧，再重新审核这三个地点。

## 5. Agent、VLA 与真机接口

`FamilyHomeVlnAdapter` 让任务级 Agent 把家庭地点指令交给现有 VLN。语言只选择审核
`place_id`，不生成任意坐标。`home-dual-agent` 已把 live RGB 搜索和家庭对象精停接到
v2 Executive；`home-task` 进一步接入仿真单右臂位置 IK、显式抓取约束和独立抬升验证。
公开 OpenVLA 仍不是 G1-D 标定控制器，其输出不会直接写入关节。

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

家庭正式图的物理 bringup 入口为
`./mobilemanibench.sh g1d-home-real-nav`，它仍强制
`allow_hardware_output:=False`。当前主机没有已确认厂商驱动、机器人网络设备、实体硬
急停回路或真实 `/scan`，所以不能据此声称物理 G1-D 已运动。
