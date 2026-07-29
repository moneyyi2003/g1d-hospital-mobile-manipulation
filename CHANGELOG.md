# 重要修改记录

本文件记录能够影响复现、行为、接口或任务状态的重要变更。日期使用 UTC；生成物刷新和
无行为影响的小改动不单独记录。

## 2026-07-29

### 家庭自扫描正式地图、语义地点与 fail-closed 网页

- 新增 `home-map`：从 G1-D 自采 RGB 依次运行 LingBot、推理后米制对齐、点云/
  occupancy、SAM3.1、map-frame 投影、semantic/region、地点审核和四层预览。
- 家庭地点不再从预设家具坐标批准；每个地点必须有匹配类别的 SAM3 证据，停靠位从该
  语义锚点附近的正式 occupancy 安全可达栅格生成并面向物体。
- 修复 `--test` 隐式启用 bootstrap 的问题；Agent 改用 `home-vln-formal`。家庭网页
  只加载正式四层资产，缺少任一制品即拒绝启动，未审核地点也不会启动 Isaac。
- SAM3 正向/反向切换触发上游 `No points are provided` 后，家庭顺序巡检采用
  forward-only；增加提示后 36 帧投影门，过滤物体离开视野后的 tracker drift。

验证：

- LingBot 实际处理 215 帧；全局 Sim(3) 仅 22/215 对应点通过 0.45 m 门槛而被拒绝，
  pose-anchored 推理后融合生成 169 × 153、0.05 m/cell occupancy。
- 沙发 87 个 SAM3 原始检测中 36 条通过时间窗并投影到 map；床、餐桌、操作台在
  0.50/0.20 阈值下均为 0，因此地点审核为 1 approved / 3 rejected。
- 正式沙发导航成功：523 帧、1.838 m、位置误差 0.119 m、航向误差 0.120 rad。
- 家庭测试 12/12、Agent 测试 28/28、Python/JavaScript 语法通过；网页 API 返回正式
  169 × 153 四层地图和唯一获批沙发地点，未批准卧室指令返回 400。

已知限制：

- 当前低多边形床、餐桌和操作台未被 SAM3 识别；必须改善资产后重新巡检，不能以场景
  真值坐标补齐。
- 正式路线仍是 `stable_assisted`，尚未做家庭 `--wheel-physics-only` 连续三次验收。

### 家庭导航五视图实时网页

- 新增 `home-web` 6012/TCP 控制台：用户输入家庭地点指令后，服务端从审核 catalog
  解析目标、规划路径并启动独立 Isaac Sim 任务。
- 页面同时显示 960 × 540 RTX 跟随画面、Point Cloud、Semantic、Occupancy 和 Region；
  四个地图 canvas 使用同一 `map` 坐标系叠加规划路径、实际轨迹、地点和 G1-D 朝向。
- 家庭 runner 增加通用 `--live-dir`、`--live-fps` 和 `--live-resolution`，原子发布
  状态 JSON 与 MJPEG 帧；相机位置会在房间内部范围内跟随，避免墙体遮挡机器人。
- 当前点云是 952 个碰撞几何代理点，语义、region 和 occupancy 也来自 bootstrap；
  页面和 API 明确显示 truth boundary。仅检测到正式导航三件套时不会错误切换为
  `FORMAL`。

验证：

- 家庭 dashboard/layout 轻量测试 9/9、Python 编译、shell 与 JavaScript 语法通过。
- HTTP `/`、`/api/config`、`/api/map-data`、`/api/state` 和 MJPEG 实际读取成功。
- 通过网页 API 提交卧室任务：530 帧成功、55 个实时轨迹点；提交餐桌任务：
  548 帧成功、73 个实时轨迹点。
- 960 × 540 最终帧已目视确认卧室和餐区都能看到 G1-D、家具和绿色规划路线。
- 原 SimpleRoom 无界面回归仍以 657 帧、2.733 m、0.119 m/0.119 rad 成功。

已知限制：

- 当前网页地图不是 LingBot/SAM3 正式四层制品；正式家庭建图完成后仍需实现逐层资产
  加载与审核切换。
- 控制台直接启动本机 Isaac，适合本机、VNC、自定义 TCP 服务或 SSH 隧道，不是脱离
  仿真后端运行的静态托管网页。

### 多区域家庭场景、G1-D RGB 巡检与 Agent 接入

- 在现有约 10 m × 10 m SimpleRoom 壳体和 SofaTablePlant 基础上增加带碰撞和语义属性
  的床、隔墙、餐桌、厨房操作台和电视柜，形成卧室、客厅、餐区和厨房四个家庭区域。
- 新增 `family_home_vln/`，提供 0.40 m G1-D footprint 的显式 bootstrap occupancy、
  四个约束地点和覆盖全屋的巡检路线；bootstrap 制品明确要求后续用 LingBot 正式地图
  替换。
- `run_g1d_simple_room_vln.py` 增加 `family-home` scene profile；统一入口增加
  `home-vln`、`home-survey` 和 fail-closed 的 `home-vln-formal`。
- Agent 增加 `FamilyHomeVlnAdapter` 和 `--navigation-scene home`；当前只允许家庭区域
  语义导航，物体预抓取和 VLA 未实现时明确阻塞。
- 新增 `docs/FAMILY_HOME_G1D_NAV.md`，记录场景、命令、正式 RGB-only 地图替换流程、
  VLA 接管边界和实体 G1-D 安全边界；同步更新 `vlaandvln.md`。

验证：

- 家庭布局测试 5/5、Agent 测试 28/28、Python 编译和 shell 语法通过。
- 主 Isaac Sim 6.0.1 中卧室导航成功：530 帧、2.378 m、位置误差 0.120 m、航向误差
  0.119 rad。
- 全屋 RGB 巡检成功：3750 帧、19.285 m、215 张 `640x360` RGB，最终位置误差
  0.119 m、航向误差 0.119 rad。
- 原 SimpleRoom 真实 Isaac 回归仍成功：657 帧、2.733 m、0.119 m/0.119 rad。

已知限制：

- 家庭 occupancy 和地点仍是 bootstrap；尚未完成 LingBot RGB-only、SAM3 正式语义
  投影、地点审核和纯轮地接触验收。
- SofaTablePlant 的部分相对纹理资产缺失，几何与碰撞可用但视觉材质不完整。
- 本次只驱动 Isaac 中的 G1-D 数字孪生，没有驱动物理机器人；实体硬件输出继续默认关闭。

## 2026-07-28

### Warehouse 正式 RGB-only 地图、纯轮路线与实体 G1-D ROS 2/Nav2 接口

- G1-D 在 `warehouse_multiple_shelves.usd` 完成 188 帧 `640x360` RGB 巡检，
  覆盖路线 32.605 m；LingBot-Map 只读取 RGB 完成 188 帧推理。Isaac 相机位姿没有作为
  模型输入，只在推理后用于明确标注的 pose-anchored 米制融合。
- 全局 Sim(3) 因 `0/188` 对应点落入 0.45 m 阈值而拒绝；随后以
  `lingbot_depth_to_metric_survey_pose_anchor` 生成 372 x 617、0.05 m/cell 的正式
  occupancy，并保留约 238.6 万个融合点。
- 用官方 SAM3.1 multiplex 权重和 `warehouse shelf` 提示处理全部 188 帧，得到
  548 次检测，其中 546 条通过 LingBot 深度、相机内参和巡检位姿投影到 `map`。
- 新增正式地点构建与审核：只批准 `east_shelf_aisle` 和 `west_shelf_aisle`；
  `loading_zone` 因目标周围 0.75 m 内没有 footprint-safe free cell 保持拒绝。地点库
  绑定正式地图哈希，地图被替换后会 fail-closed。
- 为朝向约束终点增加定向 approach pose；东侧路线先到 `(4,8,+pi/2)`，再直行到
  `(4,9,+pi/2)`，避免原地 180 度终端转向引起的物理限环。Isaac runner 和 ROS 2/Nav2
  语言目标节点都执行同一 `approach -> destination` 序列并分别验收到达位置与朝向。
- 纯轮控制新增物理遥测、姿态稳定、停止速度和 2 秒制动漂移门槛；正式 occupancy 东侧
  32.538 m 路线连续三次通过，每次 10,306 帧、物理路程 32.516 m、位置误差
  0.190 m、朝向误差 0.051 rad、最大 roll/pitch 0.026/0.147 rad、制动漂移
  0.0086 m，停止线/角速度 0.0034 m/s、0.0095 rad/s。
- 新增实体 G1-D ROS 2/Nav2 bringup：发布 `odom -> AGV_link`、轮编码器里程计、
  Nav2 `/scan` 障碍层、安全轮速、制动、锁存急停、硬急停释放心跳、driver/feedback
  watchdog 与诊断；提供 arm/disarm/brake/e-stop/clear 服务。
- 所有硬件输出默认关闭；启动时急停锁存且未 armed。缺失或过期的 driver feedback、
  driver-ready 或硬急停心跳都会制动/锁存，`allow_hardware_output=False` 时 arm 必然
  失败。

验证：

- Warehouse 轻量测试 11/11、LingBot 23/23、Hospital 18/18、Agent 27/27 通过；
  ROS 2 包构建和安全核心测试 4/4 通过，launch 参数可完整展开。
- 正式地图 plan-only 与审核器得到一致路径；三个独立 Isaac 输出
  `formal_physics_5`、`formal_physics_6`、`formal_physics_7` 均满足全部物理门槛。
- ROS 消息级零输出联调确认可发布 odom/TF，安全命令保持零且 brake 为真；完整 Nav2
  lifecycle 在合成当前时间戳的 joint/ready/e-stop/scan 输入下进入 active，硬件输出
  仍为零。

已知限制：

- 本机没有已确认的实体 G1-D 厂商底盘驱动、机器人网络、物理硬急停回路和真实
  `/scan`。本次只完成仿真纯轮路线和 fail-closed ROS 接口，未驱动实体机器人。
- `loading_zone` 尚未通过地点审核；必须扩展 RGB 巡检、重建地图并重新审核，不能手工
  改为 approved。
- VLA backend、G1-D 右臂/多指手映射和真实抓取成功判据仍等待后续交付。

### MobileManiBench 多货架 Warehouse 的 G1-D 导航

- 审计本机 MobileManiBench 场景配置和可用资产，在完整官方 `Assets.zip` 缺失的条件下
  选择 NVIDIA `warehouse_multiple_shelves.usd` 作为复杂导航场景；新增通用场景审计
  脚本，场景以 reference 组合，避免把旧版远端 USD 直接作为 root stage。
- 新增 `run_g1d_warehouse_vln.py` 和 `warehouse_vln/`：加载项目真实 G1-D USD，复用
  现有地点解析、GridMap/A*、PathFollower 和 LingBot map loader，支持计划、RGB 巡检、
  导航、GIF 和轻量运行摘要。
- 新增带真实性声明的 collision bootstrap：只栅格化审核的 `Shelf_*` 和
  `PalletBin_*` 顶层碰撞根并按 G1-D footprint 膨胀；墙/柱 AABB 会抹掉真实开口，故不
  冒充正式 occupancy。无 `--allow-bootstrap` 时必须提供 LingBot map 和审核地点库。
- 实机型 USD 探针确认导航 `+X` 与导入模型 root 相差 `pi`、导航角速度需在轮子边界
  反号，相关转换已集中到 `warehouse_vln/kinematics.py` 并由单元测试锁定。
- Agent 增加 `--navigation-scene hospital|warehouse` 和
  `WarehouseVlnAdapter`。Warehouse 当前只开放区域语义导航；物体预抓取未实现时在启动
  Isaac 前阻断，不会错误复用 Hospital 坐标。
- 增加 `docs/WAREHOUSE_G1D_NAV.md`，记录场景选择、命令、正式
  LingBot/SAM3 替换流程、轮子证据及物理 G1-D 的 ROS 2/安全前置条件；同步更新
  `MOBILEMANIBENCH_SETUP.md`、`vlaandvln.md` 和维护入口。

验证：

- 主 Isaac Sim 6.0.1 实际组合场景成功：8,139 prim、1,878 mesh、
  1,878 collision prim，范围约 `x=[-12,12] m`、`y=[-18,20.818] m`。
- “请带我到东侧货架通道”加载项目 `Assets/g1_d_robot/g1_d.usd` 后成功：
  3 个 waypoint、22.029 m、1,936 帧，位置误差 0.149 m、航向误差 0.146 rad；
  65 帧第三人称 GIF 末帧已目视确认机器人位于货架通道。
- Warehouse 单元测试 6/6、Agent 测试 27/27、Python 编译和 Shell 语法通过；
  Agent plan-only 正确生成 Warehouse VLN 步骤；随后经 Agent `--execute` 再次启动
  Isaac 并以相同误差成功完成任务。
- 既有 Hospital 18/18、LingBot 语义导航 21/21 轻量回归通过。
- 修正后的 300 帧物理探针中，直行和正向转弯方向正确；长路线仍未完成。

已知限制：

- 通过的完整导航仍是 `stable_assisted`，不能表述为纯轮地接触或物理真机验收。
- Warehouse 当前成功制品是 Isaac collision bootstrap；正式 LingBot RGB-only
  occupancy、SAM3 语义地点审核和实体机 ROS 2/Nav2 驱动尚未完成。
- 远端 Warehouse URL 需要网络或本地缓存；完整 MobileManiBench 官方 G1/YCB 资产仍
  缺失，`doctor` 为 12/15。

### VLA 现场启动门与物体—技能站位配置

- Agent 从一次性 `VLN/VLA` 路由升级为两层监督：高层仍决定能力序列，操作前新增
  `VlaReadinessGate`，只有实时现场条件全部满足才把控制权交给 VLA。
- 新增严格的“物体 + 技能”交互配置库。当前 `red_cube_demo + pick` 使用
  provisional 0.80 m 推荐距离、0.65–0.90 m 允许区间，并定义横向/朝向、检测稳定性、
  必需相机、观测时效、底盘停止阈值和抬升保持成功判据。
- 现有 `hospital-object-docking` 增加显式 `--standoff` 覆盖；Agent 从审核 profile
  注入推荐值，用户自由文本距离不能绕过交互区间。缺少唯一 profile 时在启动子进程前
  `blocked`。
- 新增版本化 `ObjectObservation` 和门控检查：环境、object ID、`base_link` frame、
  可见性/置信度/相机、稳定帧/时效/不确定度、碰撞、底盘速度、距离、横向/朝向与右臂
  IK。
- 每类失败生成具体恢复动作；`ReadinessRecoveryController` 最多执行三次有界
  重识别/等待/停车/靠近/后退/对齐/换站位，每次重新观测。碰撞和配置不匹配直接阻断。
- VLA integration backend 可选实现 `observe_readiness` 和 `recover_readiness`；
  CLI 自动装配到同一会话监督器。静态 observation JSON 只用于验证 contract，并被禁止
  与已启用的 VLA backend 同用。
- 更新 `vlaandvln.md` 和 `AGENTS.md`，写清实时 provider、距离标定、同一
  SimulationApp 接管以及当前未验收边界。

验证：

- G1-D Agent 测试 26/26 通过；覆盖交互 profile/技能解析、standoff 强制覆盖、缺配置
  预阻断、全部门控分支、有限恢复重检查、碰撞不可恢复和 backend 可选 hook。
- Hospital 轻量测试 18/18、LingBot 语义导航测试 21/21 通过。
- Python 编译、三个 JSON schema 实例解析和 Shell 语法通过；plan-only 不加载缺失的
  VLA config。
- CLI 实测未知杯子 profile 和“门控通过但 VLA 未交付”均返回 `blocked` / 退出码 3，
  未启动 Isaac 或伪造操作成功。

已知限制：

- 本次未运行新的大型 Isaac 仿真；没有实时 SAM3/RGB-D/TF、右腕相机、IK、碰撞或局部
  底盘恢复 controller。
- 红色方块距离区间为 provisional 接口初值，必须通过右臂无接触预抓取和 VLA 实测后
  才能升级为 `sim_validated`，不能用于物理真机。

### G1-D VLN/VLA 任务 Agent 与 VLA 接口占位

- 新增任务级 `G1DTaskAgent` 和可审计 planner，把自然语言分解为纯 VLN、纯 VLA 或
  `VLN -> VLA`；含义不明确时拒绝执行，前置步骤失败时跳过后续操作。
- VLN adapter 只调用已经存在的 `hospital-vln` 与 `hospital-object-docking`，没有引入
  第二套导航、坐标生成或地点解析逻辑；DeepSeek 仍只在现有 Hospital VLN 中选择审核
  `place_id`。
- 新增动态加载的 VLA backend 协议、禁用配置模板和 G1-D 右臂/多指手动作顺序。权重未
  交付时操作阶段返回 `blocked`，不会把预抓取停靠误报为抓取成功。
- Agent 把前序 VLN 结果、docking plan 和 run summary 作为 handoff context 传给 VLA；
  `vlaandvln.md` 进一步规定最终移动操作必须在同一 Isaac 会话中完成状态接管。
- `mobilemanibench.sh agent` 默认只生成计划，只有显式 `--execute` 才启动现有
  Isaac/VLA adapter；任务结果以版本化 JSON 写入被忽略的 `outputs/g1d_agent/`。
- 新增 `vlaandvln.md`，说明既有 LingBot/SAM3/语义地点库/DeepSeek/Nav2 链路、Agent
  构造、VLA 交付清单、Isaac 6.0.1 接入步骤和物理 G1-D 的 sim-to-real 安全边界。

验证：

- G1-D Agent 单元测试 9/9 通过，覆盖三种路由、含义不明拒绝、VLA 未就绪阻塞、
  导航失败阻止操作、adapter 异常结构化失败、阶段 context 传递、既有 VLN 入口委托和
  根工作区相对路径解析。
- 既有 Hospital 轻量测试 18/18 通过；Python 编译、Shell 语法和两个
  `mobilemanibench.sh agent` 规划示例通过。

已知限制：

- 尚未执行新的大型 Isaac 仿真；本次验证了 Agent 编排和现有 VLN 回归，没有声称 VLA
  抓取、同一 SimulationApp 连续接管、纯轮地接触底盘或物理真机已经通过。
- 外部 VLA 团队仍需交付权重、推理环境、观测预处理、G1-D 动作映射和成功判据。

## 2026-07-24

### Hospital 移动操作桌面与动态方块场景

- 将隔离的物体停靠 demo 从视觉台座升级为可用于后续抓取的四腿桌：桌面和四条桌腿均为
  静态 USD/PhysX 碰撞体，不修改正式 Hospital 场景资产、地点库、地图或 6006 dashboard。
- 红色方块升级为带碰撞的动态刚体，边长 `0.2 m`、质量 `0.25 kg`，初始底面与
  `z=0.95 m` 桌面齐平。
- 新增无 Isaac 依赖的确定性场景布局模块和测试；运行摘要记录桌面几何、方块物理属性、
  实际世界坐标和碰撞 API 状态，方便后续抓取验收。

验证：

- 场景布局、物体停靠和统一 dashboard 相关轻量测试 10/10 通过；Python 编译、
  JSON 解析和 `git diff --check` 通过。
- 主 Isaac Sim 6.0.1 headless 真实运行 120 帧：桌子包含 5 个碰撞件，方块具备
  Collision/RigidBody/Mass API；最终世界坐标
  `(-2.500, 0.200, 1.050)`，未穿透或跌落。
- 生成 10 帧第三人称预览并目视确认 Hospital、G1-D、桌子和红色方块同时可见。

已知限制：

- 当前仅完成场景和物理物体搭建；尚未实现右臂 IK、手指闭合、抓取接触判据、抬升或
  送达阶段，不能声称移动操作闭环成功。
- 导航仍使用既有 `stable_assisted`，方块位姿仍来自 demo catalog；纯轮地接触与
  RGB 物体检测限制不变。

## 2026-07-23

### 6009 合并区域语义导航与物体精确停靠

- 6009 从单一物体停靠升级为统一自然语言入口：命中物体目录时解析 object/standoff，
  否则复用 6006 的 `HospitalIntentResolver` 和正式地点库执行受约束区域导航；不修改
  6006 服务。
- 前端增加任务类型、语言解析结果、统一目标位姿和区域指令快捷项；两类任务共用实时
  chase camera、LingBot/occupancy 地图、规划路径和实际轨迹。
- 任务状态显式记录 `semantic_region_navigation` 或 `object_relative_docking`；
  区域导航使用正式地点 docking pose，物体导航继续使用 footprint/occupancy 验证后的
  参数化 SE(2) 位姿和 3 cm 阈值。

验证：

- 真实 DeepSeek 将“我累了，带我去坐下”解析为 `waiting_area`，置信度 0.90；
  从 6009 API 启动 Isaac 后 1092 帧成功，路径 7.376 m、位置误差 0.119 m、
  朝向误差 0.117 rad，MJPEG 实时流正常。
- 统一路由单元测试覆盖同一页面的模糊区域指令和物体距离指令；相关轻量回归通过。

### 物体级精确停靠实时控制台

- 新增独立 `hospital-object-web` / 6009 控制台；浏览器可提交包含不同物体和距离的
  自然语言停靠指令，后端每次重新解析约束、检查 footprint/occupancy/可达性并启动
  Isaac Sim，不再只是展示既有 GIF。
- 页面通过 TCP MJPEG 与 live state 实时显示 chase camera、LingBot RGB 点云、
  occupancy map、规划路径、实际轨迹、物体、停靠点和误差遥测；可停止运行中的任务。
- 新增显式场景 profile 注册表。当前只启用已实现并验证的 `hospital_demo` runner；
  其他场景必须补齐仿真加载、坐标系、物体生成、地图和实时发布后才能开放。
- 6009 输出隔离到 `outputs/hospital_object_docking_web/`，不覆盖既有 0.8 m 制品，
  也不修改或停止 6006 正式 Hospital dashboard。

验证：

- 物体停靠、实时控制台和既有 Hospital live dashboard 相关轻量测试 9/9 通过；
  Python/JavaScript/shell 语法检查和 `git diff --check` 通过。
- 从 6009 API 真实提交“请停到红色方块前0.6米”：重新计算停靠位姿
  `(-2.500,-0.400,1.571)`，Isaac 622 帧成功；实际基座—物体距离 0.611 m、
  位置误差 0.030 m、朝向误差 0.049 rad；MJPEG 实时端点返回连续画面。

### Hospital 物体级参数化精确停靠

- 新增独立 `hospital-object-docking` 入口、demo-only 物体目录和无 Isaac 依赖的停靠
  模块；支持“方块前 0.6/0.8 米”等指令，按物体交互面计算带朝向的 SE(2) 终点。
- 终点在正式 LingBot occupancy map 上检查机器人 footprint 和路径可达性；过近距离会按
  机器人半径与物体尺寸拒绝，而不是直接把任意 LLM 坐标交给控制器。
- 独立 demo 显式使用 0.03 m 位置和 0.05 rad 朝向阈值；原 Hospital 默认仍为
  0.12/0.12，正式地点库、地图、6006 dashboard 和既有输出目录均未改变。
- Isaac 场景仅在 opt-in demo 中生成台座、方块、停靠圆盘和 standoff 线；输出写入
  `outputs/hospital_object_docking/`，包括计划、运行摘要、头部 RGB、live state 和 GIF。

验证：

- `hospital_vln.tests.test_object_docking` 4/4 通过：覆盖距离解析、对象解析、安全下限、
  精确相对位姿和正式地图可达性。
- Isaac Sim 6.0.1 headless 真实运行“请停到红色方块前0.8米”成功：619 帧、路径
  2.657 m、停靠点位置误差 0.030 m、航向误差 0.050 rad，实际基座到方块中心
  0.809 m；G1-D 头部 RGB 与 52 帧第三人称 GIF 已生成并目视检查。
- USD 运行时包围盒确认方块位于预期世界坐标；`--plan-only` 不启动 Isaac 即可复算
  距离约束和路径。

已知限制：

- 当前方块是 demo-only 视觉几何，不带刚体/接触/抓取成功判据；本次只验收到
  `PREGRASP_DOCK`，没有声称 OpenVLA 已完成抓取。
- 精度来自 `stable_assisted`；纯轮地接触底盘仍沿用既有失败证据，接 VLA 前还需要
  物体检测/跟踪和近距离视觉伺服。

### Hospital 候诊区多候选停靠（进行中）

- 新增依赖轻量的候选生成/检查/排序模块和独立构建脚本，输出固定写入
  `outputs/hospital_docking/`，不修改正式地点库、LingBot 点云或 occupancy map。
- 从 Hospital USD 实测的 4 个 `SM_Chair_02a` 长椅边界生成南/北共 8 个候选；每项记录
  footprint、occupancy、可达性、clearance、路径长度、朝向误差、评分和拒绝原因。
- 首次构建保留 3/8 个候选，选择路径较短的
  `chair_02a6_south=(-5.117,-0.878,1.571)`；另外 5 个因占用/未知或 footprint clearance
  不足被拒绝。
- 选择器接受显式 `blocked_candidate_ids`，为后续人员/动态障碍输入预留接口；当前不伪造
  动态感知结果。
- 新增 `hospital-docking` 构建命令；dashboard 只有显式传入 `--dynamic-docking` 才加载
  隔离候选，并把选定 pose 作为实验参数交给 Isaac。默认模式继续使用正式固定点。

验证：

- 构建前后 `map.pgm`、`places_formal.json`、`rgb_pointcloud.png` 的 SHA-256 未变化。
- `hospital_vln/docking.py` 和 `scripts/build_hospital_docking.py` 编译通过，
  `git diff --check` 通过。
- Hospital 测试 8/8、`lingbot_semantic_nav` 测试 21/21；覆盖 occupancy 拒绝、
  动态阻塞切换、默认固定点和 opt-in 动态点。

未完成：

- 尚未执行隔离的 Isaac 动态停靠 demo；默认 6006 dashboard 仍运行原固定停靠点。

### Hospital DeepSeek 模糊地点理解

- Hospital dashboard 从精确别名解析升级为 DeepSeek 结构化意图解析；模型只能在正式
  地点库中选择 `place_id`，不能生成坐标、路径或未审核地点。
- 正式 Hospital 地点条目增加功能描述和典型自然语言请求，并迁移到严格 schema-v2
  docking checks/review 格式，可被统一 `PlaceDatabase` 验证。
- DeepSeek 系统提示明确使用地点 metadata 做目的/活动/服务语义归一，例如将“找个能
  坐着等医生的地方”映射到提供等待就诊功能的地点。
- dashboard 将已校验的 `--target-id` 交给 Isaac，保留原始自然语言用于日志和页面；
  避免模拟器再次用字符串规则解析模糊原句。
- 页面新增“语言理解”遥测，显示解析器、目标和置信度；默认输入和快捷按钮改为不含
  地点名的模糊表达。

验证：

- 真实 DeepSeek 单独解析“带我去找个能坐着等医生的地方”为 `waiting_area`，首次
  置信度 0.95；返回坐标来自正式地点库而非模型。
- 从 dashboard 再次提交同一模糊指令，DeepSeek 返回 `waiting_area`、置信度 1.00；
  Isaac Sim 6.0.1 完成 1092 帧、7.376 m 导航，位置误差 0.119 m，航向误差
  0.117 rad。
- Hospital 测试在系统 Python 和主 Isaac Python 下均为 6/6；`lingbot_semantic_nav`
  测试 21/21 通过。

已知限制：

- 当前正式地点仍只有 `reception` 和 `waiting_area`；模糊理解不能弥补地点库缺失。
- DeepSeek API 失败时可回退到精确别名，但无法离线理解新的功能表达；严格 demo 可用
  `--no-rule-fallback` 禁止回退。
- 语言理解只扩展 VLN 目标选择，不代表“拿水”等抓取动作已经实现。

### Hospital 指令驱动三视图实时 dashboard

- 新增 `hospital-web` 命令和 `scripts/serve_hospital_dashboard.py`，提供纯 HTTP/TCP
  指令接口、任务状态接口、地图资源和 Isaac chase-camera MJPEG 流。
- 新增 `hospital_dashboard/` 响应式页面；输入已审核地点指令后，可同时观察 Isaac Sim
  运动画面、LingBot RGB 点云俯视图和 occupancy map。两张地图实时叠加规划路径、实际
  轨迹、机器人位置与朝向。
- `run_g1d_hospital_vln.py` 新增原子化 live state/JPEG 发布；无 Isaac 依赖的发布逻辑
  拆到 `hospital_vln/live.py`，便于轻量回归。
- dashboard 在启动前验证正式地图、地点库和预览资产，拒绝与其他 Isaac Kit 并发；
  支持停止当前任务、子进程异常状态和可迁移的地图资产路径。
- 更新 `docs/HOSPITAL_SEMANTIC_NAV.md`，记录启动、TCP 暴露、画面语义、运行输出和
  `stable_assisted` 限制。

验证：

- Hospital 轻量测试 5/5 通过，其中新增 live state 原子写入/轨迹采样、失败状态和
  dashboard 正式地图规划/可迁移资产测试。
- `python3 -m py_compile`、`node --check hospital_dashboard/app.js` 和
  `bash -n mobilemanibench.sh` 通过。
- 实际从 dashboard 提交“请带我到候诊区”，主 Isaac Sim 6.0.1 成功运行 1092 帧；
  总路径 7.376 m，位置误差 0.119 m，航向误差 0.117 rad，进程返回成功。
- 目视确认最终 chase camera 中 G1-D 停靠在候诊椅前；HTTP 页面、RGB 点云、
  occupancy 资源和 MJPEG 流均实际读取成功。

已知限制：

- dashboard 的点云和 occupancy 是离线建图底图，实时部分是机器人位姿和轨迹；导航时
  不会持续重建点云。
- 当前导航仍为 `stable_assisted`；纯轮地接触失败证据和 P0 调优任务不变。
- TCP dashboard 绕过了当前 WebRTC UDP 不可达问题，但不修复原生 WebRTC 网络条件。

### Isaac Sim 6.0.1 Streaming 网络诊断

- 新增 `docs/ISAAC_SIM_STREAMING.md`，记录日志位置、extension 版本、端口职责、检查命令
  和获得 UDP endpoint 后的正确启动方式。
- 确认 `omni.kit.livestream.app 10.1.1`、`webrtc 10.3.2`、`core 10.2.1`
  均已启动，Kit 日志明确报告 49100/TCP 和 47998/UDP 配置生效。
- 确认 RTX viewport 持续渲染，系统提供 NVENC/解码库，故问题不在 GPU 驱动或编码器。
- 确认当前 AutoDL 容器只有私网地址，平台自定义服务/SSH 隧道没有浏览器可达的
  47998/UDP；这解释了页面正常、TCP 信令存在但媒体始终等待的现象。
- 暂不重启或修改驱动：在没有 UDP 映射、覆盖网络或 TURN relay 时，重复启动不会改变
  结果。

### 长期维护与 Git 基线

- 根据工作区代码、任务清单、运行摘要和本机命令建立 `AGENTS.md`、`TODO.md` 与本文件。
- 记录当前 RTX PRO 6000 Blackwell、Ubuntu 22.04、Isaac Sim 6.0.1-rc.7、
  MobileManiBench Python 3.10 和 LingBot-Map PyTorch 2.8 环境。
- 明确旧文档中的 Isaac Sim 4.5 / RTX 4090 信息属于历史环境，防止后续对话混用运行时。
- 增加根目录 `.gitignore`，排除 Isaac Sim 安装、Python 环境、权重、下载、USD/网格、
  输出、缓存、日志和凭据。
- 将已有 MobileManiBench 仓库登记为根项目子模块，以便根提交记录准确的上游代码状态，
  同时保留其独立历史。
- 制定重要步骤的固定收尾流程：测试、更新 TODO、更新 CHANGELOG、检查暂存区、提交。

验证：

- 从 `outputs/simple_room_vln/run_summary.json` 和
  `outputs/hospital_vln/run_summary.json` 核对两个 assisted 导航成功基线。
- 从 `outputs/hospital_vln/physics_probe/run_summary.json` 核对纯轮地物理模式失败：
  300 帧、位置误差 4.941 m。
- 用 `nvidia-smi`、各环境 Python 和 `isaacsim/VERSION` 核对当前运行环境。
- Hospital 轻量测试 2/2 通过，`lingbot_semantic_nav` 单元测试 21/21 通过，
  G1-D 工具脚本语法检查通过。
- `isaacsim-web-client` 生产构建通过；仅有单个 bundle 大于 500 kB 的 Vite 性能提示。
- 根仓库基线提交：`0896392`；MobileManiBench 子模块提交：`28422ba`。

已知限制：

- 本次只新增维护文档、忽略规则和版本控制元数据，不修改核心代码逻辑。
- 大型运行输出被有意排除，文档内指标是 2026-07-23 的最近已知基线。
- 纯轮地接触模式仍未通过；当前成功结果只证明高层语言—地图—控制链路。
