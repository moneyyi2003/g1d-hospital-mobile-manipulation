# 当前任务与交接状态

更新时间：2026-07-28（UTC）

## 当前结论

Hospital 语义导航 MVP 已在确定性 `stable_assisted` 模式通过，当前主线应转向
G1-D 纯轮地接触物理控制的稳定性与回归；完成后再推进移动操作任务。根目录长期维护
机制已经建立。Hospital TCP dashboard 已能在浏览器同步显示 Isaac chase camera、
LingBot RGB 点云和 occupancy map 上的实时机器人轨迹，并已接入 DeepSeek 模糊地点
理解，不依赖 WebRTC UDP。2026-07-23 的 WebRTC 排查仍确认：当前 AutoDL 公有云实例
没有浏览器可达的 47998/UDP 媒体路径；该外部网络条件解决前，原生 WebRTC 页面仍会
停在 `WAITING FOR STREAM`。任务级 G1-D Agent 已能把指令安全分解为 VLN、VLA 或
VLN → VLA，并复用既有 Hospital 导航；VLA 仍等待外部团队交付，不属于当前已验收能力。

## 已完成并验证

- [x] 下载并安装当前 Isaac Sim 6.0.1-rc.7 standalone，Python 3.12.13 可用。
- [x] 建立 MobileManiBench Python 3.10.20 独立环境及统一入口
  `mobilemanibench.sh`。
- [x] G1-D URDF 已转换为分层 USD，并完成关节/刚体基本加载工具。
- [x] SimpleRoom 中英文地点解析、路径规划和确定性导航演示成功。
- [x] Hospital 前台/候诊区 ROI 加载和 G1-D 头部 RGB 巡检成功。
- [x] LingBot-Map RGB-only 推理、米制对齐、彩色点云和 ROS occupancy map 已生成。
- [x] 正式地点库已审核开放 `reception` 和 `waiting_area`。
- [x] Hospital 指令“请带我到候诊区”端到端导航成功：
  1092 帧、路径 7.376 m、位置误差 0.119 m、航向误差 0.117 rad。
- [x] 增加 Hospital 无界面验收命令和 GUI/WebRTC 演示入口。
- [x] 盘点当前硬件、系统、Python 和 Isaac Sim 版本，识别旧任务文档中的环境漂移。
- [x] 建立根仓库忽略策略，排除约数十 GB 的运行时、环境、权重、资产和生成物。
- [x] 定位 Isaac Sim Streaming 日志，确认 livestream app/webrtc/core extensions 正常
  加载，RTX 与 NVENC 库可用。
- [x] 确认 49100/TCP 正常，47998/UDP 媒体协商未建立；根因是 AutoDL 无独立公网 IP，
  自定义服务及 SSH 隧道仅提供 TCP/HTTP 路径。
- [x] 增加 `hospital-web` 三视图控制台：浏览器输入正式地点指令后，同步显示 Isaac
  chase camera、LingBot RGB 点云和 occupancy map 上的规划路径与实际轨迹。
- [x] 实机通过 dashboard 执行“请带我到候诊区”：约 57 秒结束，1092 帧，位置误差
  0.119 m，航向误差 0.117 rad；HTTP 地图资源和 MJPEG 流均读取成功。
- [x] Hospital dashboard 接入受地点库约束的 DeepSeek 解析；地点库增加可信功能描述，
  LLM 只能返回审核 `place_id`，坐标仍由地点库提供。
- [x] 模糊指令“带我去找个能坐着等医生的地方”真实 DeepSeek + Isaac demo 成功：
  未出现地点名，解析为 `waiting_area`（置信度 1.00），1092 帧到达，位置误差 0.119 m。
- [x] 建立隔离的候诊区多候选生成器：4 个真实 `SM_Chair_02a` 长椅实例生成 8 个候选，
  occupancy/footprint 自动淘汰 5 个，3 个可达；当前排序选择
  `chair_02a6_south=(-5.117,-0.878,1.571)`，输出仅在 `outputs/hospital_docking/`。
- [x] 增加显式 `--dynamic-docking` dashboard 接口和 `hospital-docking` 构建命令；
  默认仍为 `formal_fixed_pose`，动态模式才把验证候选交给 Isaac。
- [x] 新增完全隔离的物体级精确停靠 demo：把“方块前 0.8 米”转换为面向物体的 SE(2)
  位姿，并用正式 occupancy map 验证 footprint 与路径；Isaac 实测位置误差 0.030 m、
  朝向误差 0.050 rad，实际基座—方块中心距离 0.809 m；输出仅写入
  `outputs/hospital_object_docking/`，6006 默认 demo 未改变。
- [x] 新增独立 6009 统一实时控制台：同一输入框自动分流“受审核区域语义导航”和
  “物体/距离精确停靠”，通过 TCP MJPEG 和 live state 显示相机与轨迹；物体 `0.6 m`
  指令真实运行 622 帧成功（实际距离 0.611 m、位置误差 0.030 m），模糊指令
  “我累了，带我去坐下”由 DeepSeek 解析到候诊区并真实运行 1092 帧成功（位置误差
  0.119 m），输出与 6006 隔离。
- [x] 在隔离的 Hospital 物体停靠场景中搭建四腿碰撞桌和动态红色方块：桌面
  `1.0 x 0.7 x 0.08 m`、5 个静态碰撞件，方块边长 `0.2 m`、质量 `0.25 kg`；
  Isaac Sim 6.0.1 运行 120 帧后方块稳定在桌面，第三人称预览已目视确认。
- [x] 新增 G1-D 任务级 Agent：保守路由 `VLN`、`VLA`、`VLN -> VLA`，导航阶段只调用
  既有 `hospital-vln` / `hospital-object-docking`；VLA 提供外部 backend 配置和
  handoff context 插槽，未接入时明确 `blocked`。
- [x] 新增 `vlaandvln.md`，记录现有 LingBot/SAM3/语义数据库/DeepSeek 导航链、Agent
  状态机、VLA 交付接口、Isaac 同会话接管要求与真机 sim-to-real 安全步骤。
- [x] Agent 增加严格的“物体 + 技能”交互配置库；预抓取距离由配置显式交给既有
  `hospital-object-docking`，当前红色方块 `pick` 使用 provisional 0.80 m 推荐值和
  0.65–0.90 m 允许区间。
- [x] Agent 增加 `VlaReadinessGate` 和有限恢复协议：VLA 前检查实时检测、相机、观测
  时效、底盘停止、距离/横向/朝向、右臂 IK 和碰撞；失败时输出明确恢复动作，碰撞和
  配置错误不自动重试。
- [x] VLA 插件支持可选 `observe_readiness` / `recover_readiness` 同会话 hook；提供
  版本化静态观测 contract 示例，只用于接口测试，不冒充真实感知。

## 当前问题

- [ ] **P0：Isaac Sim WebRTC 缺少 UDP 可达路径。** 需要由运行平台提供
  47998/UDP 映射、UDP 覆盖网络或外部 TURN relay；仅重启 Kit、设置 HTTP 代理域名或
  转发 49100/TCP 无法显示原生 Streaming 视频。三视图演示可先使用 TCP-only
  `hospital-web`；原生 WebRTC 限制详见 `docs/ISAAC_SIM_STREAMING.md`。
- [ ] **P0：纯轮地接触导航尚未通过。** 当前 300 帧 physics probe 失败，
  位置误差 4.941 m；需要检查轮轴方向、驱动符号、摩擦、质量/惯量、力矩上限和底盘稳定性。
- [ ] **P0：两套 Isaac 依赖链并存。** 主 standalone 是 6.0.1/Python 3.12，
  MobileManiBench 环境仍是 Python 3.10/PyTorch 2.5.1+cu121 的早期兼容链；运行命令必须
  明确选择，后续需决定是否统一迁移。
- [ ] **P1：正式 Hospital 覆盖仅限前台和候诊区。** 主走廊及全院墙体 occupancy 尚未
  审核，不应开放为语言目标。
- [ ] **P1：候诊区仍使用单一固定停靠点。** 当前
  `waiting_area_reviewed_v1=(-5.95, 2.20, -1.571)` 已审核且 demo 稳定，但不会根据
  多把椅子、路径长度或动态占用选择不同停靠位置。下一步必须以独立制品和显式 opt-in
  实现多候选生成/排序，默认 `hospital-web` 不得改变。
- [ ] **P1：移动操作尚未闭环。** G1-D 右臂、多指手、IK、接触参数和抓取成功判据仍需
  单独配置，不能复用官方 G1 平行夹爪动作空间。
- [ ] **P1：VLA backend 尚未交付。** 当前 Agent 的 VLA 插槽会 fail-closed 为
  `blocked`；需要 VLA 团队提供权重、预处理、相机协议、G1-D 动作映射、依赖环境和成功
  判据，并在 Hospital runner 到达后保持同一 Isaac SimulationApp 完成连续操作。
- [ ] **P1：VLA 启动门尚无实时 provider。** `interaction_profiles.json` 当前仅有
  provisional 红色方块拿取配置；仍需在同一 Isaac 会话接入头部/右腕相机、SAM3 +
  metric depth/TF、底盘速度、右臂 IK 与碰撞结果，并用实测标定距离区间。
- [ ] **P1：物体精确停靠仍依赖已知物体位姿和 assisted 控制。** demo 方块现已是带
  碰撞和质量的动态刚体，但还没有抓取/抬升控制与成功判据；后续仍需 RGB 物体检测/跟踪
  和末端视觉伺服。当前结果不能表述为纯轮地接触或 OpenVLA 抓取验收。
- [ ] **P1：物体停靠实时控制台当前只启用 Hospital runner。** 场景 profile 和前端切换
  接口已经存在，但 SimpleRoom 或其他场景仍需各自的 USD 加载、坐标系/起点、物体生成、
  地图与 live publisher 验证；不能把仅有 profile 名称视为多场景运行成功。
- [ ] **P1：MobileManiBench 官方 G1/YCB smoke 的最新资产状态需复核。** 旧
  `task.md` 中“Assets.zip 下载中”的记录可能已过时，应以 `doctor`、ZIP 校验和实际
  reset/step 返回码重新验收。
- [ ] 生成输出不进入 Git；若运行结果需要长期保存，应记录小型 JSON 指标或建立外部制品
  存储，而不是提交 GIF、点云、地图和模型。

## 下一步执行计划

### 0. Hospital 三视图演示使用方式

- 运行 `./mobilemanibench.sh hospital-web --host 0.0.0.0 --port 6006`，通过 AutoDL
  HTTP 自定义服务或 SSH TCP 隧道访问。
- 输入“带我去找个能坐着等医生的地方”或“我想找工作人员问点事情”；DeepSeek 从审核
  地点库选择 ID，地图底图是离线 LingBot 结果，机器人位姿、规划路径和轨迹为实时叠加。
- 启动任务前必须停止其他 Isaac Kit；dashboard 会主动拒绝并发 Kit。
- 详细接口、输出和限制见 `docs/HOSPITAL_SEMANTIC_NAV.md` 第 5 节。

### 0.1 候诊区多候选停靠（当前进行中）

- 保留现有 `places_formal.json`、LingBot 点云、occupancy map、6006 dashboard 和
  `waiting_area_reviewed_v1`；不得覆盖当前已验收 demo。
- [x] 从 Hospital 椅子实例边界生成多个面向座椅的停靠候选，输出到新的隔离目录
  `outputs/hospital_docking/`。
- [x] 每个候选检查 map 边界、机器人 footprint clearance、从起点可达性和最终朝向；
  记录失败原因，不能只保留成功项。
- [x] 排序考虑可达性、clearance、路径长度和朝向；动态占用已预留
  `blocked_candidate_ids` 输入接口，不伪造
  人员/障碍检测结果。
- [x] 只通过显式 `--dynamic-docking` 或独立 demo 入口启用；默认 `hospital-web` 继续
  使用当前固定审核点。
- [ ] 验收：轻量测试已覆盖候选过滤、动态阻塞切换和默认/opt-in 隔离；还需在隔离输出下
  完成一次真实 Isaac 导航；
  再次确认默认 dashboard 的目标坐标和既有 0.119 m 基线未被修改。

### 1. 恢复 Streaming 网络可达性（外部前置条件）

- 获取一个浏览器可通过 49100/TCP 和 47998/UDP 直达的服务器/覆盖网络 IP，或准备
  可用 TURN relay。
- 用 `docs/ISAAC_SIM_STREAMING.md` 中的显式 `publicIp` 命令重启。
- 验收浏览器 `Stream Ready`、ETLI candidate pair 和实际 UDP 流量。

### 2. 纯轮地接触最小诊断

- 固定空旷平面和短时限，分别施加左轮、右轮、同向和反向速度命令。
- 每种命令记录 wheel joint 实际速度、base 位姿、接触力和是否出现滑移/倾倒。
- 核对右轮符号约定、轮半径 0.0848 m、轮距 0.4062 m 与 USD joint axis。
- 验收：机器人能直行和原地转向，状态有限且无 NaN；测试脚本与小型 JSON 摘要可提交，
  大型日志和视频不提交。

### 3. 物理参数调优与回归

- 调整轮地材质摩擦、wheel damping/effort、底盘质心和非底盘关节保持增益。
- 把成功参数固化为明确配置，不隐藏在运行时魔数中。
- 用 `--wheel-physics-only` 重跑前台短路径，再跑候诊区完整路径。
- 验收：至少连续 3 次固定种子成功，位置误差不超过 0.20 m，并保留失败时诊断信息。

### 4. 扩展 Hospital 与移动操作

- 审核主走廊 occupancy、连通性和 docking pose 后再增加地点。
- **唯一近期下一步：**基于现有碰撞桌和动态方块，完成 G1-D 右臂关节映射与一个不接触
  物体的预抓取位姿，使右手稳定到达方块侧面且不碰撞桌面；把该 IK/碰撞结果接入
  `ObjectObservationProvider`，据实标定并更新 provisional 距离区间。
- 再完成右手闭合、接触判定和抬升至少 5 cm，然后替换为 YCB 物体。
- 最终组合 `NAVIGATE -> ALIGN -> GRASP -> LIFT -> SUCCESS/FAIL` 单回合流程。
- VLA 交付后按 `vlaandvln.md` 接入 `g1d_agent.vla_backend_v1`，先单独验收观察/动作
  schema，再启用同一 Isaac 会话内的 `VLN -> VLA` 连续执行。

## 本次维护机制验收

- [x] 根据实际项目和运行输出编写 `AGENTS.md`。
- [x] 根据当前真实进度编写本文件。
- [x] 初始化 `CHANGELOG.md`。
- [x] 创建 `.gitignore`，禁止提交大文件和凭据。
- [x] 初始化根 Git 仓库并提交当前可维护状态；基线 commit 为 `0896392`。
- [x] 基线提交后已复核暂存文件范围；最终维护记录提交后再次确认工作区状态。

## 每次工作结束前

- [ ] 更新本文件的完成项、问题证据和下一步。
- [ ] 更新 `CHANGELOG.md`。
- [ ] 执行相关测试及 `git diff --check`。
- [ ] 检查没有权重、资产、输出、日志或凭据进入暂存区。
- [ ] `git commit` 保存可恢复状态。
