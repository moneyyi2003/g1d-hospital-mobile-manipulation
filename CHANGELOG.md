# 重要修改记录

本文件记录能够影响复现、行为、接口或任务状态的重要变更。日期使用 UTC；生成物刷新和
无行为影响的小改动不单独记录。

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
