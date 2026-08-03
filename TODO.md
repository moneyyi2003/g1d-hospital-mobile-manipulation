# 当前任务与交接状态

更新时间：2026-08-03（UTC）

## 当前结论

Hospital 语义导航已经通过；多货架 Warehouse 已完成 G1-D RGB 巡检、LingBot
RGB-only 建图、SAM3.1 语义投影、正式地点审核和 occupancy 替换。东侧货架正式定向路线
也已在 `--wheel-physics-only` 下连续三次通过。面向最终家庭任务的新多区域家庭场景已
完成过 G1-D RGB 巡检、LingBot RGB-only 点云/occupancy、SAM3.1 投影、region 和正式
网页。现已加入不带类别语义的 ReplicaCAD 家庭物品，并完成 292 帧 `1280x720`
近距离重巡检和 Florence-2 无类别清单自主发现：1302 条原始检测经跨帧门接受 29 个
标签，包括 `coffee cup`、`mug` 和 `bowl`。LingBot/SAM3 正式流水线得到
170 × 154 occupancy、18 个语义锚点/region、5 个审核对象和 2 个审核地点；杯子三维
锚点来自 7 帧 SAM3 mask 的 RGB 多视角射线三角化，不读取 USD 语义或物体坐标。
同一 Isaac SimulationApp 内的“去餐区—实时找杯—两级停靠—OpenVLA 推理—右手拿取—
携杯返回沙发”已实跑成功。当前唯一近期仿真主线是给这条闭环补充场景/自碰撞检查和
家庭纯轮地接触回归；真机侧已找到官方 Unitree SDK2 G1-D AGV API，
并完成 fail-closed ROS 2 命令适配。下一步必须在机器人旁确认网络、左右轮真实反馈
topic、实体硬急停和 `/scan`，在这些外部条件到位前不得打开硬件输出。
Hospital TCP dashboard 已能在浏览器同步显示 Isaac chase camera、LingBot RGB 点云和
occupancy map 上的实时机器人轨迹，并已接入 DeepSeek 模糊地点理解，不依赖 WebRTC
UDP。2026-07-23 的 WebRTC 排查仍确认：当前 AutoDL 公有云实例没有浏览器可达的
47998/UDP 媒体路径；该外部网络条件解决前，原生 WebRTC 页面仍会停在
`WAITING FOR STREAM`。保留的 v1 Agent 已能做固定顺序路由；新增的并行 v2 目录已提供
对象级共享记忆、五技能动态路由、控制权仲裁和有界重规划。它继续复用现有正式 VLN，
其中家庭实时对象搜索、扫描可见方位停靠、右臂位置 IK、仿真拿取和物理验证已接入同会话
backend。公开 OpenVLA-7B 仍只提供未标定的 BridgeData 动作建议；当前仿真执行使用
审核锚点驱动的有界 IK 和显式 PhysX 固定约束。面向 G1-D 微调/标定的 VLA、场景与
自碰撞 IK、真机手部控制和独立真机操作验证仍未交付。

家庭 6012 控制台现已接入 Dual Brain 复合任务：页面可输入普通地点导航或
“去—拿—返回”，并实时显示 Isaac 第三人称行为、Agent 阶段、轨迹和四层正式地图。
2026-08-03 网页端实跑完成导航、RGB 找杯、两级对齐和 OpenVLA 推理；该次抓取因
约束吸附位移 0.054 m 超过 0.04 m 安全门，重试 IK 误差 0.038 m 超过 0.025 m 门槛而
fail-closed。近期仿真下一步仍是场景/自碰撞 IK 与抓取鲁棒性，而不是降低安全阈值。

网页演示权限已按用户要求分层放宽：正式地点仍保留原审核结论，另生成不覆盖正式制品的
`places_web_demo.json/objects_web_demo.json`。卧室和厨房经正式 occupancy 最近可达自由
栅格验证后临时开放；碗、椅子、长凳等只开放实时搜索，不开放抓取。导航新增正向一致性
门，卧室实跑 526 帧、反向运动帧为 0；完整去餐区—拿杯—回沙发任务再次成功，4 段
导航均 `forward_only_verified=true`。RGB 搜索不再瞬移底盘航向，右臂预抓取、靠近、
闭手、抬升和稳定保持现在连续发布网页画面。

已建立独立、未纳入 Git 的 `vln_vla_expert_handoff_20260803/` 同事交付包：复制 G1-D
完整 USD/URDF 资产、网页 6012 的完整家庭组合场景、正式 LingBot 地图和导航代码，并新增专家数采操作
桌、3 个 3 cm/54 g 红色动态方块、第三视角/右腕相机适配、PD 参数及 action.jsonl
合同。结构检查已在 Isaac Sim 6.0.1 通过。下一协作节点是把同事现有专家脚本放入该
副本环境，先验收一条同步双相机轨迹，再冻结 RLDS 字段和批量随机化范围。

## 已完成并验证

- [x] 家庭 6012 网页升级为自然语言任务控制台，复用正式 VLN，并把复合指令路由到
  同一 Isaac SimulationApp 的 `NAVIGATE -> SEARCH_OBJECT -> APPROACH_AND_ALIGN ->
  OPENVLA_PICK -> VERIFY -> RETURN`；第三人称 MJPEG 和 Agent 状态实时可见。

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
- [x] 新增并行 `g1d_dual_brain_agent/` v2，不删除 v1：Executive 按
  `NAVIGATE -> SEARCH_OBJECT -> APPROACH_AND_ALIGN -> MANIPULATE -> VERIFY`
  动态选择技能；持久化对象记忆和任务 blackboard，互斥仲裁底盘/右臂/右手，并按
  `PATH_BLOCKED`、`OUT_OF_REACH`、`OBJECT_SLIPPED` 等失败原因有界重规划。
- [x] 新 `dual-agent` 通过兼容桥继续调用既有 Hospital/Warehouse/Family Home adapter；
  家庭纯 VLN 明确使用 `home-vln-formal`。新增 Mission JSON、未来 VLA/搜索/对齐/验证
  method 插槽和 14 项轻量测试；缺少 live backend 时 fail-closed。
- [x] 新增 `vlaandvln.md`，记录现有 LingBot/SAM3/语义数据库/DeepSeek 导航链、Agent
  状态机、VLA 交付接口、Isaac 同会话接管要求与真机 sim-to-real 安全步骤。
- [x] 新增公开 OpenVLA-7B 单右臂诊断接入：隔离 Python 3.10 sidecar 读取
  `MANIPULATE` 时刻的实时 G1-D head RGB，按官方 `bridge_orig` 合同返回
  `[dx,dy,dz,droll,dpitch,dyaw,gripper]`；Agent 校验动作并生成右臂交接制品，但不会把
  7 维末端增量误作 7 个 G1-D 关节角。模型未针对 G1-D 微调、动作帧/碰撞 IK/手部映射
  未完成时强制 `execution_permitted=false`。
- [x] 公开 OpenVLA-7B 已在既有家庭 RGB 和同一 Isaac 会话各完成一次真实推理；同会话
  摘要为 `pre_vla_pipeline_succeeded=true`、`openvla_inference_succeeded=true`、
  `same_simulation_app=true`，最终模型推理 1.03 s。最终图中植物仅在右上角部分可见，因此
  新增最终帧“目标可见且未贴边截断”门，当前仍禁止关节执行。
- [x] Agent 增加严格的“物体 + 技能”交互配置库；预抓取距离由配置显式交给既有
  `hospital-object-docking`，当前红色方块 `pick` 使用 provisional 0.80 m 推荐值和
  0.65–0.90 m 允许区间。
- [x] Agent 增加 `VlaReadinessGate` 和有限恢复协议：VLA 前检查实时检测、相机、观测
  时效、底盘停止、距离/横向/朝向、右臂 IK 和碰撞；失败时输出明确恢复动作，碰撞和
  配置错误不自动重试。
- [x] VLA 插件支持可选 `observe_readiness` / `recover_readiness` 同会话 hook；提供
  版本化静态观测 contract 示例，只用于接口测试，不冒充真实感知。
- [x] 审计 MobileManiBench/Isaac 场景清单并选定
  `warehouse_multiple_shelves.usd`：主 Isaac 6.0.1 实测 8,139 prim、1,878 mesh、
  1,878 collision prim，范围约 `24 x 38.8 m`。
- [x] 新增 Warehouse G1-D 巡检/语义导航入口、collision bootstrap 制品、正式
  LingBot map/place 插槽和 Agent `--navigation-scene warehouse`；物体预抓取尚未实现时
  adapter 会 fail-closed。
- [x] Warehouse 指令“请带我到东侧货架通道”真实 Isaac 回归成功：加载项目
  `g1_d_robot/g1_d.usd`，3 个 waypoint、路径 22.029 m、1936 帧、位置误差
  0.149 m、航向误差 0.146 rad；65 帧第三人称 GIF 已目视确认。
- [x] Warehouse 完成 188 帧 `640x360` G1-D RGB 巡检和 32.605 m 覆盖路线；LingBot
  只读 RGB 完成 188 帧推理，pose 仅用于推理后的明确标注米制融合。
- [x] Warehouse 生成 372 x 617、0.05 m/cell 正式 occupancy；SAM3.1 以
  `warehouse shelf` 跟踪 188 帧，548 次检测中 546 条投影到 `map`。
- [x] Warehouse 正式地点审核只开放 `east_shelf_aisle` 和
  `west_shelf_aisle`；`loading_zone` 因 0.75 m 内无安全 free cell 保持拒绝。
- [x] 正式地点增加定向预停靠约束；东侧路线 `(4,8) -> (4,9)` 规划 32.538 m，
  runner 和 ROS 2/Nav2 语言目标节点都按“approach -> destination”执行。
- [x] Warehouse 正式 occupancy 东侧路线纯轮地接触连续三次通过：每次 10,306 帧、
  物理路程 32.516 m、位置误差 0.190 m、朝向误差 0.051 rad，最大 roll/pitch
  0.026/0.147 rad，2 秒制动漂移 0.0086 m，停止线/角速度
  0.0034 m/s、0.0095 rad/s。
- [x] 新增物理 G1-D ROS 2/Nav2 bringup：`map -> odom -> AGV_link`、轮编码器里程计、
  `/scan` 障碍层、制动/锁存急停/硬急停心跳、driver/feedback watchdog 和诊断；默认
  硬件输出关闭且 arm 必定失败，零输出消息级联调通过。
- [x] 新增多区域家庭场景：复用 SimpleRoom 壳体和 SofaTablePlant，增加带碰撞的卧室、
  客厅、餐区、厨房隔墙与家具；四个审核 bootstrap 地点均 footprint-safe 且可达。
- [x] 家庭卧室指令真实 Isaac 导航成功：530 帧、2.378 m、位置误差 0.120 m、航向误差
  0.119 rad；Agent `--navigation-scene home` 可路由纯 VLN，物体预抓取仍 fail-closed。
- [x] 家庭全屋 RGB 巡检成功：14 个路径点、19.285 m、3750 帧，采集 215 张
  `640x360` RGB；最终位置/航向误差 0.119 m/0.119 rad。
- [x] 新增家庭 6012/TCP 实时网页：输入审核地点指令后启动 Isaac，同步显示 G1-D
  960x540 跟随画面、Point Cloud、Semantic、Occupancy、Region、规划路径和实际轨迹。
- [x] 家庭网页端到端验证卧室和餐桌任务：卧室 530 帧/55 个轨迹点，餐桌
  548 帧/73 个轨迹点；状态、MJPEG 和最终画面均通过。该初版验证当时所有地图层均
  显式标为 `BOOTSTRAP`，现已由下述正式四层实现替换。
- [x] 家庭 215 帧 RGB 完成 LingBot 推理；全局 Sim(3) 因仅 22/215 对应点通过阈值而
  被拒绝，使用明确标注的 pose-anchored 推理后融合生成 169 × 153、0.05 m/cell
  occupancy。
- [x] 家庭 SAM3.1 沙发提示产生 87 个检测，经 36 帧漂移门过滤保留 36 条 map-frame
  证据；正式 semantic/region/pointcloud/occupancy 四层和 fail-closed 网页已接入。
- [x] 正式地点审核只批准扫描生成的 `living_room_sofa` 停靠位；正式 Isaac 导航
  523 帧、1.838 m、位置误差 0.119 m、航向误差 0.120 rad，成功。
- [x] 家庭网页首屏改为不等待四层图片解码，地图渐进加载并缓存；新增 G1-D 扫描识别
  报告，明确显示 215 帧 RGB、4 类提示、沙发 87 次原始检测/36 条 map-frame 证据，
  以及床、餐桌、操作台 0 检测和不可导航原因。
- [x] 家庭场景加入 11 个本地 ReplicaCAD 可见碰撞实体；USD prim 仅使用
  `Item01...`，不注册类别语义，资产真值名称和坐标不会进入感知模型。
- [x] 新增 `home-assets` / `home-discover` 和 category-free Florence-2 首阶段：
  只接收 RGB 与 `<OD>` / `<DENSE_REGION_CAPTION>` task token，使用重叠局部视图、
  跨帧一致性、结构/巡检线伪影过滤和描述词归一化，再把模型生成标签交给 SAM3。
- [x] 新物品版本实际重巡检成功：3750 帧、215 张 `640x360` RGB、终点
  0.119 m/0.119 rad；80 个抽样帧的自主发现产生 621 条原始检测，接受 14 类，
  `houseplant` 跨 14 帧、`monitor` 跨 2 帧。巡检和发现制品均声明未提供类别清单。
- [x] 新物品版 LingBot 实际重跑 215 帧；全局 Sim(3) 因 22/215 inlier 被拒绝后，
  使用明确标注 pose-anchored 融合重建 169 × 152、0.05 m/cell occupancy。
- [x] SAM3 对 14 个自主标签逐类运行并做 36 帧漂移门；7 类形成 map-frame 锚点和
  7 个 region。审核批准 `table/houseplant/couch/stool/coffee table` 进入
  `objects_formal.json`，拒绝歧义、重复、错误和无三维证据标签；地点仍只批准客厅沙发。
- [x] 家庭 v2 Agent 在一个 Isaac SimulationApp 内完成正式 VLN、9 帧实时 RGB
  类别自由搜索和对象精停：导航误差 0.120 m，live 检出 `houseplant` 4 帧和
  `potted plant` 3 帧，精停误差 0.030 m、对象距离 0.749 m、朝向误差 0.049 rad；
  最后按设计阻塞于 `VLA_UNAVAILABLE`。
- [x] 新增 `g1d-home-real-nav`，把重建家庭 map/places 接到已有 ROS 2/Nav2、TF、
  轮里程计、制动和急停边界；该入口继续强制硬件输出关闭。
- [x] 接入本机 Unitree SDK2 的 G1-D `AgvClient`：新增 C++ ROS 2 驱动，把安全
  `/cmd_vel` 转为 `Move(vx,0,wz)`，制动优先转零速度，发布 RPC ready/status，并以
  SDK 连接、SDK 非零运动和上游硬件输出三重门默认禁止实体输出。
- [x] 完成新版家庭正式图：292 张 `1280x720` G1-D RGB、29 个 Florence 自主标签、
  SAM3 29/29 提示完成、399 条 map-frame 观察、18 个语义锚点/region、5 个审核对象和
  2 个审核地点；`coffee cup` 由 7 帧 reviewed mask 三角化为
  `(1.746, 2.968, 0.828) m`，基线 0.183 m、中位射线误差 0.016 m。
- [x] 家庭长任务“请带我去餐厅，拿杯子，再回到客厅沙发旁”在同一个
  `application_id=isaac-sim-122461` 成功：去程 3.581 m/904 帧，操作可见停靠和
  0.766 m 手臂停靠通过，OpenVLA 推理成功，杯子抬升 0.293 m 并稳定 30 帧，回程
  4.237 m/946 帧，携带相对距离漂移 0.00019 m。

## 当前问题

- [ ] **P0：Isaac Sim WebRTC 缺少 UDP 可达路径。** 需要由运行平台提供
  47998/UDP 映射、UDP 覆盖网络或外部 TURN relay；仅重启 Kit、设置 HTTP 代理域名或
  转发 49100/TCP 无法显示原生 Streaming 视频。三视图演示可先使用 TCP-only
  `hospital-web`；原生 WebRTC 限制详见 `docs/ISAAC_SIM_STREAMING.md`。
- [ ] **P0：实体 G1-D 尚不能启用。** 命令侧 Unitree SDK2 适配已完成，但尚未连接
  机器人网络，SDK 中也没有已确认的左右轮编码器、独立机械制动和硬急停接口；真实
  `/scan` 仍缺失。三个输出门只能保持关闭。仿真纯轮路线通过不能替代架空轮、落地
  低速、通信丢失制动和实体急停验收。
- [ ] **P0：两套 Isaac 依赖链并存。** 主 standalone 是 6.0.1/Python 3.12，
  MobileManiBench 环境仍是 Python 3.10/PyTorch 2.5.1+cu121 的早期兼容链；运行命令必须
  明确选择，后续需决定是否统一迁移。
- [ ] **P1：正式 Hospital 覆盖仅限前台和候诊区。** 主走廊及全院墙体 occupancy 尚未
  审核，不应开放为语言目标。
- [ ] **P1：候诊区仍使用单一固定停靠点。** 当前
  `waiting_area_reviewed_v1=(-5.95, 2.20, -1.571)` 已审核且 demo 稳定，但不会根据
  多把椅子、路径长度或动态占用选择不同停靠位置。下一步必须以独立制品和显式 opt-in
  实现多候选生成/排序，默认 `hospital-web` 不得改变。
- [ ] **P1：移动操作仅完成仿真原型闭环。** 当前单右臂位置 IK 不控制末端姿态，也没有
  场景/自碰撞查询；抓取依赖透明标注的 PhysX 固定约束，OpenVLA 动作只作建议。必须补
  碰撞 IK、末端姿态、真实接触抓取和放置，不能把当前结果表述为正式 VLA 或真机验收。
- [ ] **P1：正式 VLA backend 尚未交付。** 公开 OpenVLA 使用 BridgeData 统计，未针对
  G1-D 标定；需要 VLA 团队提供权重、预处理、相机协议、动作坐标系、G1-D 关节/多指手
  映射和成功判据。现有 `home-task` 可以直接替换推理 backend，但不能直接复用公开动作。
- [ ] **P1：VLA 启动门尚无实时 provider。** `interaction_profiles.json` 当前仅有
  provisional 红色方块拿取配置；仍需在同一 Isaac 会话接入头部/右腕相机、SAM3 +
  metric depth/TF、底盘速度、右臂 IK 与碰撞结果，并用实测标定距离区间。
- [ ] **P1：家庭物体精确停靠仍使用静态扫描锚点和 assisted 控制。** live RGB 会确认
  目标仍可见，但尚不能为被移动物体重新估计米制三维位姿；动态物体仍需 RGB 多视角/
  深度/TF 重定位和末端视觉伺服。当前结果不能表述为纯轮地接触或 OpenVLA 抓取验收。
- [ ] **P1：物体停靠实时控制台当前只启用 Hospital runner。** 场景 profile 和前端切换
  接口已经存在，但 SimpleRoom 或其他场景仍需各自的 USD 加载、坐标系/起点、物体生成、
  地图与 live publisher 验证；不能把仅有 profile 名称视为多场景运行成功。
- [ ] **P1：MobileManiBench 官方 G1/YCB smoke 仍缺完整资产。**
  2026-07-28 `./mobilemanibench.sh doctor` 为 12/15；项目 G1-D 和已有房间资产可用，
  但官方 G1/YCB/完整 `Assets.zip` 检查未通过，不能执行官方 reset/step 验收。
- [x] **家庭近距离巡检已发现并审核杯子。** 292 帧近距离巡检自主发现
  `coffee cup/mug/bowl`，杯子通过 mask 三角化和人工审核成为
  `manipulation_ready=true`；其他小物体仍必须逐类走相同流程，不能从资产真值补录。
- [ ] **P1：家庭场景尚未做纯轮地接触验收。** 当前导航与巡检是
  `stable_assisted`；正式地图完成后，需要在家庭门洞和家具附近做
  `--wheel-physics-only` 连续三次导航、制动和姿态门槛验证。
- [ ] **P1：Warehouse 装卸区仍未开放。** 当前 RGB 巡检和正式 occupancy 足以审核
  东/西货架通道，但 `(4,-10)` 周围 0.75 m 内没有 footprint-safe free cell；必须扩展
  RGB 巡检覆盖并重新建图、审核，不能手工把该地点改成 approved。
- [ ] 生成输出不进入 Git；若运行结果需要长期保存，应记录小型 JSON 指标或建立外部制品
  存储，而不是提交 GIF、点云、地图和模型。

## 下一步执行计划

### 0. 家庭碰撞操作和纯轮回归（唯一近期仿真下一步）

- [x] 加入真实尺度家庭物品并重新运行 `home-assets -> home-survey -> home-discover`。
- [x] 运行 `home-map`，分别人工检查自主发现标签的 SAM3 首见帧和 mask；每类必须有
  稳定 map-frame 证据，晚期跟踪漂移不得进入地点审核。
- [x] 只从匹配类别的语义锚点附近生成 footprint-safe、可达、面向物体的 docking pose；
  未检出类别继续 rejected。
- [x] 7 类锚点完成多 region 验收；同会话 live 搜索/精停通过。
- [x] 增加台面近距离 RGB 巡检段，并完成杯子三角化、审核和同会话拿取/返回原型。
- 给右臂 IK 增加末端姿态、场景/自碰撞查询，并把显式固定约束替换成接触保持验收。
- 以 `--wheel-physics-only` 连续三次验证家庭正式路线、精停、制动和姿态门槛。

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

### 2. 实体 G1-D SDK、反馈与安全验收

- [x] 用 Unitree SDK2 `g1::AgvClient` 完成 ROS 2 命令适配、零速度优先、RPC ready
  心跳和默认三重输出门。
- **唯一近期真机下一步：**在机器人旁连接实际网卡，只发零速度确认 SDK RPC；用厂商
  文档或 DDS 实测找到左右轮真实 position/velocity topic，禁止用指令积分代替反馈。
- 把真实 `/joint_states`、硬急停释放心跳和 `/scan` 接到现有 bridge/Nav2，核对单位、
  左右轮符号、时间戳和 TF。
- 先验证实体硬急停能独立切断驱动力，再做架空轮方向/编码器/制动优先级测试。
- 落地后把最大线速度限制为 `0.10 m/s`，验证直行、原地转向、松手制动、
  0.25 s cmd 超时和 0.50 s driver/feedback/急停心跳超时。
- 只有隔离场短路线与急停连续三次通过后，才显式设置
  `allow_hardware_output=True` 并逐步开放正式 Warehouse 路线。

### 3. Warehouse 覆盖扩展

- 扩展 RGB 巡检到装卸区，重新运行 LingBot、SAM3 投影、occupancy 和地点审核。
- 保留当前东/西通道基线；新地图若改变 map hash，旧地点库必须 fail-closed。
- 只有装卸区 footprint、定向 approach 和从起点可达都通过才批准。

### 4. 扩展 Hospital 与移动操作

- 审核主走廊 occupancy、连通性和 docking pose 后再增加地点。
- 底盘物理回归通过后，基于现有碰撞桌和动态方块完成 G1-D 右臂关节映射与一个不接触
  物体的预抓取位姿，使右手稳定到达方块侧面且不碰撞桌面；把该 IK/碰撞结果接入
  `ObjectObservationProvider`，据实标定并更新 provisional 距离区间。
- 再完成右手闭合、接触判定和抬升至少 5 cm，然后替换为 YCB 物体。
- 最终组合 `NAVIGATE -> ALIGN -> GRASP -> LIFT -> SUCCESS/FAIL` 单回合流程。
- 公开 OpenVLA 诊断链只作为观察/动作 schema 和同会话路由基线；正式 VLA 交付后按
  `vlaandvln.md` 替换 checkpoint 与动作统计，补齐 G1-D 右臂坐标标定、碰撞 IK 和手部
  映射，再启用真正的 `VLN -> VLA -> VERIFY` 连续执行。

## 本次维护机制验收

- [x] 根据实际项目和运行输出编写 `AGENTS.md`。
- [x] 根据当前真实进度编写本文件。
- [x] 初始化 `CHANGELOG.md`。
- [x] 创建 `.gitignore`，禁止提交大文件和凭据。
- [x] 初始化根 Git 仓库并提交当前可维护状态；基线 commit 为 `0896392`。
- [x] 基线提交后已复核暂存文件范围；最终维护记录提交后再次确认工作区状态。

## 每次工作结束前

- [x] 更新本文件的完成项、问题证据和下一步。
- [x] 更新 `CHANGELOG.md`。
- [x] 执行相关测试及 `git diff --check`。
- [x] 检查没有权重、资产、输出、日志或凭据进入暂存区。
- [x] `git commit` 保存可恢复状态。
