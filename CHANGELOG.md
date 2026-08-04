# 重要修改记录

本文件记录能够影响复现、行为、接口或任务状态的重要变更。日期使用 UTC；生成物刷新和
无行为影响的小改动不单独记录。

## 2026-08-04

### SimpleRoom 同会话网页命令与 Isaac 桌面

- `run_g1d_simple_room_vln.py` 新增 `--interactive-port` / `--interactive-host`。
  该模式在已加载的 SimpleRoom GUI 中启动本地控制页；HTTP 线程仅接收和排队命令，
  所有 USD、PhysX、相机与轮控操作仍由同一个 Isaac 主线程完成，避免跨线程 Kit 调用。
- 每条命令从机器人当前位姿重新规划，完成后保留场景和 GUI，可连续输入下一条；当前仅
  支持 SimpleRoom 单地点导航，拒绝 survey、Dual Brain、家庭任务和右臂 probe 的组合。
- 本机 Docker GUI 使用 Xvfb + password-protected noVNC 提供 6080/TCP 桌面；控制页默认
  绑定 127.0.0.1:6013，预期经 SSH tunnel 暴露，不依赖不可达的 WebRTC UDP 链路。

验证：

- `python3 -m py_compile run_g1d_simple_room_vln.py` 与 `git diff --check` 通过。
- 在同一 noVNC 可见 Isaac SimulationApp 中，控制页 HTTP 状态实际经历
  `idle -> queued -> running -> succeeded`；提交“请带我到沙发旁边”后到达 `sofa_side`，
  位置误差 `0.119 m`。

已知限制：

- noVNC 是临时 Docker 容器配置；重建镜像前会在下次桌面启动时重新安装其依赖。
- 当前页面是最小导航控制台，没有取消按钮、身份认证或家庭任务支持；端口保持 loopback
  绑定，不能直接暴露到公网。

## 2026-08-03

### 复核 G1-D 正向与预抓取姿态

- 结合 URDF/ROS `base_link +X`、右臂可达工作空间和实际导航位移复核 G1-D 前向；没有
  采用会把杯子置于机械臂结构背面的 180 度模型翻转。网页中双手位于身体侧后方来自全零
  收臂姿态，不能作为底盘前后方向判据。
- 仿真拿取在张手和笛卡尔 IK 前，新增 60 帧平滑右臂弯肘预备姿态；关节 seed 基于
  Unitree G1 Isaac 资产默认值，执行证据同时记录目标、实测关节角和最大误差。

验证：

- 原始指令“请带我去餐厅，拿杯子，再回到卧室”在同一 Isaac SimulationApp 完整成功；
  4 段导航均为 0 个反向运动帧，最终到达 `(-0.868, -2.047)`。
- 杯子连续抬升、稳定保持并携带返程通过；返程结束的掌心—物体距离漂移约
  `0.000002 m`。

### 修复复合指令多地点歧义

- 家庭 `--family-task` 在创建场景和 live 预览路径时，不再调用单地点
  `resolve_place(整句指令)`；现在先使用长任务编译器解析去程、对象和返回目标，再按两个
  确定的审核 `place_id` 分别规划去程/返回预览路径。
- 修复“去餐厅，拿杯子，再回到卧室”因“餐厅/卧室”同长度匹配而抛出
  `ValueError: 指令同时匹配多个地点`。普通单地点导航仍保留原歧义检查。

验证：

- Python 编译和 `git diff --check` 通过。
- 同一原始中文指令从网页启动并完整实跑成功；mission 顺序为
  `dining_area -> scan_coffee_cup_05 -> bedroom_bed`，最终位置
  `(-0.868, -2.047)`，去程和返回段均通过正向一致性门。

### 家庭网页临时开放、正向导航与连续操作画面

- 网页服务在 `outputs/family_home_web/` 生成独立 demo 地点/对象库，不修改正式
  `places_formal.json/objects_formal.json`。卧室、厨房停靠点先吸附到正式 occupancy
  最近可达自由栅格，再以 `provisional_demo` 开放；页面明确显示 DEMO 标签。
- 沙发/mug 合并为已审核对象别名；碗、椅子、长凳、桌子在存在 SAM3 map anchor 时以
  `provisional_search_only` 开放，仍保持 `manipulation_ready=false`，不会假装可抓取。
- 导航段增加位移与机器人前向轴点积检查，检测到反向运动立即使验收失败；普通 VLN 和
  Dual Brain 导航摘要均输出 `reverse_motion_frames`、`forward_only_verified`。
- SEARCH_OBJECT 原先通过直接改写底盘 yaw 获取多视角，现改为连续原地转向并逐帧发布；
  OpenVLA 操作阶段在张手、预抓取 IK、靠近、闭手、建立约束、连续抬升和稳定保持期间
  持续发布 RTX 第三人称画面，消除网页只看到抓取前后状态的问题。

验证：

- Dashboard 测试 9/9、Dual Brain 测试 20/20、Python/JavaScript 语法和
  `git diff --check` 通过。
- 临时卧室指令实跑成功：526 帧、位置误差 0.119 m、反向运动帧 0。
- “请带我去餐厅，拿杯子，再回到客厅沙发旁”在同一 Isaac SimulationApp 再次成功；
  4 个导航/对齐段反向运动帧均为 0，右手连续操作状态可在网页观察，杯子抬升
  0.304 m 并稳定保持 30 帧后携带返回。

已知限制：

- 临时开放对象仅表示可在网页发起实时搜索；除杯子外没有三维操作审核，不能用于抓取。
- 当前连续抓取仍使用位置 IK 和显式 PhysX 约束，尚未完成场景/自碰撞 IK 和纯接触抓取。

### G1-D OpenVLA-OFT 专家数采协作包

- 新建独立 `vln_vla_expert_handoff_20260803/` 交付目录，只复制而不移动/覆盖现有资产；
  包含完整 G1-D USD/URDF、SimpleRoom/SofaTablePlant、正式家庭 LingBot 点云与
  occupancy、审核地点/对象库和 VLN/Dual Brain 代码快照。
- 新增 `expert_collection_scene.usda`：操作桌上放置 3 个边长 0.03 m、密度
  2000 kg/m³、质量 0.054 kg 的红色动态方块，固定机器人、桌子、方块和放置区 Stage
  路径，供外部专家脚本参数化接入。
- 新增第三视角/右腕相机适配、专家输入 config、现有 G1-D PD/阻尼基线、action.jsonl
  schema/example 和双方 RLDS 联调检查表。腕部相机明确是项目新增传感器，外参需在正式
  批量数采前目视标定并冻结。
- 根据协作要求将网页 6012 运行时家庭组合完整物化到交付 USD：除
  SimpleRoom/SofaTablePlant 外，加入同一组 7 个家庭 fixture 和 11 个 ReplicaCAD
  家庭物品；专家桌和红色方块是在这份完整副本上的增量，不再是简化场景。

验证：

- Isaac Sim 6.0.1 成功打开组合层；14 个关键 prim、7 个家庭 fixture、11 个家庭物品、
  3 个方块尺寸/密度及全部相对资产引用检查通过。交付包约 309 MB；Python 编译、JSON 解析和
  `git diff --check` 通过。

已知限制：

- 交付包含大体积 USD/点云资产，按项目规则不进入 Git；它是文件传输制品。
- 上游 `SofaTablePlant.usd` 仍有既有纹理相对路径缺失警告；几何可加载，双方应在正式
  采集前决定修复或冻结当前渲染域。
- 本轮仅验证结构和参数合同，尚未接入同事的专家脚本，也未把首条双相机轨迹转换并回放
  为 OpenVLA-OFT RLDS。

### 家庭 Dual Brain 可视化任务控制台

- 将 6012 家庭网页从单地点 VLN 控制台扩展为自然语言任务控制台。普通指令继续调用
  正式 LingBot/SAM3 地图上的既有 VLN；含拿取的指令严格编译为审核过的
  `place_id/object_id/place_id`，然后启动同会话 Dual Brain。
- 新增 Agent 六阶段状态条、复合任务示例、任务遥测和 Isaac 第三人称实时行为；头部
  RGB 同时保留给无类别清单实时找物及 OpenVLA，网页相机不再抢占操作观测。
- Dual Brain backend 在导航、扫描、对齐、OpenVLA、验证和最终结果阶段持续发布 live
  state/MJPEG；最终失败信息保留 Executive 的结构化原因。

验证：

- 页面 HTTP 200，首字节 0.289 s；配置 API 返回 4 个正式图层，网页复合指令成功启动
  Isaac，MJPEG 单帧约 61 KB。实际状态连续经过 `NAVIGATE`、`SEARCH_OBJECT`、
  `APPROACH_AND_ALIGN` 和 `OPENVLA_PICK`。
- 该次实跑正式 VLN、实时 RGB 找杯、0.766 m 对齐和 OpenVLA 推理成功；物理抓取首次因
  约束吸附位移 0.054 m 超过 0.04 m 门槛失败，第二次因 0.038 m IK 误差超过 0.025 m
  门槛失败，Agent 正确停止且未虚构携物返回。
- Dashboard 测试 7/7、Dual Brain 测试 20/20、Python/JavaScript 语法及
  `git diff --check` 通过。完整家庭 discovery 集初次运行因缺少
  `lingbot_semantic_nav/src` 的 `PYTHONPATH` 出现 2 个导入错误，补齐环境后相关网页测试
  全部通过。

已知限制：

- 网页任务使用 `stable_assisted` 和仿真右臂；不代表纯轮地接触或实体 G1-D 验收。
- OpenVLA 仍为 BridgeData 建议，执行侧是审核锚点驱动的有界 IK/PhysX 约束；需要补充
  场景/自碰撞 IK 和更鲁棒的预抓取，不能通过放宽现有安全门来掩盖失败。

### 家庭去—拿—返回同会话闭环

- 新增可审计的家庭长任务编译器，把“去地点—拿物体—返回地点”解析为审核
  `place_id/object_id/place_id`，返回导航只有在 `VERIFY` 写入匹配
  `carried_object_id` 后才放行。
- 将家庭巡检扩展到 19 个 waypoint、292 张 `1280x720` RGB。Florence-2 在没有类别
  清单、USD 语义或物体坐标输入时接受 29 个自主标签；LingBot/SAM3 29/29 提示完成，
  生成 170 × 154 occupancy、399 条 map-frame 观察和 18 个语义锚点/region。
- 新增 reviewed SAM3 mask 多视角射线三角化。杯子使用 7 帧 RGB mask、0.183 m
  相机基线得到正式三维锚点，中位射线误差 0.016 m；对象审核将其标记为唯一
  `manipulation_ready=true` 杯子。
- `APPROACH_AND_ALIGN` 增加扫描可见方位与机械臂可达位两级停靠、3 航向 × 3 俯角
  RGB 门控，以及同会话 `SEARCH_OBJECT` 合格图像的 120 秒 freshness 备份。
- 新增仿真单右臂有界 DLS 位置 IK、G1-D 多指手目标、URDF 指尖包络门、连续坐标系
  PhysX 固定约束和实际抬升/稳定保持验证。公开 OpenVLA BridgeData 动作仍只作建议，
  不直接写入 G1-D 关节。

验证：

- 指令“请带我去餐厅，拿杯子，再回到客厅沙发旁”在同一个
  `application_id=isaac-sim-122461` 中成功。去程 3.581 m/904 帧；操作停靠
  0.766 m/0.049 rad；杯体抬升 0.293 m 并稳定 30 帧；回程 4.237 m/946 帧；
  掌心—杯体携带距离漂移 0.00019 m。摘要同时为
  `pre_vla_pipeline_succeeded=true`、`openvla_inference_succeeded=true`、
  `same_simulation_app=true`。
- 家庭测试 28/28、双脑 Agent 测试 20/20、OpenVLA 合同测试 4/4 通过；
  Python 编译、shell 语法和 `git diff --check` 通过。

已知限制：

- 本次为 `stable_assisted` Isaac 数字孪生原型，`hardware_output=false`；
  家庭纯轮地接触和实体 G1-D 均未验收。
- 右臂 IK 只控制位置，`scene_collision_query=false`；抓取依赖显式固定约束，尚未完成
  末端姿态、场景/自碰撞、真实接触抓取和放置。
- 公开 OpenVLA 未按 G1-D/家庭数据标定；不能把模型推理成功解释为模型直接控制了手臂。

## 2026-07-30

### 公开 OpenVLA-7B 单右臂诊断接入

- 新增 `g1d_openvla/` 动作合同、checkpoint 完整性检查和 G1-D 右臂交接制品。公开
  OpenVLA 的 7 维输出被明确解释为
  `[dx,dy,dz,droll,dpitch,dyaw,gripper]`，不会误作 G1-D 七个右臂关节角。
- 新增隔离的 Python 3.10 推理 sidecar 和 `openvla-infer` 入口；它只读取 RGB 与语言
  指令、保存推理结果，永远不写底盘或机械臂控制器。若未来 checkpoint 提供
  `dataset_statistics.json`，会按交付统计覆盖公开模型配置。
- 家庭 v2 Agent 增加显式 `--openvla` 路径：在同一个 Isaac SimulationApp 中完成
  正式导航、实时对象搜索和精停后，保持底盘零速度并对当时的 head RGB 运行 OpenVLA；
  推理动作经过合同校验后才进入 fail-closed 单右臂 handoff。
- 增加 checkpoint 未完成、动作维度错误、非有限值和“无关节命令”安全合同测试；更新
  双脑 Agent 与 VLN/VLA 接入文档。

验证：

- OpenVLA 合同测试 4/4、双脑 Agent 测试 14/14、旧 Agent 测试 28/28、家庭测试
  24/24 通过；Python 编译和 shell 语法检查通过。
- 三个公开权重分片均通过官方 SHA-256；已有家庭 RGB 单帧推理成功，模型加载
  6.50 s、推理 0.98 s。
- 最终同一 `application_id=isaac-sim-625387` 的实际任务完成正式导航
  （0.120 m 误差）、live RGB 搜索、0.749 m 对象距离精停（0.030 m 位置误差、
  0.049 rad 朝向误差）和实时 OpenVLA 推理。Agent 摘要为
  `pre_vla_pipeline_succeeded=true`、`openvla_inference_succeeded=true`、
  `same_simulation_app=true`；模型加载 6.79 s、推理 1.03 s，随后按设计阻塞关节执行。

已知限制：

- 公开 checkpoint 使用 BridgeData 动作统计，未针对 G1-D 单臂微调或标定；当前只验收
  模型推理和 Agent 交接，`execution_permitted=false`，不表述为伸手或抓取成功。
- 本轮最终 OpenVLA RGB 中植物只在右上角部分可见；地图几何距离/底盘朝向合格不等于
  操作视角合格。右臂 handoff 因而增加“最终帧目标可见且未贴边截断”安全门。
- 正式开放执行仍需 G1-D 动作坐标系、右臂关节限位/速度约束、场景与自碰撞 IK、
  多指手映射、独立 `VERIFY` 和可抓取小物体近距离地图证据。

### Unitree SDK2 G1-D 命令侧 ROS 2 接入

- 确认本机 `unitree_sdk2` 为官方仓库 `21d0a3b`（`2.0.2-67`），包含 G1-D
  `AgvClient::Move`、升降柱和双臂示例；三个 G1-D 官方示例在 Ubuntu 22.04/x86_64
  上实际编译链接通过。
- 新增 `unitree_g1d_driver` C++ ROS 2 包，把
  `/g1d/hardware/cmd_vel` 转为 `AgvClient::Move(vx,0,wz)`；制动、命令超时、
  非有限值和制动心跳超时均优先转为零速度。
- 增加 SDK DDS 连接、SDK 非零运动和上游 `allow_hardware_output` 三重门；
  `/g1d/hardware/driver_ready` 只有非零输出已显式允许且最近 RPC 成功时才为真。
- 家庭/Warehouse 物理 bringup 默认启动适配节点，但 DDS 不连接、非零运动禁用；
  SDK 官方角速度上限 `0.6 rad/s` 已同步到上游安全桥。

验证：

- `unitree_g1d_driver` 和 `lingbot_semantic_nav_ros` 在 ROS 2 Humble 下构建通过；
  两包合计 10 项测试全部通过，其中新增安全核心 GTest 5/5。
- 独立 dry-run 实测持续发布 `driver_ready=false`，状态明确报告
  `connect_sdk=false`、零速度制动语义、轮反馈和硬急停缺失。
- `g1d-home-real-nav` 实际拉起 Unitree 适配节点、现有安全桥和 Nav2；arm 返回
  `hardware output is disabled by configuration`，未产生 SDK DDS 或运动输出。

已知限制：

- 当前 SDK 示例没有确认左右轮编码器 topic，不能生成真实 `/joint_states` 和
  `odom -> AGV_link`；完整 bringup 因而按预期等待 odom TF。
- SDK 的 brake 只能落实为 `Move(0,0,0)`，不是独立机械制动；实体硬急停、机器人网络
  和真实 `/scan` 仍是启用前置条件。

### 家庭正式重建、实时对象搜索与同会话精停

- 用新增家庭物品后的 215 帧 G1-D RGB 重跑 LingBot；全局 Sim(3) 仅 22/215
  对应点通过 0.45 m 门，按既有真实性边界改用推理后 pose-anchored 融合，生成
  169 × 152、0.05 m/cell occupancy。
- 对 Florence-2 自主发现的 14 个标签逐类运行 SAM3 并做 36 帧漂移门，生成 133 条
  map-frame 观察、7 个语义锚点和 7 个测地 region。新增可复现人工审核策略和
  `objects_formal.json`：5 类批准搜索/停靠，9 类因歧义、重复、错误或无三维证据拒绝。
- 新增类别自由 live RGB 搜索 sidecar。目标类别不送入 Florence；模型完成推理后才与
  审核对象 ID/别名匹配。Dashboard 现在区分“自主发现、已入图、可搜索/对齐、可导航、
  可操作”状态。
- 新增 `home-dual-agent`：v2 Executive 在同一 Isaac SimulationApp 中连续执行正式
  `NAVIGATE -> SEARCH_OBJECT -> APPROACH_AND_ALIGN -> MANIPULATE`。每个对象保存独立
  停靠距离/容差，二次规划只在膨胀正式 occupancy 的当前可达空间中选择面向对象的位姿。
- 新增 `g1d-home-real-nav`，用家庭正式地图启动已有物理 G1-D ROS 2/Nav2、TF、轮里程
  计、制动和急停链；硬件输出仍强制关闭，因为本机没有可确认的厂商驱动、真机网络、硬
  急停回路和真实 `/scan`。
- 修复 ROS 入口在 `set -u` 下 source Humble 环境失败，以及家庭 docking candidate
  缺少 `clearance_m` 而无法由 `language_goal_node` 加载的问题；家庭初始位姿显式设为
  `(0,0,0)`。

验证：

- 家庭测试 24/24、双脑 Agent 测试 14/14、G1-D 安全核心测试 4/4 通过；Python 和
  shell 语法检查通过。
- 同一 Isaac `application_id` 的实际任务：正式客厅导航误差 0.120 m；9 帧 live RGB
  自主发现 12 类并确认 `houseplant` 4 帧、`potted plant` 3 帧；精停位置误差
  0.030 m、对象距离 0.749 m、朝向误差 0.049 rad。
- VLA 未交付时任务最终为 `blocked/vla_unavailable`，但
  `pre_vla_pipeline_succeeded=true`、`same_simulation_app=true`，未误报抓取成功。
- 家庭物理 bringup 实际展开并加载 169 × 152 地图；安全桥报告
  `disarmed + estop_latched + hardware_output=False`，语言目标节点成功保持运行。
  由于没有真实轮反馈和 `/scan`，`odom -> AGV_link` 不存在、Nav2 等待 TF，符合当前
  真机前置条件。

已知限制：

- live RGB 只确认静态扫描对象仍可见；物体移动后的米制三维重定位尚未实现。
- 本轮远距离巡检批准的 5 类主要是家具/地标，均为
  `manipulation_ready=false`；小型可抓取物体需增加近距离巡检。
- 家庭精停仍为 `stable_assisted`，尚未通过纯轮地接触三次回归；物理真机输出也未因
  软件接口完成而解锁。

### 并行双脑协同 Agent v2 框架

- 新增 `g1d_dual_brain_agent/`，不删除或替换原 `g1d_agent/`。新 Executive 以
  `NAVIGATE`、`SEARCH_OBJECT`、`APPROACH_AND_ALIGN`、`MANIPULATE`、`VERIFY`
  五个结构化技能管理长时程任务。
- 新增持久化对象级世界记忆和任务 blackboard，记录全局/局部位姿、房间和支撑关系、
  可见性、可达性、携带状态、任务进度及结构化失败原因；相同 mission ID 支持续跑。
- 新增底盘、右臂和右手互斥控制租约、generation 防旧租约误释放和急停锁存。VLN/VLA
  不联合训练、不同时争夺控制，但共享任务状态与执行结果。
- Executive 按失败事件有界重规划：路径阻塞回到导航，目标丢失回到搜索，不可达回到
  对齐，物体滑落回到对齐/操作；碰撞、TF、对象歧义、控制租约丢失或缺失 backend
  fail-closed。
- 新增旧 adapter 兼容桥和 `dual-agent` 入口。家庭导航仍委托
  `home-vln-formal`，继续使用 G1-D 自采 RGB、LingBot/SAM3、semantic/region、审核地点
  和正式 occupancy，不引入现成真值地图。VLA 到货后可沿用 factory 配置，并实现可选
  `search_object`、`approach_and_align`、`verify_task` 方法。
- 新增版本化 Mission 示例和中英文接入说明；更新 `vlaandvln.md`，明确 v1 为保留的
  顺序基线、v2 为推荐动态协同框架。

验证：

- v2 控制仲裁、对象记忆、动态执行和旧 VLN 桥共 14 项轻量测试通过。
- 旧 Agent 28 项测试通过；Python 编译和 shell 语法检查通过。
- `dual-agent` 家庭纯导航 plan-only 成功生成合同，未启动 Isaac。

已知限制：

- 本次未启动耗时 Isaac 仿真；只验证框架、合同和兼容映射，不新增导航物理验收证据。
- 实时对象搜索、家庭精确停靠、VLA 权重/G1-D 动作映射和独立物理验证后端尚未交付；
  交互 Mission 当前会按设计阻塞，不能表述为移动抓取闭环成功。

## 2026-07-29

### 家庭物品实体与无类别清单自主发现

- 从本机 ReplicaCAD 加入植物、台灯、显示器、篮子、杯、碗、包、书、遥控器和厨房
  小物件；`home-assets` 转换为 ignored USD。场景只暴露 `Item01...` prim 和可见/
  碰撞几何，不注册类别语义，源码真值仅供离线验收。
- 新增 `home-discover`：Florence-2 只接收 G1-D RGB 和 `<OD>` /
  `<DENSE_REGION_CAPTION>` task token，通过全图/重叠局部视图自行生成标签和框；跨帧
  一致性、结构/绿色巡检线过滤及颜色/材质描述归一化均不读取场景真值。
- `home-map` 改为先自主发现、再把模型生成的标签和首见帧交给 SAM3；人工 prompt
  必须显式启用诊断 override。semantic/region 支持动态标签，地点 ontology 只在发现
  后做别名审核，不反向提示感知。
- 巡检、LingBot、SAM3 和正式地点增加 survey/object-set 签名门；物品集或 RGB 变化后
  旧缓存和旧 occupancy 会 fail-closed。网页识别报告改为展示模型自己发现的名称、
  跨帧次数、SAM3/map 证据和导航审核状态。

验证：

- 主 Isaac Sim 6.0.1 成功转换 11 个 GLB 并完成新版家庭巡检：3750 帧、215 张
  640×360 RGB、终点误差 0.119 m/0.119 rad。
- Florence-2 base-ft 在 80 个均匀 RGB 帧及无类别局部视图上生成 621 条原始检测；
  跨帧门接受 14 个标签，包括新增实体 `houseplant`（14 帧）和 `monitor`（2 帧）。
- discovery/formal mapping/dashboard 共 15 项轻量测试通过；Python、shell 语法检查
  通过。
- `home-map --stage discover` 命中同一 survey/pipeline 签名缓存；实际网页 config
  返回 14 个自主发现类别、0 个新版 map 类别和 `stale`，旧图下发导航指令返回 HTTP
  400，未启动 Isaac。

已知限制：

- 本次只完成新版 RGB 巡检与自主发现实测；LingBot/SAM3/semantic/region/地点库仍是
  旧物品版本，必须重新运行并审核 `home-map`，因此新版正式导航当前按设计拒绝启动。
- Florence-2 仍会产生 `bathtub`、`board` 等候选；它们会保留在报告中，只有形成
  SAM3 map-frame 证据并通过地点审核后才允许影响导航。

### 家庭网页快速首屏与扫描识别报告

- 家庭网页初始化不再等待四张正式图层全部解码；配置、状态和识别结果先显示，Point
  Cloud/Semantic/Occupancy/Region 随后独立加载。
- 四层 PNG 在服务启动时读入内存，并使用带制品版本号的缓存 URL；状态轮询从约 5.6 Hz
  降到 2 Hz，减少远程浏览器和 dashboard 的无效请求。
- 页面增加“机器人扫描识别结果”：分别列出 SAM3 原始检测、通过质量门的 map-frame
  证据、语义锚点、审核和导航开放状态；场景区域明确区分“已确认”和“巡检覆盖但语义
  未确认”。
- 默认指令改成当前唯一已审核的“请带我到客厅沙发旁”。文档将一次性
  `home-survey -> home-map` 与日常 `home-web` 启动分开，避免每次打开页面重复建图。

验证：

- 家庭测试 12/12，Python、JavaScript 和 shell 语法检查通过。
- 使用现有正式制品启动服务后，HTML、config 和 state 本机首字节均约 1–2 ms；四层
  PNG 均返回 HTTP 200 和版本化缓存头。
- 识别 API 返回 215 张 640×360 RGB、4 个目标类别、1 个已识别/已审核地点和 1 个
  semantic region；沙发为 87 次原始检测/36 条 map-frame 证据，其他三类均为 0。

已知限制：

- 提交任务后的 Isaac Sim/RTX 画面仍需要独立冷启动；本次优化的是网页首屏和离线制品
  展示，不把仿真启动时间伪装成浏览器加载时间。
- 当前只有客厅由沙发语义锚点确认；卧室、餐区和厨房仍需更真实资产后重新巡检。

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
