# 当前任务与交接状态

更新时间：2026-07-23（UTC）

## 当前结论

Hospital 语义导航 MVP 已在确定性 `stable_assisted` 模式通过，当前主线应转向
G1-D 纯轮地接触物理控制的稳定性与回归；完成后再推进移动操作任务。根目录长期维护
机制已经建立。Hospital TCP dashboard 已能在浏览器同步显示 Isaac chase camera、
LingBot RGB 点云和 occupancy map 上的实时机器人轨迹，并已接入 DeepSeek 模糊地点
理解，不依赖 WebRTC UDP。2026-07-23 的 WebRTC 排查仍确认：当前 AutoDL 公有云实例
没有浏览器可达的 47998/UDP 媒体路径；该外部网络条件解决前，原生 WebRTC 页面仍会
停在 `WAITING FOR STREAM`。

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
- [ ] **P1：移动操作尚未闭环。** G1-D 右臂、多指手、IK、接触参数和抓取成功判据仍需
  单独配置，不能复用官方 G1 平行夹爪动作空间。
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
- 先完成静态立方体右手靠近、闭合、抬升至少 5 cm，再替换为 YCB 物体。
- 最终组合 `NAVIGATE -> ALIGN -> GRASP -> LIFT -> SUCCESS/FAIL` 单回合流程。

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
