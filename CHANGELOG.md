# 重要修改记录

本文件记录能够影响复现、行为、接口或任务状态的重要变更。日期使用 UTC；生成物刷新和
无行为影响的小改动不单独记录。

## 2026-07-23

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
