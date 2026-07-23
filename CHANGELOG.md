# 重要修改记录

本文件记录能够影响复现、行为、接口或任务状态的重要变更。日期使用 UTC；生成物刷新和
无行为影响的小改动不单独记录。

## 2026-07-23

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
