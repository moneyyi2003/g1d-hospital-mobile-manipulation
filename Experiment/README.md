# Navigation Experiments

本目录用于 ICRA 导航实验。范围只包括自然语言语义导航；不评测 VLA 抓取。

## Benchmarks

| Benchmark | 目录 | 任务与用途 | 数据状态 |
|---|---|---|---|
| HM3D ObjectNav v2 | Habitat 官方 episodes；与 `benchmarks/hm3d-ovon` 共用 HM3D 场景 | 标准单目标语义导航；主评 HM3D val | 场景需 Matterport 学术账号授权下载；episodes 可公开下载 |
| GOAT-Bench | `benchmarks/goat-bench` | 连续多目标、多模态终身导航；主评 language-goal/val_unseen | 场景需 Matterport 授权；episodes 可公开下载 |

## Comparison methods

| Method | 目录 | 对比理由 |
|---|---|---|
| VLFM (ICRA 2024) | `baselines/vlfm` | HM3D ObjectNav；视觉语言前沿地图与零样本语义导航 |
| OpenFMNav (NAACL Findings 2024) | `baselines/openfmnav` | HM3D ObjectNav；LLM 语义解析 + VLM 语义评分地图 |
| VLMnav (2025) | `baselines/vlmnav` | 同时评测 HM3D ObjectNav 与 GOAT-Bench；端到端 VLM 对照 |
| Modular GOAT (CVPR 2024) | `baselines/modular-goat` | GOAT-Bench 官方模块化语义地图与规划基线 |

每个仓库均为浅克隆；其精确提交记录见 `sources.lock.json`。

## 统一实验协议

1. 两个 benchmark 都使用未见场景的验证集；不把测试集用于调参。
2. 所有方法使用相同场景、起点、目标、相机、动作空间、最大步数和停止距离。
3. 本方法的语言脑只能从 benchmark 目标目录/审核语义库选择合法目标；它不能生成坐标或路径。
4. 导航脑负责地图、目标可达性、footprint 膨胀、路径规划、控制与失败重规划。
5. 所有方法都以同一个 Habitat 评测器输出逐 episode JSON，再由统一脚本计算指标。

## 指标

主指标：SR（Success Rate）与 SPL（Success weighted by Path Length）。这是四个主对比方法及两个 benchmark 中最常见、可直接横向比较的指标。

补充指标：SoftSPL、DTG、Collision/m；它们用于诊断，不替代主表的 SR 与 SPL。

GOAT-Bench 另报：subtask SR、episode SR、每个子任务的 SPL、记忆收益（后续子任务相对首个子任务）以及 LLM 调用次数/端到端规划延迟。

## 数据下载前提

HM3D 的场景网格并未随本目录下载。请先申请 Matterport/HM3D 的学术研究访问权限，再按照以下两个官方仓库的说明下载：

- `benchmarks/hm3d-ovon/README.md`
- `benchmarks/goat-bench/README.md`

不要把授权令牌、下载的受限场景数据或 API 密钥提交到 Git。
