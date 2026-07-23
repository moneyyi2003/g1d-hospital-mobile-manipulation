# LingBot RGB-only Semantic Navigation

这是一个按“官方开源组件 + 最薄适配层”重建的语义导航框架：

```text
RGB 图像/视频
  → 官方 LingBot-Map：预测深度、相机位姿、点云
  → ROS occupancy map
  → 官方 SAM3：文本检测、分割、跨帧跟踪
  → mask 与 LingBot 几何投影到 map 三维坐标
  → footprint 安全停靠候选与人工审核
  → 地点坐标库 v2
  → DeepSeek 只选择受约束的地点 ID
  → Nav2 逐段规划、避障和控制真实差速轮仿真机器人
```

## 不可破坏的边界

- LingBot-Map 的正式输入只有 RGB；Habitat depth、pose、semantic truth、navmesh 不进入建图链。
- SAM3 完整替换旧的 OWLv2 + SAM2 串联；正式主链不再安装或调用这两个模型。
- SAM3 和 DeepSeek 都不能输出导航坐标。只有已审核的地点库能产生 `PoseStamped`。
- `candidate/rejected/stale` 地点不会进入在线导航目录；正式目录必须绑定 `map_id + map_sha256`。
- “去 B”只向 Nav2 提交 B；“经过 A 到 B”提交 `pass(A), arrive(B)`；“先到 A 再到 B”提交 `arrive(A), arrive(B)`。
- 配置拓扑图本身不会插入途经点。只有用户明确给出方向约束时才允许拓扑展开。
- 正式仿真必须是 `cmd_vel → 差速轮物理模型 → scan/imu/odom/TF → Nav2` 闭环，不能瞬移或播放规划轨迹。

## 官方上游

精确版本在 [`config/upstreams.lock.json`](config/upstreams.lock.json)：

| 组件 | 固定版本 | 集成方式 |
|---|---|---|
| LingBot-Map | `7ff6f3ed0913…` | 直接导入官方模型和 demo helpers |
| SAM3 / SAM3.1 | `46957e47805e…` | 直接调用官方 video predictor session API |
| Nav2 Humble | `3c3db59d696…` | ROS 2 Humble 官方包；提交 `NavigateToPose` |
| Habitat-Sim | `v0.3.3` / `acbe6f4922e…` | 采集、仿真和展示，真值只可用于评测 |
| TurtleBot3 Simulations | `a35a56c8b048…` | Gazebo Classic 差速轮物理仿真 |

模型源码已按锁文件做稀疏 checkout，且被 `.gitignore` 排除。重新获取：

```bash
python3 scripts/fetch_upstreams.py --group models
```

SAM3 使用自定义 SAM License，权重还需要按官方说明申请访问；不能把源码许可和权重授权混为一谈。详细声明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 环境隔离

三个运行环境必须分开：

1. 核心环境：Python 3.10+，运行地点库、DeepSeek、工件融合和测试。
2. LingBot-Map GPU 环境：按 `third_party/lingbot-map/pyproject.toml` 安装官方包和对应 PyTorch/CUDA。
3. SAM3 GPU 环境：按官方仓库安装 Python 3.12、PyTorch 2.7+、CUDA 12.6+ 和 SAM3。
4. ROS 环境：Ubuntu 22.04、ROS 2 Humble、Nav2、Gazebo Classic、TurtleBot3。

模型环境通过文件工件衔接，不在 ROS Python 进程里同时加载两个大模型。

## 当前框架结构

```text
config/upstreams.lock.json               官方仓库、commit 和许可证锁
third_party/                              可复现的官方稀疏 checkout，不提交
schemas/places.v2.schema.json             正式地点坐标库契约
src/lingbot_nav/
  mapping/lingbot_backend.py              官方 LingBot-Map I/O 适配
  perception/sam3_backend.py              官方 SAM3 视频跟踪 I/O 适配
  mapping/mask_projection.py              mask + LingBot 几何 → map 三维观测
  mapping/docking.py                      occupancy/footprint 停靠候选
  place_catalog_builder.py                候选库生成与显式人工晋升
  place_db.py                             在线只读、只加载 approved 地点
  intent.py                               DeepSeek/规则结构化 ID 解析
  mission.py                              禁止隐式途经点的有序任务解析
ros2_ws/src/lingbot_semantic_nav_ros/
  lingbot_semantic_nav_ros/language_goal_node.py
                                           有序 NavigateToPose 执行和审计
  launch/gazebo_wheel_nav2.launch.py       正式轮式物理仿真入口
tests/                                     安全边界和数据契约测试
```

旧的 `habitat_nav2.launch.py`、`lingbot_map_wheel_nav2.launch.py` 和 dashboard wheel bridge 仅用于历史结果复现。它们使用简化 odom/静态局部地图，不属于正式轮式物理验收入口。

## 安装核心工具

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mapping]"
lingbot-nav verify-upstreams
python -m unittest discover -s tests -v
```

DeepSeek 配置写入本机 `.env`：

```bash
cp .env.example .env
# 填写 DEEPSEEK_API_KEY，不要提交
```

正式运行建议关闭规则回退：

```dotenv
LLM_PROVIDER=deepseek
LLM_ALLOW_RULE_FALLBACK=false
```

DeepSeek 请求只包含 `{id, name, aliases}`；地点坐标永远不放进提示词。

## 离线工件流水线

### 1. LingBot-Map RGB-only 推理

```bash
lingbot-nav lingbot-infer \
  --rgb data/scene/rgb \
  --checkpoint checkpoints/lingbot-map.pt \
  --output outputs/scene/lingbot
```

适配器直接运行官方模型，输出：

- `preprocessed_rgb/`：与 LingBot 几何逐像素对齐的 RGB；
- `predictions/frame_*.npz`：depth、camera-to-world、world points、confidence；
- `lingbot_manifest.json`：官方 commit、checkpoint hash 和 RGB-only 真值声明。

### 2. 构建米制 ROS occupancy map

```bash
lingbot-nav build-map \
  --predictions outputs/scene/lingbot/predictions \
  --alignment config/scene_lingbot_to_map.json \
  --scale 1.234 \
  --resolution 0.05 \
  --output outputs/scene/map
```

这一步只消费 LingBot 预测工件。对齐矩阵和 `m/unit` 尺度必须显式给出；free 只来自观察到的地面，未观察区域保持 unknown。

### 3. SAM3 文本跟踪

让 SAM3 读取 LingBot 导出的 `preprocessed_rgb/`，避免 crop/resize 后 mask 与几何错位：

```bash
lingbot-nav sam3-track \
  --video outputs/scene/lingbot/preprocessed_rgb \
  --prompts config/scene_prompts.txt \
  --output outputs/scene/sam3
```

每个文本概念使用一个官方 SAM3 session，原始 object ID 加 prompt namespace 后保存。这里没有 OWLv2 或 SAM2。

### 4. mask 投影到三维地图

```bash
lingbot-nav project-tracks \
  --predictions outputs/scene/lingbot/predictions \
  --sam3 outputs/scene/sam3/000_sofa \
  --alignment config/scene_lingbot_to_map.json \
  --scale 1.234 \
  --prompt sofa \
  --output outputs/scene/sofa_observations.json
```

`--scale` 和 4×4 对齐矩阵必须显式提供；LingBot 本地单位不会被假装成米。

### 5. 生成候选地点库

```bash
lingbot-nav build-place-candidates \
  --observations outputs/scene/sofa_observations.json \
  --map outputs/scene/map/map.yaml \
  --map-id scene-001 \
  --start-x 0.0 --start-y 0.0 \
  --robot-radius 0.22 \
  --output outputs/scene/places.json
```

候选点只会出现在完整 footprint 为 free 且从起点可达的连通区域。生成结果仍是 `candidate`，不能在线导航。

### 6. 显式审核并晋升

```bash
lingbot-nav approve-place \
  --places outputs/scene/places.json \
  --map outputs/scene/map/map.yaml \
  --place-id sofa_p0_o1 \
  --candidate-id dock_r100_c80 \
  --reviewer operator-name \
  --evidence outputs/scene/reviews/sofa_p0_o1.png
```

晋升时会重新核对地图 hash 和 footprint。只有成功晋升的 `approved` 地点会进入 `PlaceDatabase`。

## 有序语言任务

离线检查：

```bash
lingbot-nav parse "先到A，再到B" \
  --places outputs/scene/places.json \
  --provider deepseek
```

在线 ROS 节点逐个发送 Nav2 goal，并等待前一段 Nav2 成功和 TF 到达复核后才发送下一段。A 失败时默认终止任务，不跳过 A 去 B。审计日志记录用户请求的 ID 序列、实际提交的 ID 序列、每段结果和到达误差。

## 正式轮式仿真

安装 ROS 2 Humble、Nav2 与官方 TurtleBot3 Gazebo 包并构建工作区后：

```bash
ros2 launch lingbot_semantic_nav_ros gazebo_wheel_nav2.launch.py \
  map:=/absolute/path/to/map.yaml \
  places:=/absolute/path/to/places.json \
  model:=waffle_pi \
  provider:=deepseek
```

该入口使用 Gazebo 物理、差速轮、LiDAR、IMU、轮式 odom、AMCL、TF 和 Nav2。地图必须来自同一个仿真世界的 RGB/LingBot 建图结果。导航执行只通过 `cmd_vel`，不会读取 Gazebo 真值 pose 来控制机器人。

发送命令：

```bash
ros2 topic pub --once /semantic_nav/command std_msgs/msg/String \
  "{data: '先到A，再到B'}"
```

状态：

```bash
ros2 topic echo /semantic_nav/status
```

正式验收还需要在目标场景完成 LingBot/SAM3 权重实跑、生成与 Gazebo 世界一致的地图、审核多个地点，并统计真实轮式闭环的 Success、SPL、碰撞、超时和目标到达误差。
