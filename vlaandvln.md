# G1-D 的 VLN + VLA Agent 设计与接入说明

更新时间：2026-07-28（UTC）

## 1. 目标和当前状态

本项目使用一个任务级 Agent 决定自然语言任务应交给：

- VLN：把 G1-D 移动到审核地点或物体预操作停靠位；
- VLA：在物体进入机器人可操作范围后执行观察—动作闭环；
- VLN → VLA：先导航、精确停靠，再抓取、放置、开关或其他操作。

现有 Hospital VLN 是正式语义导航基线；多货架 Warehouse 复用相同的
地图/地点/规划/路径跟随组件，并通过独立场景 adapter 接入 Agent，不会让语言模型或
VLA 直接生成全局坐标。当前 VLA 尚未交付，因此代码提供了明确的 backend 插槽；执行到
VLA 步骤时会返回 `blocked`，不会把“导航到物体旁边”误报为“抓取成功”。

这里的 G1-D 是家庭轮式双臂机器人。现阶段验收目标是先在 Isaac Sim 6.0.1 中用同构
G1-D 数字孪生完成任务，再经过独立的 sim-to-real 安全验收接入物理机器人。仿真中的
机器人模型运动不等于物理真机已经执行。

## 2. 系统结构

```text
用户中文/英文指令
        |
        v
G1DTaskAgent / RuleTaskPlanner
        |
        +-- 仅换地点 ----------------------> VLN
        |
        +-- 操作已在手边 ------------------> VLA
        |
        +-- 远处物体操作 --> VLN 预抓取停靠 --> VLA

VLN:
HospitalVlnAdapter / WarehouseVlnAdapter
  -> 现有 hospital-vln / hospital-object-docking / warehouse-vln
  -> LingBot/SAM3 语义制品 + 审核数据库
  -> DeepSeek 只选择 place_id
  -> occupancy/path/Nav2 或现有 Isaac runner
  -> G1-D 底盘到达并停止

VLA:
PluginVlaAdapter
  -> 外部 VLA backend + checkpoint
  -> Isaac 相机/机器人状态观测
  -> G1-D 右臂和多指手动作
  -> 独立安全层与任务成功判据
```

代码位置：

- `g1d_agent/router.py`：判断走 VLN、VLA 或 VLN → VLA；
- `g1d_agent/agent.py`：按顺序执行、失败即停的任务状态机；
- `g1d_agent/adapters.py`：Hospital/Warehouse VLN 适配器和 VLA 插件接口；
- `g1d_agent/interaction.py`：严格加载“物体 + 技能”交互配置；
- `g1d_agent/readiness.py`：VLA 启动条件与恢复动作判定；
- `g1d_agent/supervisor.py`：现场观测、有限恢复循环和 VLA handoff；
- `g1d_agent/models.py`：任务计划、步骤和结果的 JSON 数据结构；
- `g1d_agent/interaction_profiles.json`：物体距离区间、视角、安全和成功判据；
- `g1d_agent/object_observation.example.json`：现场观测 schema 示例，不是真实感知；
- `g1d_agent/vla_backend.example.json`：等待 VLA 团队填写的配置模板；
- `scripts/run_g1d_agent.py`：命令行入口；
- `g1d_agent/tests/test_agent.py`：路由和失败传播测试。

## 3. 现有 VLN 如何被复用

Hospital VLN 的数据和执行链保持原样：

1. G1-D 戴头部相机在 Hospital 环境巡检并采集 RGB。
2. LingBot-Map 只接收 RGB，预测深度、相机运动和三维点云。
3. 推理后进行米制 Sim(3) 对齐；若使用 pose-anchored 融合，必须在制品中明确标注。
4. SAM3/语义处理识别物体或区域，把审核后的地点、物品和停靠位写入语义数据库。
5. 生成 occupancy map、region 信息和 semantic map。
6. 用户给出模糊指令时，现有 DeepSeek 解析器只能从审核数据库选择 `place_id`；它不能
   生成坐标。
7. 程序从数据库读取 docking pose，在 occupancy map 上规划并调用现有导航执行层。

Agent 只决定“这个阶段需要 VLN”，然后调用以下已有入口：

```bash
# 区域语义导航
./mobilemanibench.sh hospital-vln \
  --headless --test --no-camera \
  --command '带我去找个能坐着等医生的地方'

# 操作前的物体相对精确停靠
./mobilemanibench.sh hospital-object-docking \
  --headless --test --no-camera \
  --command '请停到红色方块前0.8米'

# 多货架 Warehouse 正式 RGB-only 地图导航
./mobilemanibench.sh warehouse-vln-formal \
  --headless --no-camera --wheel-physics-only \
  --steps 12000 --position-tolerance 0.20 --yaw-tolerance 0.20 \
  --command '请带我到东侧货架通道'
```

区域导航仍使用审核地点库。物体操作前必须把 SAM3/检测结果转换成带交互面朝向和安全
距离的 SE(2) 预抓取停靠位，再检查 footprint、occupancy 和路径可达性。不能把任意检测
中心点直接交给底盘。

Warehouse 的 bootstrap 和正式 occupancy 路线都已在 `--wheel-physics-only` 下分别
连续三次通过，同时满足目标误差、倾斜、制动漂移和停止速度判据。正式路线使用审核的
预停靠点沿 docking yaw 进入目标，三次均为 10,306 帧、32.516 m 物理路程、
0.190 m/0.051 rad 位置/朝向误差。188 帧 G1-D RGB 巡检、LingBot RGB-only 推理、
SAM3.1 货架语义投影和正式地点审核也已完成。证据边界见
`docs/WAREHOUSE_G1D_NAV.md`。

## 4. Agent 如何做决定

当前 `RuleTaskPlanner` 使用保守、可审计的任务语义规则，不调用新的导航模型：

| 指令类型 | 示例 | 路由 |
|---|---|---|
| 只改变地点 | “带我去候诊区” | `VLN` |
| 当前可达物体操作 | “抓起眼前的杯子” | `VLA` |
| 移动操作 | “去桌边拿起红色方块” | `VLN → VLA` |
| 含义不清 | “帮我一下” | 拒绝并要求明确任务 |

DeepSeek 仍位于现有 VLN 内部，负责把模糊地点描述约束到审核 `place_id`。Agent 的路由层
不重复调用 DeepSeek，避免同一句指令被两个语言模型分别生成互相冲突的地点。后续如果
任务种类变多，可以把 `RuleTaskPlanner` 替换为结构化 LLM planner，但输出仍必须通过
`Capability`、`StepKind` 和审核 catalog 校验。

执行状态机是顺序且 fail-closed 的：

1. 生成不可变 `MissionPlan`。
2. 根据每步的 `capability` 选择 VLN 或 VLA adapter。
3. 只有前一步 `succeeded` 才能执行下一步。
4. 导航失败时跳过 VLA，禁止机器人在未知底盘位姿上伸手。
5. VLA 前先解析唯一的“物体 + 技能”交互配置，再读取现场观测。
6. `VlaReadinessGate` 检查物体、相机、距离、朝向、IK、碰撞和底盘静止。
7. 条件不满足时给出有限恢复动作；没有恢复控制器或超过三次仍不满足则 `blocked`。
8. VLA 未就绪时任务为 `blocked`，不把预抓取停靠当成操作完成。
9. 只有所有步骤都通过各自成功判据，任务才是 `succeeded`。

### 4.1 物体距离不是单点

`interaction_profiles.json` 以 `(object_id, skill)` 为唯一键。同一个物体执行拿取、放置、
推拉或开关时可以使用不同站位。每项至少定义：

- `preferred_distance_m`：交给现有物体停靠 runner 的推荐距离；
- `minimum_distance_m` / `maximum_distance_m`：允许启动 VLA 的距离区间；
- 横向和朝向误差上限；
- 检测置信度、稳定帧、位姿不确定度、观测时效和必需相机；
- 底盘停止阈值、惯用手、运行环境和独立成功判据。

当前只有 `red_cube_demo + pick` 的 provisional profile：推荐 0.80 m、允许
0.65–0.90 m。它会覆盖用户自由文本中的距离，防止“拿起方块但停在 1.5 m”直接进入操作。
这些数值是接口联调初值，尚未经过右臂 IK 或 VLA 实测，不能标记为
`sim_validated`/`real_validated`。

### 4.2 VLA 启动门

现场观测统一在导航底盘坐标下表达：Isaac 集成可使用 `base_link` 适配层，物理 G1-D
使用 URDF 真实根 frame `AGV_link`。距离定义为底盘中心到物体中心。只有以下检查全部
通过才生成 `start_vla`：

```text
环境和 object_id 与 profile 一致
物体可见，检测置信度和连续稳定帧达标
头部/腕部必需相机存在，观测未过期
位姿不确定度在阈值内
碰撞检查通过
底盘线速度和角速度低于停止阈值
距离位于物体—技能允许区间
横向误差和朝向误差达标
右臂 IK 可达
```

失败会分别产生 `reacquire_object`、`wait_for_stable_pose`、`stop_base`、
`move_closer`、`move_away`、`realign_base`、`reposition_for_reachability`、
`block_collision` 或 `block_configuration`。`ReadinessRecoveryController` 最多做三次
有界恢复，每次都必须重新获取观测和重新门控；碰撞或配置错误不会自动重试。

## 5. 当前使用方法

默认只规划，不启动 Isaac：

```bash
./mobilemanibench.sh agent \
  --command '带我去找个能坐着等医生的地方'

./mobilemanibench.sh agent \
  --command '去桌边拿起红色方块'

./mobilemanibench.sh agent \
  --navigation-scene warehouse \
  --command '请带我到东侧货架通道'
```

计划和结果写入 `outputs/g1d_agent/mission.json`。该输出不进入 Git。

显式执行纯 VLN 任务：

```bash
./mobilemanibench.sh agent --execute \
  --command '带我去找个能坐着等医生的地方'
```

显式执行移动操作时，当前会先运行预抓取停靠；到 VLA 阶段后，因为默认 backend 尚未
交付、实时观测 provider 尚未接入，会返回 `blocked` 和退出码 3：

```bash
./mobilemanibench.sh agent --execute \
  --command '去桌边拿起红色方块'
```

可以用只用于接口测试的静态观测验证“门控通过后仍因 VLA 缺失而阻塞”：

```bash
./mobilemanibench.sh agent --execute \
  --command '抓起眼前的红色方块' \
  --readiness-observation g1d_agent/object_observation.example.json \
  --vla-config g1d_agent/vla_backend.example.json
```

这个示例 JSON 不能用于仿真或真机验收；CLI 会拒绝把静态 observation 与已启用的 VLA
backend 同时使用。真实任务必须由同一 Isaac 会话或 ROS 2 机器人桥持续实现
`ObjectObservationProvider`。

大型仿真前仍须检查是否已有 Isaac Kit 进程，避免多个实例争用 GPU。

## 6. VLA 团队需要交付什么

至少需要以下内容，不能只给一个权重文件：

1. checkpoint、模型代码、精确依赖版本和推理启动方式；
2. 训练时的图像尺寸、归一化、相机名称和相机外参约定；
3. 文本 tokenizer/prompt 模板；
4. 动作类型、数值范围、单位、控制频率和 action chunk 长度；
5. G1-D 关节顺序以及训练机器人到本项目 G1-D 的动作映射；
6. 模型 reset、推理、终止和异常处理接口；
7. 训练任务对应的成功判据和已知失败模式；
8. 模型许可证和权重校验和。
9. 如果 VLA 集成包同时负责现场接管，实现可选的 `observe_readiness(request)`；
10. 如果它同时负责局部视觉停靠，实现可选的 `recover_readiness(request)`。

本项目的第一版 VLA 动作范围建议只包含右臂和右手，不让 VLA 同时控制底盘：

```text
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
right_hand_thumb_0_joint
right_hand_thumb_1_joint
right_hand_thumb_2_joint
right_hand_middle_0_joint
right_hand_middle_1_joint
right_hand_index_0_joint
right_hand_index_1_joint
```

VLN 到达后锁定/保持底盘，由 VLA 操作右臂和多指手。等这一边界稳定后，才考虑让 VLA
输出短距离视觉伺服底盘动作。

## 7. VLA backend 接口

复制并修改模板：

```bash
cp g1d_agent/vla_backend.example.json /path/outside/git/g1d_vla.json
```

配置中的 factory 格式为：

```json
{
  "enabled": true,
  "backend": {
    "factory": "your_vla_package.g1d_backend:create_backend",
    "checkpoint": "/absolute/path/to/checkpoint",
    "device": "cuda:0"
  }
}
```

factory 接收完整配置并返回 backend。backend 最小接口：

```python
class G1DVlaBackend:
    def ready(self) -> bool:
        # 权重、相机、关节映射、安全层全部通过后才返回 True
        ...

    def execute(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        # request["mission_context"] 包含 VLN 结果、handoff_artifacts、
        # interaction_profile、现场 observation 和完整 readiness 检查。
        # 内部完成 observation -> VLA -> safe action -> Isaac step 的闭环。
        ...
        return {
            "status": "succeeded",  # succeeded / blocked / failed
            "success": True,
            "message": "cube lifted 0.071 m for 30 frames",
            "metrics": {
                "lift_height_m": 0.071,
                "hold_frames": 30
            }
        }


def create_backend(config):
    return G1DVlaBackend(config)
```

为了直接接入同一 Isaac 会话，backend 还可以实现：

```python
def observe_readiness(self, request):
    # 从当前相机、TF、底盘速度、IK 和碰撞场景返回 object_observation_v1 字段。
    return observation

def recover_readiness(self, request):
    # 根据 move_closer/move_away/realign/... 做一次有界动作，不能内部无限循环。
    return {"status": "succeeded"}
```

CLI 会自动发现这两个方法并接到 `ReadinessVlaAdapter`；未实现时不会绕过门控，而是明确
阻塞。建议把 SAM3/RGB-D/TF、Nav2 局部调整和 VLA 放在 integration backend 中，模型
权重代码本身仍只负责推理。

接入后：

```bash
./mobilemanibench.sh agent --execute \
  --vla-config /path/outside/git/g1d_vla.json \
  --command '去桌边拿起红色方块'
```

VLA 返回 `succeeded` 还不够；`success` 必须为 `true`，并由仿真环境的独立判据验证。
例如抓取任务应检查物体相对桌面抬升至少 5 cm、连续保持若干帧且没有穿模或 NaN。

当前 VLN adapter 会把 `docking_plan.json` 和 `run_summary.json` 路径放入下一步的
`mission_context.previous_steps[].details.handoff_artifacts`。这能验证阶段边界和支持从
停靠位恢复，但动态物体任务最终应让导航与 VLA 运行在同一个 Isaac 会话中：在现有
Hospital runner 报告到达后保持 SimulationApp、相机、机器人和物体状态不变，直接调用
backend 的操作循环。不能依赖关闭并重开场景来声称连续移动操作已经通过。

## 8. 在 Isaac Sim 中接入 VLA 的步骤

1. 确认使用主 Isaac Sim 6.0.1/Python 3.12，或把 VLA 部署成独立进程；不要把
   MobileManiBench Python 3.10 依赖直接混入主 Isaac Python。
2. 在现有 Hospital 操作 demo 中保留碰撞桌和动态物体，先完成无接触预抓取。
3. 为 G1-D 建立稳定的右臂/右手关节控制映射、默认姿态、限位和控制增益。
4. 建立头部 RGB 和右腕 RGB 观测；若 VLA 训练使用深度或多帧历史，也必须按训练协议
   提供，不能临时猜测。
5. 实现 `ObjectObservationProvider`：输出 `base_link` 下物体距离/横向/朝向误差、
   检测稳定性、底盘速度、IK 和碰撞结果。
6. 实现 `ReadinessRecoveryController`：只执行一次有界扫描、等待、停止、靠近、后退、
   对齐或换站位，随后把控制权交回 Agent 重新检查。
7. 把 Isaac observation 转成 VLA 训练时完全一致的 tensor/prompt。
8. 把 VLA action 解码成 G1-D 关节目标，再经过限位、速度限制、碰撞停止和时间戳检查。
9. 每个控制周期执行 `observe -> infer -> validate -> apply -> step -> verify`。
10. 在现有 Hospital runner 的 arrival 分支增加 VLA hook，使导航和操作共享同一
   SimulationApp；该 hook 使用 Agent 生成的 VLA step 和同一份 backend 配置。
11. 先验收 `PREGRASP -> GRASP`，再验收 `GRASP -> LIFT`，最后由 Agent 串成
   `NAVIGATE -> PREGRASP_DOCK -> GRASP -> LIFT -> SUCCESS/FAIL`。
12. 固定随机种子至少重复三次，并记录位置误差、碰撞、抓取保持帧数和物体抬升高度。

如果 VLA 运行环境与 Isaac Python 不兼容，推荐让 VLA 独立运行，通过本机 ROS 2、ZeroMQ
或 gRPC 传输版本化 observation/action 消息；Agent 的 `PluginVlaAdapter` 可以再封装这个
客户端。跨进程消息必须带 schema 版本、frame ID、时间戳和 action sequence ID。

## 9. 从仿真迁移到物理 G1-D

仿真和真机应复用任务计划与 VLA observation/action schema，但执行桥必须分开：

```text
同一个 Agent / MissionPlan
       +-- IsaacRobotBridge -> Isaac G1-D articulation
       +-- RealRobotBridge  -> ROS 2 / 真机控制器
```

迁移顺序：

1. 校准物理链 `map -> odom -> AGV_link -> right_wrist -> camera`；若 VLA 内部使用
   `base_link`，必须提供显式静态变换并记录版本。
2. 核对物理机器人关节名称、方向、零点、限位和控制周期。
3. 用 recorded observation 做离线推理，不下发动作。
4. 在仿真回放相同动作，检查动作缩放和关节映射。
5. 真机先断电/悬空或低力矩单关节测试，再做空载手臂动作。
6. 设置硬件急停、软件 watchdog、关节/速度/力矩限制、碰撞停止和工作空间边界。
7. 真机控制必须显式人工 enable；VLA 超时、图像过期、TF 缺失或网络断开立即停止。
8. 先使用软物体、低速和隔离工作区，再逐步验证导航—停靠—抓取闭环。

物理机器人验收必须单独记录，不能沿用 Isaac 的 `stable_assisted` 指标。尤其是当前
纯轮地接触 bootstrap 与正式 occupancy 路线均已连续三次通过，但实体 G1-D 尚无厂商
驱动、硬急停和现场低速验收。ROS 2/Nav2、TF、轮里程计、制动和急停接口现已默认断能
接入，详见
`docs/G1D_REAL_ROS2_NAV.md`。

## 10. 近期建议验收顺序

1. 保持现有 Hospital VLN 回归不变。
2. 完成 G1-D 右手到红色方块侧面的无接触预抓取位姿。
3. VLA 到货后先验证 `ready()`、观测形状和动作映射，不启动完整任务。
4. 单独验收 VLA 抓取与抬升。
5. 用本 Agent 验收 `VLN → VLA`，并确认导航失败会阻止操作。
6. 在现有 fail-closed 真机桥上接入厂商驱动并完成硬急停、架空轮和隔离场验收后，再开始
   物理 G1-D 运动联调。
