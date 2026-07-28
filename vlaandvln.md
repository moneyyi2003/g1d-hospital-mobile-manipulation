# G1-D 的 VLN + VLA Agent 设计与接入说明

更新时间：2026-07-28（UTC）

## 1. 目标和当前状态

本项目使用一个任务级 Agent 决定自然语言任务应交给：

- VLN：把 G1-D 移动到审核地点或物体预操作停靠位；
- VLA：在物体进入机器人可操作范围后执行观察—动作闭环；
- VLN → VLA：先导航、精确停靠，再抓取、放置、开关或其他操作。

现有 Hospital VLN 是唯一导航实现，Agent 不会另建导航系统，也不会让语言模型或 VLA
直接生成全局坐标。当前 VLA 尚未交付，因此代码提供了明确的 backend 插槽；执行到 VLA
步骤时会返回 `blocked`，不会把“导航到物体旁边”误报为“抓取成功”。

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
HospitalVlnAdapter
  -> 现有 hospital-vln / hospital-object-docking
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
- `g1d_agent/adapters.py`：现有 Hospital VLN 适配器和 VLA 插件接口；
- `g1d_agent/models.py`：任务计划、步骤和结果的 JSON 数据结构；
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
```

区域导航仍使用审核地点库。物体操作前必须把 SAM3/检测结果转换成带交互面朝向和安全
距离的 SE(2) 预抓取停靠位，再检查 footprint、occupancy 和路径可达性。不能把任意检测
中心点直接交给底盘。

当前成功导航使用 `stable_assisted`。它证明语言—地图—规划—高层控制链路，但不等于
纯轮地接触控制或真实机器人底盘已经验收；`--wheel-physics-only` 仍是独立 P0 工作。

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
5. VLA 未就绪时任务为 `blocked`，不把预抓取停靠当成操作完成。
6. 只有所有步骤都通过各自成功判据，任务才是 `succeeded`。

## 5. 当前使用方法

默认只规划，不启动 Isaac：

```bash
./mobilemanibench.sh agent \
  --command '带我去找个能坐着等医生的地方'

./mobilemanibench.sh agent \
  --command '去桌边拿起红色方块'
```

计划和结果写入 `outputs/g1d_agent/mission.json`。该输出不进入 Git。

显式执行纯 VLN 任务：

```bash
./mobilemanibench.sh agent --execute \
  --command '带我去找个能坐着等医生的地方'
```

显式执行移动操作时，当前会先运行预抓取停靠；到 VLA 阶段后，因为默认 backend 尚未
交付，会返回 `blocked` 和退出码 3：

```bash
./mobilemanibench.sh agent --execute \
  --command '去桌边拿起红色方块'
```

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
        # request["mission_context"] 包含 VLN 结果与 handoff_artifacts。
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
5. 把 Isaac observation 转成 VLA 训练时完全一致的 tensor/prompt。
6. 把 VLA action 解码成 G1-D 关节目标，再经过限位、速度限制、碰撞停止和时间戳检查。
7. 每个控制周期执行 `observe -> infer -> validate -> apply -> step -> verify`。
8. 在现有 Hospital runner 的 arrival 分支增加 VLA hook，使导航和操作共享同一
   SimulationApp；该 hook 使用 Agent 生成的 VLA step 和同一份 backend 配置。
9. 先验收 `PREGRASP -> GRASP`，再验收 `GRASP -> LIFT`，最后由 Agent 串成
   `NAVIGATE -> PREGRASP_DOCK -> GRASP -> LIFT -> SUCCESS/FAIL`。
10. 固定随机种子至少重复三次，并记录位置误差、碰撞、抓取保持帧数和物体抬升高度。

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

1. 校准 `map -> odom -> base_link -> right_wrist -> camera` 坐标链。
2. 核对物理机器人关节名称、方向、零点、限位和控制周期。
3. 用 recorded observation 做离线推理，不下发动作。
4. 在仿真回放相同动作，检查动作缩放和关节映射。
5. 真机先断电/悬空或低力矩单关节测试，再做空载手臂动作。
6. 设置硬件急停、软件 watchdog、关节/速度/力矩限制、碰撞停止和工作空间边界。
7. 真机控制必须显式人工 enable；VLA 超时、图像过期、TF 缺失或网络断开立即停止。
8. 先使用软物体、低速和隔离工作区，再逐步验证导航—停靠—抓取闭环。

物理机器人验收必须单独记录，不能沿用 Isaac 的 `stable_assisted` 指标。尤其是当前
纯轮地接触导航尚未通过，必须在接真机底盘前完成轮轴方向、摩擦、速度闭环和制动验证。

## 10. 近期建议验收顺序

1. 保持现有 Hospital VLN 回归不变。
2. 完成 G1-D 右手到红色方块侧面的无接触预抓取位姿。
3. VLA 到货后先验证 `ready()`、观测形状和动作映射，不启动完整任务。
4. 单独验收 VLA 抓取与抬升。
5. 用本 Agent 验收 `VLN → VLA`，并确认导航失败会阻止操作。
6. 完成纯轮地物理底盘和真机安全桥后，再开始物理 G1-D 联调。
