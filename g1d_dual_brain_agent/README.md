# G1-D 双脑协同移动操作 Agent（v2）

这个目录是与旧 `g1d_agent/` 并行的新框架。旧目录、旧
`./mobilemanibench.sh agent` 入口和旧测试均保留；新入口为
`./mobilemanibench.sh dual-agent`。

## 边界

- Executive Agent 只拆解任务、选择技能、维护进度和恢复路线，不直接输出轮速或关节角。
- VLN、VLA 不联合训练，也不会同时取得同一执行器的控制权。
- 两者共享对象级世界记忆、当前任务阶段、携带物、失败原因和执行结果。
- VLN 继续调用现有场景 adapter。家庭场景仍是
  `home-vln-formal`，使用 G1-D 自采 RGB、LingBot RGB-only、SAM3、semantic、
  region、审核地点和正式 occupancy；Agent 不换用现成真值地图，也不生成任意坐标。
- 家庭 `SEARCH_OBJECT`、两级 `APPROACH_AND_ALIGN`、单右臂仿真拿取和独立
  `VERIFY` 已有同会话实现。公开 OpenVLA 7B 可读取通过门控的 head RGB 并生成动作，
  但未标定 BridgeData 动作不直接写入 G1-D 关节；当前执行是审核锚点驱动的有界位置
  IK 和透明标注的 PhysX 固定约束。

## 运行结构

```text
Mission
  -> Executive
       -> SharedWorldMemory
       -> NAVIGATE            (现有 VLN；独占 base)
       -> SEARCH_OBJECT       (live RGB/语义跟踪；独占 base)
       -> APPROACH_AND_ALIGN  (可达位姿/视角/碰撞；独占 base)
       -> MANIPULATE          (VLA；独占 base + right_arm + right_hand)
       -> VERIFY              (独立成功判据)
       -> 按结构化失败原因动态重规划
```

主要文件：

- `models.py`：版本化 Mission、Skill、结果和失败码；
- `memory.py`：可持久化的对象记忆与任务 blackboard；
- `control.py`：带 generation 的执行器互斥租约和急停锁存；
- `executive.py`：事件驱动路由、进度管理和失败恢复；
- `skills.py`：技能接口、可选集成后端和 fail-closed 占位；
- `legacy.py`：把新 `NAVIGATE`/`APPROACH_AND_ALIGN`/`MANIPULATE`
  转换为已有 `g1d_agent` adapter；
- `planner.py`：加载 Mission，或复用旧规则路由器编译单句命令；
- `mission.example.json`：长时程“找—拿—送—放”合同示例。

## Executive 的决策规则

交互目标不是固定的 `VLN -> VLA` 两步，而是根据事件选择下一项：

```text
未到目标区域                         -> NAVIGATE
对象缺失、不可见或观测过期           -> SEARCH_OBJECT
对象新鲜可见但本任务尚未完成可达对齐 -> APPROACH_AND_ALIGN
已对齐且未执行操作                   -> MANIPULATE
操作已返回成功                       -> VERIFY
验证成功                             -> 完成目标
```

恢复也是结构化的。例如 `PATH_BLOCKED` 回到导航，
`TARGET_NOT_FOUND` 回到搜索，`OUT_OF_REACH` 回到对齐，
`OBJECT_SLIPPED` 回到对齐/操作，`VERIFY_FAILED` 重新验证。
碰撞风险、TF 缺失、对象身份歧义、控制租约丢失和未实现技能直接阻塞。
每种技能和整项任务都有有界次数，避免无限循环。

对象记忆包含对象 ID/名称、全局和局部位姿、房间、支撑面、可见性、置信度、
不确定度、观测来源、地图版本、可达性、携带关系、上次结果和失败原因。
对象 ID 必须来自巡检—语义对齐—审核流程，不能由自由文本临时猜测。

## 当前可立即使用的部分

只验证和输出家庭导航 Mission，不启动 Isaac：

```bash
./mobilemanibench.sh dual-agent \
  --navigation-scene home \
  --command '请带我到客厅沙发旁'
```

执行纯导航时，会通过兼容桥调用现有 `home-vln-formal`：

```bash
./mobilemanibench.sh dual-agent --execute \
  --navigation-scene home \
  --command '请带我到客厅沙发旁'
```

交互任务必须显式提供审核对象 ID；也可给一个审核地点描述，让 VLN 先到目标区域：

```bash
./mobilemanibench.sh dual-agent \
  --navigation-scene home \
  --command '去厨房拿起红色杯子' \
  --object-id cup_red_03 \
  --region-hint '请导航到已审核的厨房操作区'
```

完整长任务建议使用合同文件：

```bash
./mobilemanibench.sh dual-agent \
  --mission g1d_dual_brain_agent/mission.example.json
```

家庭审核语法也可直接编译“去—拿—返回”，并在一个 Isaac 会话执行：

```bash
./mobilemanibench.sh home-task \
  --headless --test --resolution 640x360 \
  --command '请带我去餐厅，拿杯子，再回到客厅沙发旁'
```

只有 `VERIFY` 取得杯体至少抬升 0.05 m、稳定保持 30 帧的证据后，Executive 才写入
`carried_object_id` 并允许返回导航。公开 OpenVLA 动作、候选抓取或单纯闭手都不能
越过该门。

默认输出：

- `outputs/g1d_dual_brain_agent/mission_result.json`
- `outputs/g1d_dual_brain_agent/world_memory.json`

这些都是运行制品，不进入 Git。相同 `mission_id` 会从 blackboard 续跑；需要重新开始时
应生成新的 ID，而不是删除旧输出。

## VLA 文件到达后的接入

当前可先运行公开 OpenVLA 的单右臂诊断链：

```bash
./mobilemanibench.sh home-dual-agent \
  --headless --test --resolution 640x360 \
  --command '请带我到客厅沙发旁' \
  --target-object houseplant \
  --openvla \
  --openvla-instruction 'move the robot hand toward the potted plant'
```

这条命令在同一 Isaac 会话完成导航、实时搜索和对齐，然后由隔离 sidecar 对当前
机载 RGB 做真实 OpenVLA 推理。返回的 7 维值按末端增量合同保存，绝不会直接当作
G1-D 七个右臂关节角。具体制品、环境和开放执行前的门槛见
`g1d_openvla/README.md`。

仍可使用旧配置的 factory 机制：

```bash
./mobilemanibench.sh dual-agent --execute \
  --mission path/to/mission.json \
  --vla-config path/to/vla_backend.json
```

factory 创建的对象至少实现旧 VLA 合同：

```python
def ready() -> bool: ...
def execute(request: Mapping[str, Any]) -> Mapping[str, Any]: ...
```

为了跑通 v2 的完整闭环，同一个集成对象还应实现：

```python
def search_object(request): ...
def approach_and_align(request): ...
def observe_readiness(request): ...
def recover_readiness(request): ...
def verify_task(request): ...
```

`search_object`、`approach_and_align` 和 `verify_task` 返回：

```json
{
  "status": "succeeded",
  "success": true,
  "failure_code": "none",
  "message": "human-readable result",
  "object_updates": [
    {
      "object_id": "cup_red_03",
      "visible": true,
      "detection_confidence": 0.93,
      "local_pose": {"frame_id": "base_link", "xyz": [0.78, 0.02, 0.74]},
      "global_pose": {"frame_id": "map", "xyz": [3.2, -1.1, 0.74]},
      "observation_source": "head_rgb_sam3_tracker",
      "reachable": true
    }
  ]
}
```

搜索成功且明确返回 `visible: true` 时，adapter 会记录接收时刻；其他时候必须由后端
报告新鲜观测。失败必须使用 `models.py` 中的 `FailureCode`，否则按
`adapter_error` 处理。

VLA 接入时仍要完成下面这些工程动作：

1. 固定相机名称、RGB 预处理、归一化、时间同步和 `base_link`/`AGV_link` TF。
2. 把 VLA 动作映射到 G1-D 的右臂和多指手，不复用 MobileManiBench 官方 G1 夹爪空间。
3. 用 `interaction_profiles.json` 为每个 `(object_id, action)` 实测距离、视角、
   IK、底盘停止和碰撞阈值。
4. `APPROACH_AND_ALIGN` 只生成 occupancy/footprint 安全且手臂可达的 SE(2)
   底盘候选，制动后重新观测，再允许 VLA。
5. `VERIFY` 使用任务物理事实（抬升、稳定持有、释放、支撑关系），不能只复述 VLA
   自己的成功字段。
6. 先在同一 Isaac SimulationApp 做连续闭环，再经过独立急停、制动和
   sim-to-real 验收开放物理 G1-D 输出。

当前框架已预留这些接口，但没有把“接口存在”表述成 VLA、家庭抓取或真机验收成功。
