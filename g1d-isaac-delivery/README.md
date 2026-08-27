# G1-D Isaac Sim 交付包

本交付包包含 G1-D 人形机器人在 Isaac Sim 6.0.1 中的完整仿真资产、场景、参考脚本和 OpenVLA 基础设施，用于专家脚本开发和 VLA 模型训练。

## 目录结构

```
g1d-isaac-delivery/
├── README.md                          # 本文档
├── robot/                             # G1-D 机器人资产
│   ├── g1_d.usd                       # 主 USD（可动，非固定底座）
│   ├── g1_d_flat.usd                  # 平地配置变体
│   ├── g1_d_fixed.usd                 # 固定底座变体（调试用）
│   ├── g1_d_composed.usd              # 组合场景变体
│   ├── g1_d.urdf                      # URDF 描述（完整运动学 + 惯性参数）
│   ├── config.yaml                    # URDF→USD 转换配置
│   ├── configuration/
│   │   ├── g1_d_base.usd              # 骨架定义
│   │   ├── g1_d_physics.usd           # 物理参数（质量、惯量、碰撞体）
│   │   └── g1_d_sensor.usd            # 传感器配置
│   ├── g1_d/                          # 实例化资产（payloads, materials, physics）
│   ├── g1_d_1/                        # 第二套实例（flat 变体使用）
│   ├── source/                        # 源 USD 定义 + 全部 62 个 STL 网格
│   └── meshes/                        # 网格文件副本
├── scenes/                            # Isaac Sim 场景
│   ├── SimpleRoom.usd                 # 带围墙的简易房间
│   ├── SimpleRoom_flat.usd            # 平地版本（推荐操作任务使用）
│   └── Hospital.usd                   # 医院场景
├── scripts/                           # 参考脚本
│   ├── g1_d_smoke.py                  # 冒烟测试：验证资产加载与关节控制
│   ├── g1_d_vln.py                    # 底盘导航演示（语言指令→颜色目标点）
│   ├── manipulation_scene.py          # 程序化生成桌子 + 红色方块
│   ├── run_g1d_simple_room_vln.py     # SimpleRoom 完整任务脚本
│   └── run_g1d_agent.py / run_g1d_dual_brain_agent.py  # VLN+VLA Agent 入口
├── openvla/                           # OpenVLA 基础设施
│   ├── action_contract.py             # 7 维 delta-action 契约
│   ├── checkpoint.py                  # 模型权重完整性检查
│   ├── run_openvla_inference.py       # 单帧 RGB + 指令推理脚本
│   ├── README.md                      # OpenVLA 接入说明
│   └── tests/                         # 契约单元测试
├── assets/                            # 场景物体
│   └── family_home_objects/           # 家居物品（杯、碗、植物、书本、遥控器等 12 种）
└── docs/                              # 参考文档与数据
    ├── interaction_profiles.json      # 红色方块夹取任务交互配置
    ├── vla_backend.example.json       # VLA 后端配置样例
    ├── vlaandvln.md                   # VLN+VLA 流程设计文档
    ├── object_observation.example.json # 物体观测格式样例
    ├── g1d_right_arm_probe.json       # 右臂 IK 探针数据
    └── openvla_example/               # OpenVLA 推理输出样例
        ├── g1d_right_arm_handoff.json # 动作交接合约
        ├── inference.json             # 完整推理输出
        └── head_rgb.png               # 推理输入图像
```

---

## G1-D 机器人参数

### PD 控制器参数

以下参数已在 Isaac Sim 中验证，可直接用于 `ImplicitActuatorCfg` 或自定义控制器。

**轮组 (Wheels)**
```
joints:         Left_Wheel_Joint, Right_Wheel_Joint
effort_limit:   400.0 N·m
velocity_limit: 20.0 rad/s
stiffness:      0.0          (速度控制模式)
damping:        150.0
```

**升降机构 (Lift)**
```
joints:         LZ_mt_Joint, LZ_it_Joint
effort_limit:   500.0 N·m
velocity_limit: 1.0 rad/s
stiffness:      500.0
damping:        50.0
```

**躯干 (Torso)**
```
joints:         Yaw_Joint, torso_Joint
effort_limit:   200.0 N·m
velocity_limit: 3.0 rad/s
stiffness:      200.0
damping:        20.0
```

**手臂 (Arms)**
```
joints:         .*_(shoulder|elbow|wrist)_.*joint  (正则匹配)
effort_limit:   120.0 N·m
velocity_limit: 4.0 rad/s
stiffness:      120.0
damping:        12.0
```

**手部 (Hands)**
```
joints:         .*_hand_(thumb|middle|index)_.*joint  (正则匹配)
effort_limit:   25.0 N·m
velocity_limit: 6.0 rad/s
stiffness:      30.0
damping:        3.0
```

### 右臂关节顺序（7 DOF）

```python
G1D_RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
```

### 底盘参数
```
wheel_radius:  0.0848 m
axle_track:    0.4062 m
```

---

## Stage Prim Path 约定

在 Isaac Sim Stage 中，各组件使用以下路径：

```
G1-D 机器人:          /World/G1_D
机器人底盘:           /World/G1_D/AGV_link
右臂基座:            /World/G1_D/right_shoulder_pitch_link
右手掌:              /World/G1_D/right_hand_palm_link
右手食指指尖:        /World/G1_D/right_hand_index_1_link
左轮:                Left_Wheel_Joint
右轮:                Right_Wheel_Joint

操作桌（待放置）:     /World/Table
红色方块（待放置）:   /World/Table/RedBlock
```

专家脚本通过上述 prim path 访问机器人关节和物体，无需硬编码路径以外的配置。

---

## 使用方式

### 验证 G1-D 资产

在 Isaac Sim 6.0.1 环境中运行冒烟测试，确认资产加载和关节控制正常：

```bash
# 基础测试（无重力、无地面碰撞）
<isaac_python> scripts/g1_d_smoke.py --usd robot/g1_d.usd --steps 20

# 完整物理测试（有重力 + 地面碰撞）
<isaac_python> scripts/g1_d_smoke.py --usd robot/g1_d.usd --steps 60 --with-ground
```

### 底盘导航演示

```bash
<isaac_python> scripts/g1_d_vln.py \
    --usd robot/g1_d_flat.usd \
    --instruction "前往红色目标" \
    --max-steps 1800
```

### 完整任务运行

```bash
<isaac_python> scripts/run_g1d_simple_room_vln.py --command "请带我到沙发旁边"
```

### OpenVLA 推理

```bash
python openvla/run_openvla_inference.py \
    --model <openvla_oft_checkpoint_dir> \
    --image <rgb_image.png> \
    --instruction "pick up the red block" \
    --output inference_result.json
```

---

## 场景物体

`assets/family_home_objects/` 中提供以下可放置到场景中的 USD 资产：

| 文件 | 物体 |
|------|------|
| `cup.usd` | 杯子 |
| `living_plant/` | 盆栽 |
| `dining_bowl/` | 碗 |
| `dining_basket/` | 篮子 |
| `dining_cup/` | 餐杯 |
| `media_book/` | 书本 |
| `media_remote/` | 遥控器 |
| `bedside_lamp/` | 台灯 |
| `kitchen_appliance/` | 厨房电器 |
| `kitchen_knife_block/` | 刀架 |
| `bed_handbag/` | 手提包 |
| `media_monitor/` | 显示器 |

---

## OpenVLA Action 契约

VLA 模型输出的 action 为 7 维 delta 格式：

```python
ACTION_LABELS = (
    "delta_x",      # 末端 delta X (米)
    "delta_y",      # 末端 delta Y (米)
    "delta_z",      # 末端 delta Z (米)
    "delta_roll",   # 末端 delta roll (弧度)
    "delta_pitch",  # 末端 delta pitch (弧度)
    "delta_yaw",    # 末端 delta yaw (弧度)
    "gripper",      # 夹爪开合 (0=闭合, 1=张开)
)
```

**注意事项：**
- OpenVLA 输出的 delta 值是基于训练数据集的归一化值，需要通过 `dataset_statistics.json` 中的 `unnorm_key` 进行反归一化，才能映射到 G1-D 的实际运动范围。
- G1-D 右臂 action 交接合约（`build_g1d_right_arm_handoff`）默认处于安全阻断状态（`execution_permitted: false`），需要完成以下前置条件后解除：
  1. 微调或标定 checkpoint 适配 G1-D
  2. 确定 action 的参考坐标系（camera/base/tool frame）
  3. 目标物体在当前帧中可见且未边缘裁剪
  4. 右臂 IK 可行且自碰撞/场景碰撞检查通过
  5. 实机操作时操作员使能

详见 `openvla/action_contract.py` 和 `openvla/README.md`。

---

## 任务场景搭建指引

当前的 SimpleRoom 场景已包含房间结构和 G1-D 机器人。搭建抓取任务场景时，需补充以下元素：

1. **操作桌**：使用 Isaac Sim 内置的 Cube prim 拼合（参考 `scripts/manipulation_scene.py` 中的 `BoxSpec` 参数：桌面 1.0×0.7×0.08m，桌腿 0.07×0.07m），或导入自定义桌子 USD

2. **红色方块**：使用 Isaac Sim Cube prim + 红色材质，或从 `Props/Blocks/` 添加，放置于桌面上

3. **手腕摄像头**：在 G1-D 右手腕 `right_wrist_yaw_link` 下挂载 RGB Camera prim，参考 `robot/configuration/g1_d_sensor.usd` 的结构

4. **第三视角摄像头**：在场景中固定位置添加 Camera prim，用于数据采集时的环境视角

---

## 数据采集与训练流程

整体流程如下：

```
专家脚本采集 arm trajectory
    ↓
同时采集双视角 RGB（手腕 + 第三视角）
    ↓
打包为 RLDS 格式（action + 图像 + 语言指令 + 本体状态）
    ↓
上传 AutoDL 进行 OpenVLA-OFT 训练
    ↓
交付训练好的模型权重
```

专家脚本的输入参数包括：
- 目标物体 prim path（如 `/World/Table/RedBlock`）
- G1-D 机器人 prim path（`/World/G1-D`）
- 数采轮次等参数

专家脚本的预期输出：
- `action.jsonl`：每行为抓取轨迹中一帧的机械臂关节角与末端位姿

---

## 依赖与版本

- **Isaac Sim**: 6.0.1
- **Python**: 3.10
- **关键包**: `isaaclab`, `torch`, `numpy`
- **OpenVLA 模型**: `openvla-7b`（权重需单独获取）
- **训练框架**: OpenVLA-OFT（未修改架构，代码从 GitHub 获取）
