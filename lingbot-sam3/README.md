# LingBot-SAM3: Video → Point Cloud + Semantic Map + Occupancy Map

从视频生成 3D 点云、语义地图、占用地图和区域分割的完整流程。

## 功能概述

给定一段 RGB 视频，本工具自动完成：

1. **LingBot-Map 推理** — 从 RGB 帧预测深度图、世界坐标点云、相机位姿（无需深度传感器）
2. **点云生成** — 融合多帧深度为全局彩色点云（PLY 格式）
3. **占用地图** — 生成 ROS 兼容的 trinary 占用栅格地图（PGM/YAML）
4. **SAM3 追踪** — 用文本描述追踪视频中的物体，生成逐帧分割 mask
5. **SAM3 → 3D 投影** — 将 SAM3 mask 通过几何关系投影到地图坐标系
6. **语义地图** — 使用 CLIPSeg 进行开放词汇 2D→3D 语义融合，生成 BEV 语义分类图
7. **区域地图** — 从占用地图提取连通区域，生成 region map

## 目录结构

```
lingbot-sam3/
├── run_pipeline.py              # 主入口脚本
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
├── pipeline/                    # 流程各步骤的实现
│   ├── lingbot_backend.py       # LingBot-Map 推理适配器
│   ├── sam3_backend.py          # SAM3 视频追踪适配器
│   ├── pointcloud.py            # 点云生成与 PLY 导出
│   ├── occupancy.py             # 占用地图构建与 ROS 导出
│   ├── semantic_map.py          # CLIPSeg 语义地图 + 区域分割
│   ├── mask_projection.py       # SAM3 mask → 3D 几何投影
│   └── alignment.py             # 相机轨迹对齐与尺度估计
├── lingbot_map/                 # LingBot-Map 模型代码（从官方仓库）
├── sam3/                        # SAM3.1 模型代码（从官方仓库）
├── checkpoints/                 # 模型权重存放目录
│   ├── lingbot-map.pt           # LingBot-Map 权重 (~70MB)
│   └── sam3.1/                  # SAM3.1 权重 (~2GB)
└── scripts/
    └── download_checkpoints.py  # 权重下载脚本
```

## 环境配置

### 1. 安装 Python 依赖

```bash
cd lingbot-sam3
pip install -r requirements.txt
```

### 2. 安装 ModelScope 并配置 git-lfs

```bash
# 安装 ModelScope（用于从魔搭社区下载模型）
pip install modelscope

# 确保 git-lfs 已安装并初始化
git lfs install
```

### 3. 获取 SAM3 模型代码

SAM3 需要从官方 GitHub 获取（Apache 兼容许可证）：

```bash
# 克隆 SAM3 仓库
git clone https://github.com/facebookresearch/sam3.git
cd sam3
# 本工具已验证的 commit（可选但推荐）
git checkout 46957e47805eaa273f4aa7bbbd25a88bca9108ce
pip install -e .
cd ..
```

然后将 `sam3/sam3/` 目录复制到 `lingbot-sam3/sam3/`：

```bash
cp -r sam3/sam3/* lingbot-sam3/sam3/
```

### 4. 下载模型权重

```bash
# 方法 A: 使用提供的脚本（一键下载）
python scripts/download_checkpoints.py --all

# 方法 B: 手动下载
# LingBot-Map: https://huggingface.co/robbyant/lingbot-map
#   → 下载 lingbot_map_stream.pt → checkpoints/lingbot-map.pt
# SAM3.1: 从魔搭社区 (ModelScope) 克隆
#   → git clone https://www.modelscope.cn/facebook/sam3.1.git checkpoints/sam3.1/
```
```

### 4. (可选) 语义地图依赖

如果需要生成语义地图和区域分割：

```bash
pip install transformers torch scipy pillow
```

首次运行时会自动下载 CLIPSeg 模型 (`CIDAS/clipseg-rd64-refined`)。

## 使用方法

### 基础用法：视频 → 点云 + 占用地图

```bash
python run_pipeline.py \
    --video input.mp4 \
    --fps 10 \
    --lingbot-checkpoint checkpoints/lingbot-map.pt \
    --output outputs/my_scene \
    --skip-sam3
```

输出：
- `outputs/my_scene/lingbot/predictions/` — 每帧的 depth/world_points/pose
- `outputs/my_scene/maps/pointcloud.ply` — 彩色点云
- `outputs/my_scene/maps/map.pgm` + `map.yaml` — ROS 占用地图

### 完整流程：视频 → 点云 + 语义地图 + 占用地图 + SAM3 物体追踪

```bash
# 1. 创建 prompts 文件
cat > prompts.txt << EOF
chair
table
sofa
bed
door
television
EOF

# 2. 运行完整流程
python run_pipeline.py \
    --video input.mp4 \
    --fps 10 \
    --lingbot-checkpoint checkpoints/lingbot-map.pt \
    --sam3-checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
    --prompts prompts.txt \
    --output outputs/my_scene
```

额外输出：
- `outputs/my_scene/sam3/` — 每个 prompt 的逐帧分割 mask
- `outputs/my_scene/maps/semantic_map.npy` — 语义分类图
- `outputs/my_scene/maps/semantic_map.png` — 语义可视化
- `outputs/my_scene/maps/region_map.npy` — 区域分割图
- `outputs/my_scene/maps/region_map.png` — 区域可视化
- `outputs/my_scene/maps/observations_*.json` — 每个物体的 3D 观测

### 使用已有 RGB 帧

```bash
python run_pipeline.py \
    --rgb-dir path/to/existing/frames/ \
    --lingbot-checkpoint checkpoints/lingbot-map.pt \
    --output outputs/my_scene
```

### 跳过 LingBot 推理（使用已有预测结果）

```bash
python run_pipeline.py \
    --rgb-dir path/to/frames/ \
    --lingbot-checkpoint checkpoints/lingbot-map.pt \
    --output outputs/my_scene \
    --no-lingbot \
    --skip-sam3
```

### 自定义语义标签

```bash
python run_pipeline.py \
    --video input.mp4 \
    --lingbot-checkpoint checkpoints/lingbot-map.pt \
    --output outputs/my_scene \
    --semantic-labels "floor,wall,door,bed,desk,computer,bookshelf,window"
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 输入视频文件路径 | - |
| `--rgb-dir` | 已有 RGB 帧目录（与 --video 二选一） | - |
| `--fps` | 视频抽帧帧率 | 10 |
| `--first-k` | 只使用前 K 帧 | 全部 |
| `--stride` | 帧采样步长 | 1 |
| `--output` | 输出目录（必填） | - |
| `--lingbot-checkpoint` | LingBot-Map 权重路径（必填） | - |
| `--lingbot-mode` | 推理模式: streaming / windowed | streaming |
| `--no-lingbot` | 跳过 LingBot 推理 | False |
| `--sam3-checkpoint` | SAM3.1 权重路径 | - |
| `--prompts` | 物体描述文本文件（一行一个） | - |
| `--skip-sam3` | 跳过 SAM3 追踪 | False |
| `--sam3-threshold` | SAM3 概率阈值 | 0.5 |
| `--resolution` | 占用地图分辨率（米） | 0.05 |
| `--ground-z` | 地面高度 Z（米） | 0.0 |
| `--scale` | 米/LingBot单位（自动估计） | 自动 |
| `--no-semantic` | 跳过语义地图生成 | False |
| `--semantic-labels` | 自定义语义标签（逗号分隔） | 默认12类 |
| `--visualize` | 启动交互式 3D 查看器 | False |

## 输出文件说明

### LingBot 预测 (`lingbot/predictions/frame_XXXXXX.npz`)

每个 NPZ 文件包含：
- `images`: RGB 图像 (H, W, 3), uint8
- `depth`: 预测深度图 (H, W), float32
- `depth_conf`: 深度置信度 (H, W), float32
- `world_points`: 世界坐标点云 (H, W, 3), float32
- `world_points_conf`: 点云置信度 (H, W), float32
- `intrinsic`: 相机内参 (3, 3), float32
- `camera_to_world`: 相机到世界变换 (3, 4), float32

### 点云 (`maps/pointcloud.ply`)

二进制 PLY 格式，包含 xyz 坐标和 RGB 颜色，可用 MeshLab、CloudCompare 或 Python open3d 打开。

### 占用地图 (`maps/map.{pgm,yaml}`)

ROS `nav_msgs/OccupancyGrid` 兼容格式：
- 0: 空闲
- 100: 占用
- -1: 未知

可用 `map_server` 加载到 ROS 导航栈。

### SAM3 追踪 (`sam3/<prompt>/frame_XXXXXX.npz`)

每个 NPZ 文件包含：
- `object_ids`: SAM3 内部物体 ID
- `track_ids`: 命名空间追踪 ID（如 `p0:o3`）
- `masks`: 二值分割 masks (N, H, W), uint8
- `scores`: 置信度分数 (N,), float32
- `boxes_xywh`: 边界框 (N, 4), float32

### 语义地图 (`maps/semantic_map.npy` + `.png`)

uint16 数组，每个 cell 的值为语义类别 ID（0=未分类，1=第1类...）。PNG 是彩色可视化。

### 区域地图 (`maps/region_map.npy` + `.png`)

int32 数组，每个 cell 的值为所属连通区域 ID（0=非空闲区域）。PNG 是彩色可视化。

### 3D 观测 (`maps/observations_<prompt>.json`)

每个 SAM3 追踪到的物体实例的 3D 质心、边界框、置信度等。

## 硬件要求

- **GPU**: NVIDIA GPU with CUDA support (推荐 ≥12GB VRAM)
- **LingBot-Map**: ~6GB VRAM
- **SAM3.1**: ~8GB VRAM（可开启 CPU offload 减少显存）
- **CLIPSeg**: ~2GB VRAM

## 常见问题

### Q: LingBot-Map 推理报 CUDA out of memory

减小图像分辨率或减少同时处理的帧数：
- 编辑 `pipeline/lingbot_backend.py` 中的 `LingBotInferenceConfig` 参数
- 使用 `--lingbot-mode windowed` 减少窗口大小

### Q: SAM3 追踪报错

确保：
1. `sam3/` 目录包含完整的 SAM3 代码
2. SAM3 权重文件存在且完整（~2GB 的 .pt 文件 + config）
3. 视频格式为常见格式（MP4/H.264 推荐）

### Q: 语义地图生成失败

这通常是因为 CLIPSeg 模型未下载。确保：
1. 安装了 `transformers` 包
2. 网络可访问 Hugging Face Hub
3. 或使用 `--no-semantic` 跳过

### Q: 想用自定义的相机位姿

可以使用 `--rgb-dir` + `--survey-manifest` 模式（编辑 `run_pipeline.py` 中的 scale 估计逻辑，替换为你的位姿数据）。

## 引用

- **LingBot-Map**: [Geometric Context Transformer for Streaming 3D Reconstruction](https://github.com/robbyant/lingbot-map)
- **SAM3/SAM3.1**: [Segment Anything with Concepts](https://github.com/facebookresearch/sam3)
- **CLIPSeg**: [Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg)
