# Robot Pipeline

这是家庭服务机器人导航系统的独立交付目录。当前执行环境为 Isaac Sim；真机接入后，网页中的第一人称画面、底盘状态和控制适配器可替换为真实机器人数据源。

## 当前包含

- Web 地图包导入：Occupancy、Semantic、Point Cloud、Region 四层。
- DeepSeek 严格语义解析：模型从审核地点库中选择 `place_id`，不生成坐标、不规划路径，也不会静默退回关键词规则。
- VLN 导航：读取正式 occupancy 和地点库，完成路径规划、底盘导航、实时定位与轨迹发布。
- Dual Brain 调度：导航/搜索/对齐由 VLN 侧执行；抓取阶段独立路由到 VLA 适配器。
- 双实时画面：第三人称全局视角和机器人第一人称视角。
- Web 遥测：地图位置、线速度、角速度、任务阶段和执行日志。
- Isaac Sim 家庭场景、G1-D 资产、正式地图制品、Florence 实时找物模型及网页运行环境。

OpenVLA 基座权重和训练轨迹没有复制到本目录。它们不是当前导航演示的必需文件，而且基座约 15 GB。VLA 接口和 Dual Brain 路由代码已保留；需要恢复仿真 VLA 时，可通过外部目录挂载对应权重。真机阶段建议把 `MANIPULATE` 执行器替换为遥操作/VLA 控制适配器。

## 启动

前提：本机已安装 Docker、NVIDIA Container Toolkit，并已有镜像 `isaac-family-home-gui:6.0.1-xorg`。

```bash
cd /data/MaMingyi/robot-vln/robot-pipeline
./start.sh
```

默认使用物理 GPU 4、5。可以在启动前修改：

```bash
PIPELINE_GPUS=4,5 PIPELINE_PORT=6012 ./start.sh
```

浏览器打开：<http://localhost:6012/>

停止：

```bash
./stop.sh
```

## DeepSeek

`.env` 已保存当前 API 配置，文件权限为 `600`。不要把它上传到公开仓库或发送给无关人员。

启动脚本强制使用：

```text
--intent-provider deepseek --no-rule-fallback
```

因此 DeepSeek 调用失败时任务会明确报错，不会用预设关键词假装完成语义识别。

可运行以下检查；加 `--online` 会真实请求一次 DeepSeek：

```bash
./verify.sh --online
```

## 地图包

网页可以导入与 `outputs/family_home_vln` 同结构的 ZIP。最少必须包含：

- `lingbot_map/map.yaml`
- `lingbot_map/map.pgm`
- `places_formal.json`
- `objects_formal.json`
- `mapping_summary.json`
- `map_preview/occupancy.png`
- `map_preview/semantic.png`
- `map_preview/rgb_pointcloud.png`
- `map_preview/region.png`

目录中的 `family_home_map_bundle.zip` 可直接用于网页导入测试。

## Dual Brain 边界

```text
用户文本
  → DeepSeek：选择审核 place_id / object_id
  → DualBrainExecutive
      → NAVIGATE / SEARCH_OBJECT / APPROACH_AND_ALIGN：VLN 导航大脑
      → MANIPULATE：VLA 或未来遥操作适配器
      → VERIFY
      → RETURN：VLN 导航大脑
```

两个大脑通过 `dual_agent_world_memory.json` 共享任务进度和目标状态，并通过控制资源租约避免底盘与机械臂同时被不同执行器占用。
