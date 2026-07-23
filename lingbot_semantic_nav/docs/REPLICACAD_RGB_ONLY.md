# ReplicaCAD 多房间 RGB-only 地图实验

> 历史 OWLv2/SAM2 实验记录，仅保留为来源证据。正式主链已切换到
> 锁定的官方 SAM3 适配器和 schema-v2 审核地点库，见项目 README。

## 场景与边界

- 场景：ReplicaCAD `apt_1`，官方数据集版本 `v1.6`。
- 官方介绍与许可：<https://aihabitat.org/datasets/replica_cad/>（CC BY 4.0）。
- 导出的传感器输入只有 262 张 RGB；没有创建或导出 Habitat depth、semantic 或 camera pose。
- Habitat navmesh 只用于采集阶段保持虚拟相机位于可通行区域并连接巡检视点；navmesh、路线点和坐标均不写入 LingBot 输入，也不进入点云或地图生成器。
- 点云、深度、内参、外参和相机轨迹全部来自 LingBot-Map 模型预测。
- 当前导航地图仍是 `lingbot_local_unit_not_meters`，没有使用 Habitat 位姿做导航侧 Sim(3) 对齐。8083 使用半径 0.06 map-unit 的小型差速轮式机器人；Nav2、里程计积分和碰撞检查全部在 LingBot local map 中运行，不把 map unit 冒充经过真值标定的米制尺度。画面另有一份严格限定为 `habitat_rgb_camera_only` 的 Sim(2)，仅把 LingBot 位姿换算为 Habitat COLOR sensor 的渲染位姿。

## 已生成成果

| 成果 | 路径/统计 |
|---|---|
| RGB-only 序列 | `data/habitat/replica_cad_apt1_rgb_only/rgb/`，262 帧 |
| LingBot 预测 | `outputs/lingbot_replica_cad_apt1/rgb/`，262 个 NPZ |
| RGB 点云 | `outputs/maps/replica_cad_apt1_rgb_only_blind/lingbot_local.ply`，2,955,623 点 |
| Occupancy | 118×196，0.03 local-unit/格 |
| 画面 Sim(2) | `habitat_render_alignment.json`，262 个同帧 RGB 对应点，位置 RMSE 0.441 m |
| Semantic / Instance | 374 个有效三维观测，49 个实例，4,552 个语义栅格 |
| Region | 用餐区、走廊、入口区、卧室四个通过组件与净空检查的区域 |
| 网页 | `http://127.0.0.1:8083`，五个对齐页签、Nav2 差速轮式机器人、区域与物品目标 |

## 复现命令

下载官方场景：

```bash
/root/miniconda3/envs/habitat/bin/python -m habitat_sim.utils.datasets_download \
  --uids replica_cad_dataset \
  --data-path data/habitat_assets \
  --no-replace
```

仅导出 RGB 的多房间巡检：

```bash
/root/miniconda3/envs/habitat/bin/python scripts/collect_habitat_rgb_tour.py \
  --dataset-config data/habitat_assets/replica_cad/replicaCAD.scene_dataset_config.json \
  --scene apt_1 \
  --navmesh data/habitat_assets/replica_cad/navmeshes_default/apt_1.navmesh \
  --output data/habitat/replica_cad_apt1_rgb_only \
  --viewpoints 10 --spacing 0.22 --panorama-views 8
```

LingBot-Map RGB-only 推理：

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m lingbot_nav.cli run-lingbot \
  --repo external/lingbot-map \
  --checkpoint checkpoints/lingbot-map-long.pt \
  --input-root data/habitat/replica_cad_apt1_rgb_only \
  --output-dir outputs/lingbot_replica_cad_apt1 \
  --mode windowed --window-size 128 --keyframe-interval 5 \
  --overlap-keyframes 8 --use-sdpa \
  --python /root/miniconda3/bin/python
```

点云与 occupancy：

```bash
PYTHONPATH=src /root/miniconda3/bin/python scripts/build_rgb_only_map.py \
  --predictions outputs/lingbot_replica_cad_apt1/rgb \
  --rgb-source data/habitat/replica_cad_apt1_rgb_only/rgb \
  --output outputs/maps/replica_cad_apt1_rgb_only_blind \
  --resolution 0.03
```

仅供画面的多点 RGB 对应标定：

```bash
/root/miniconda3/envs/habitat/bin/python scripts/fit_habitat_render_alignment.py \
  --predictions outputs/lingbot_replica_cad_apt1/rgb \
  --local-frame outputs/maps/replica_cad_apt1_rgb_only_blind/lingbot_local_frame.json \
  --rgb-directory data/habitat/replica_cad_apt1_rgb_only/rgb \
  --output outputs/maps/replica_cad_apt1_rgb_only_blind/habitat_render_alignment.json
```

标定产物明确声明其唯一 consumer 是 `habitat_rgb_camera_only`，并禁止 Nav2、TF、odom、目标生成和碰撞检查加载。拟合脚本会复现 RGB 采集视点，与 LingBot 对同一 RGB 帧预测的外参按帧号配对；这个离线步骤可读取 Habitat 采集路线，但输出只进入画面渲染器。

实例/semantic 与 region 使用 `build-instance-map`、`build-region-map`，模型均指向本地 `checkpoints/`。最终网页：

```bash
PYTHONPATH=src /root/miniconda3/envs/habitat/bin/python scripts/serve_mapping_dashboard.py \
  --map-yaml outputs/maps/replica_cad_apt1_rgb_only_blind/map.yaml \
  --pointcloud outputs/maps/replica_cad_apt1_rgb_only_blind/lingbot_local.ply \
  --semantic-map outputs/maps/replica_cad_apt1_instances/semantic_map.npy \
  --instance-map outputs/maps/replica_cad_apt1_instances/instance_map.npy \
  --region-map outputs/maps/replica_cad_apt1_regions_v3/region_map.npy \
  --region-catalog outputs/maps/replica_cad_apt1_regions_v3/region_catalog.json \
  --candidates outputs/maps/replica_cad_apt1_instances/place_candidates.json \
  --initial-rgb data/habitat/replica_cad_apt1_rgb_only/rgb/000000.png \
  --host 0.0.0.0 --port 8083
```

该网页接受区域和物品导航指令。默认后端是 Nav2：网页内的小型差速轮式机器人积分 Nav2 `/cmd_vel`，发布 `/odom` 和 `odom→base_link` TF；Nav2 使用 LingBot occupancy 完成全局规划、局部控制和 footprint/costmap 碰撞检查。右侧实时视角把这个积分位姿单向映射到 Habitat COLOR sensor，以 640×480 MJPEG 连续渲染 ReplicaCAD 纹理场景。当前 apt_1 产物提供 4 个可导航区域、49 个 RGB 识别物品实例，其中 44 个具有 LingBot occupancy 可达停靠点；例如可输入 `请带我到用餐区域`、`请带我到椅子` 或 `请带我到桌子3旁边`。

8083 的在线渲染器只加载 ReplicaCAD 纹理场景、RGB sensor 和上述 render-only Sim(2)，不创建 depth/semantic sensor，不查询 navmesh/pathfinder，也不调用 `step_filter`。标定只用于将 Nav2 `/cmd_vel` 积分位姿放入画面，不会发布回 Nav2，不能参与规划、避障、目标生成或路线修正。导航地图唯一来源仍是 LingBot-Map 重建的 `map.yaml`；dashboard 启动时会核对 RGB、LingBot predictions、点云、occupancy、semantic/instance/region 产物来自同一 RGB-only manifest，来源不一致会拒绝启动。

8083 的 wheel bridge 与 dashboard session 同生命周期：任务到达、失败或取消后仍以 20 Hz 保持最后 `/odom` 和 `odom→base_link`，下一任务从该位姿继续。启动阶段的 `wheel_initial_odom` 会记住 bridge 的最后位姿，即使 bridge 释放发布权也不会跳回最初起点。

物品目标来自 OWLv2→SAM2→LingBot 2D→3D 候选。8083 会把几何可达候选标记为 `demo_enabled`，用于仿真研究，不等同于生产审核通过。接入真实机器人前仍需完成非 Habitat 真值来源的尺度标定、真实底盘 footprint 验证和独立物品复核。

8083 默认使用 ROS domain 32，与 8081 隔离。首次运行或新增 launch 后先构建：

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
cd ..
```

若只需要保留历史 A* 动画，可显式增加 `--navigation-backend preview`；正式 8083 演示不要使用该选项。
