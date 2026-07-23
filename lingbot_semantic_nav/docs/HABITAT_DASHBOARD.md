# Habitat 轮式机器人网页

> 旧实验复现页面。它早于 SAM3 地点管线，不属于正式轮式物理/Nav2
> 验收入口；正式差速轮闭环使用 `gazebo_wheel_nav2.launch.py`。

这个网页把自然语言物品/区域指令接到 Habitat-Sim RGB 渲染。`--object-demo` 下，目标来自 LingBot-Map RGB-only 物品/区域地图，路径统一由 Nav2 在 LingBot occupancy 上规划；Habitat navmesh 不参与规划。页面同时显示：

- RGB 点云俯视投影；
- 物品 semantic 图和开放词汇语义 region 图；
- ROS occupancy 栅格图；
- 白色虚线完整规划路径；
- 青色机器人实际轨迹和当前位置；
- Habitat RGB 相机第一视角；
- 当前动作、位置、剩余直线距离、帧数和碰撞数。

## 启动

必须使用已经安装 Habitat-Sim 0.3.3 的 `habitat` Conda 环境：

```bash
cd /root/autodl-tmp/lingbot_semantic_nav
PYTHONPATH=src /root/miniconda3/envs/habitat/bin/python \
  scripts/serve_habitat_dashboard.py --port 8081
```

浏览器打开：

```text
http://127.0.0.1:8081
```

可以输入“请带我到公寓出口”“请带我到前台”或“出门左转，经过前台，到达咖啡厅”。Chrome/Edge 在允许麦克风权限后也支持语音输入。

`--object-demo` 会自动加载同一坐标系中的 RGB point cloud、物品 semantic、语义 region、region catalog 和 occupancy。自定义场景可分别通过 `--pointcloud`、`--semantic-map`、`--region-map`、`--region-catalog` 和 `--map-yaml` 指定；它们必须共享 LingBot local map 坐标系。`--show-habitat-gt` 仅用于显式对照。

## AutoDL 远程访问

服务默认只监听服务器本机。可在自己的电脑运行：

```bash
ssh -L 8081:127.0.0.1:8081 <用户名>@<服务器地址>
```

然后打开 `http://127.0.0.1:8081`。也可以使用 AutoDL 的端口映射；此时启动参数增加 `--host 0.0.0.0`。服务本身没有鉴权，不应直接暴露到公网。

## 输出

每次运行写入 `outputs/habitat_dashboard/runs/<UTC时间>/`：

- `rgb/`：机器人相机帧；
- `trace.json`：逐动作位姿和碰撞；
- `events.json`：解析目标、途经点和到达事件；
- `poses.txt`：相机到 Habitat world 的 4×4 位姿；
- `manifest.json`：指令、路线、运行参数和最终结果。

默认场景、地点库和拓扑可以分别用 `--scene`、`--places`、`--topology` 覆盖。三者必须来自同一 Habitat 坐标系，地点还必须提供 `metadata.habitat_y`。

## 切换到多场景测试航点

先运行多场景 episode 生成器：

```bash
PYTHONPATH=src /root/miniconda3/envs/habitat/bin/python \
  scripts/run_habitat_multiscene.py \
  --episodes-per-scene 4 \
  --output outputs/habitat_multiscene
```

例如用生成的城堡测试航点启动网页：

```bash
PYTHONPATH=src /root/miniconda3/envs/habitat/bin/python \
  scripts/serve_habitat_dashboard.py \
  --scene data/habitat_assets/versioned_data/habitat_test_scenes/skokloster-castle.glb \
  --places outputs/habitat_multiscene/skokloster-castle/places.json \
  --topology outputs/habitat_multiscene/skokloster-castle/topology.json \
  --output outputs/habitat_dashboard_castle \
  --port 8081
```

此时可输入“请导航到城堡测试航点二”。Van Gogh room 同理，把场景和输出子目录替换为 `van-gogh-room`。

自动生成的航点带有 `semantic_status: not_semantically_annotated`，只用于 navmesh 运动、稳定性和跨场景测试，不能作为人工审核的真实语义地点。
