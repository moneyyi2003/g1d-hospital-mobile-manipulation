# G1-D Hospital 语义导航

当前 Hospital MVP 把 Isaac Sim 和 `lingbot_semantic_nav` 串成三段，所有输出统一写入
`outputs/hospital_vln/`。

## 1. 机器人巡检与 RGB 采集

```bash
./mobilemanibench.sh hospital-survey --headless \
  --resolution 640x360 \
  --record-gif outputs/hospital_vln/survey.gif
```

这一步加载 `Assets/room/IsaacSim/Hospital.usd` 和
`Assets/g1_d_robot/g1_d.usd`，让 G1-D 在医院前台/候诊区巡检。机器人头部 RGB
写入 `survey/rgb/`，相机内参和离线对齐位姿写入
`survey/capture_manifest.json`，第三人称 Isaac 画面写入 `survey.gif`。

正式 LingBot 输入只有 RGB。manifest 中的 Isaac 相机位姿仅用于模型推理完成后的米制
Sim(3) 对齐，不参与深度、位姿或点云预测。

## 2. LingBot 点云和 occupancy map

LingBot 推理使用独立环境 `envs/lingbot-map`（PyTorch 2.8 + CUDA 12.8），
以支持 Blackwell `sm_120` GPU，不会改动 Isaac Sim / MobileManiBench 的锁定环境。

```bash
./mobilemanibench.sh hospital-map
```

输出包括：

- `lingbot/predictions/`：LingBot-Map RGB-only 深度、位姿与 world points；
- `lingbot_to_hospital.json`：预测轨迹到医院米制坐标的鲁棒 Sim(3)；
- `lingbot_map/lingbot_map_metric.ply`：彩色三维点云；
- `lingbot_map/map.pgm` 和 `map.yaml`：ROS occupancy map；
- `map_preview/rgb_pointcloud.png`：点云俯视图；
- `map_preview/occupancy.png`：occupancy map 可视化。

对齐默认为 `auto`：只有在预测全局轨迹通过 0.45 m 内点验收时才使用单一
Sim(3)。如果 LingBot 全局视觉里程计在合成医院序列上漂移，则改用明确标注的
`pose-anchored` 离线融合：模型仍只接收 RGB，深度尺度由已知相机高度与预测地面估计，
推理完成后再用 survey 相机位姿融合。`metric_map_manifest.json` 会将
`habitat_pose` 和 `geometry_fusion` 如实记录，不将其冒充为纯视觉全局建图。

如果推理已完成，默认复用 `lingbot/lingbot_manifest.json`；重新推理使用
`--force-inference`。

## 3. Isaac Sim 语言导航

在正式 SAM3 地点库生成前，可以显式使用仅用于联调的 USD 测量地图：

```bash
./mobilemanibench.sh hospital-vln --headless --allow-bootstrap \
  --command "请带我到医院前台" \
  --record-gif outputs/hospital_vln/navigation.gif
```

正式模式不加 `--allow-bootstrap`，会加载
`lingbot_map/map.yaml + places_formal.json`。地点坐标必须来自经过审核的地点库，语言模型
只选择地点 ID。

当前第一阶段覆盖前台和候诊大厅。bootstrap 网格只有经测量的桌椅边界，
不包含全院墙体；因此主走廊在形式地图完成前标记为待审核，不会被语言指令解析为
可导航目标。全院巡检将在墙体 occupancy 经验证后扩展。

### 候诊区无相机回归验收

```bash
./mobilemanibench.sh hospital-vln --headless --test --no-camera \
  --command '请带我到候诊区'
```

该命令使用正式 `lingbot_map/map.yaml + places_formal.json`，不会回退到
bootstrap 地图。`--test` 会在 1200 帧内断言机器人到达且位置误差不超过
0.20 m，返回码非零表示验收失败。详细结果写入 `run_summary.json`，完整端到端
日志可保存为 `test_waiting_area.log`。

当前默认执行模式是 `stable_assisted`：轮子控制命令会写入 G1-D，同时用确定性
平面位姿更新保证场景和语义导航回归稳定。纯轮地接触物理验证需显式加
`--wheel-physics-only`，不属于这条确定性回归命令的成功判据。

## 4. Isaac Sim GUI 全程演示

在已连接图形桌面或 VNC 的终端中运行：

```bash
./mobilemanibench.sh hospital-demo
```

这会打开 Isaac Sim，用第三人称视角跟随 G1-D 从初始点到候诊区。到达后窗口保持
打开，可以自由转动视角检查机器人和 Hospital 场景，关闭 Isaac Sim 窗口后进程退出。
服务器终端若没有 `DISPLAY`/`WAYLAND_DISPLAY`，命令会给出明确提示，不会假装已经打开
一个不可见窗口。

无桌面时仍可由 Isaac Sim 的 chase camera 渲染全过程：

```bash
./mobilemanibench.sh hospital-vln --headless --command '请带我到候诊区' \
  --record-gif outputs/hospital_vln/navigation_waiting_area.gif
```

## 5. 浏览器三视图实时演示（推荐）

Hospital dashboard 把以下内容放在同一个浏览器页面中：

1. Isaac Sim RTX chase camera：实时显示 G1-D 在 Hospital 中运动；
2. LingBot-Map RGB-only 彩色点云俯视图：实时叠加机器人朝向、规划路径和实际轨迹；
3. ROS occupancy map：实时叠加同一组机器人位姿、路径和轨迹。

点云和 occupancy 是 `hospital-map` 已生成并审核的静态地图底图；运行导航时实时更新的是
机器人位姿与轨迹。该演示不会在每个导航帧重新执行 LingBot 推理，也不会把 Isaac 位姿
冒充为新的纯视觉建图结果。

启动前先确认没有其他 Isaac Kit 实例：

```bash
ps -eo pid,etime,cmd | grep '/isaacsim/kit/kit '
```

然后启动 dashboard：

```bash
./mobilemanibench.sh hospital-web --host 0.0.0.0 --port 6006
```

浏览器打开 `http://服务器地址:6006`，输入“请带我到候诊区”或
“请带我到医院前台”，点击“执行”。服务端只接受正式地点库中已审核的目标；页面会先显示
规划路径，再启动主 Isaac Sim 6.0.1 headless 进程，并以约 10 Hz 更新状态和 chase
camera。点击“停止”会向当前 Isaac 子进程发送终止请求。

AutoDL 需要把 6006 配置成 TCP/HTTP 自定义服务；也可以从本地做 TCP 隧道：

```bash
ssh -L 6006:127.0.0.1:6006 USER@SERVER
```

这条链路全部基于 HTTP/TCP，不需要 WebRTC 的 47998/UDP，因此适用于当前实例。dashboard
会拒绝在另一个 Isaac Kit 正在运行时启动任务，避免两个 Kit 实例争用 GPU；若正在运行
Isaac Streaming，应先正常停止它。

运行时文件写入 `outputs/hospital_web/`：

- `live/state.json`：指令、目标、当前状态、位姿、规划路径、实际轨迹和最终结果；
- `live/camera.jpg`：最新 Isaac chase camera 帧；
- `isaac.log`：本轮 Isaac 启动与导航日志。

这些都是可覆盖的运行输出，不进入 Git。2026-07-23 实机验收“请带我到候诊区”成功：
dashboard 从指令提交到进程结束约 57 秒，导航 1092 帧，位置误差 0.119 m，航向误差
0.117 rad；HTTP 页面、两张地图资源和 MJPEG 流均实际读取成功。该结果仍属于
`stable_assisted` 高层演示，不代表 `--wheel-physics-only` 验收通过。
