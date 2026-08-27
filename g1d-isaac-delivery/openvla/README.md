# G1-D 单右臂 OpenVLA 接入

本目录把公开 `openvla/openvla-7b` 接到家庭场景的双脑 Agent，但严格区分
“模型已经推理”和“机器人已经执行”。

OpenVLA 的输出是 7 维末端增量：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

它不是 G1-D 的 7 个右臂关节角。公开 checkpoint 还使用训练数据集自己的动作坐标系和
反归一化统计；当前用 `bridge_orig` 只能证明官方模型能读取机载 RGB 与指令并返回
BridgeData 动作，不能证明它能零样本控制 G1-D。

## 已接入的路径

```text
同一 Isaac SimulationApp
  NAVIGATE
  -> live SEARCH_OBJECT
  -> APPROACH_AND_ALIGN
  -> 保存当前 G1-D head_rgb
  -> 隔离 Python 3.10 sidecar 加载 OpenVLA 7B
  -> 返回真实 7-D action
  -> OpenVlaAction 校验
  -> G1-D 右臂 handoff
  -> 因缺少最终帧目标可见性复核、标定动作帧、碰撞 IK 和多指手映射而 fail-closed
```

推理 sidecar 不导入 Isaac，也永远不写关节。Isaac 主进程在等待时继续更新仿真并保持
底盘零速度，避免 Python 3.12/3.10、PyTorch 和 `transformers` 依赖互相污染。

本机使用 `envs/openvla`：Python 3.10、PyTorch 2.8.0+cu128、
`transformers==4.40.1`、`tokenizers==0.19.1`、`timm==0.9.10`。环境和约
15.1 GB 的 `checkpoints/openvla-7b` 都由根目录 `.gitignore` 排除。公开模型的配置和
shard 必须完整；入口会根据 `model.safetensors.index.json` 检查文件和总字节数，在
下载未完成时先阻塞，不会启动半个模型。
后续团队交付全量微调 checkpoint 时可直接替换 `--openvla-model`；若目录包含
`dataset_statistics.json`，sidecar 会按官方部署方式覆盖 config 中的动作统计，避免
继续错误使用 `bridge_orig`。新的 `unnorm_key` 仍必须与交付统计一致。

## 运行

单张已有机载 RGB：

```bash
./mobilemanibench.sh openvla-infer \
  --model /root/autodl-tmp/checkpoints/openvla-7b \
  --image /root/autodl-tmp/outputs/family_home_vln/live_search/20260730T113530Z/rgb/000006.png \
  --instruction 'move the robot hand toward the potted plant' \
  --unnorm-key bridge_orig \
  --output /root/autodl-tmp/outputs/openvla/single_frame.json
```

完整家庭 Agent：

```bash
./mobilemanibench.sh home-dual-agent \
  --headless --test --resolution 640x360 \
  --command '请带我到客厅沙发旁' \
  --target-object houseplant \
  --openvla \
  --openvla-instruction 'move the robot hand toward the potted plant'
```

成功时会生成：

- `openvla/<timestamp>/head_rgb.png`：进入 MANIPULATE 时的实时机载图像；
- `inference.json`：checkpoint、输入哈希、真实动作和耗时；
- `g1d_right_arm_handoff.json`：动作语义、安全门和未生成关节命令的原因；
- `sidecar.log`：模型加载和推理日志。

Agent 此时仍以 `blocked` 结束，但运行入口在
`pre_vla_pipeline_succeeded=true` 且 `openvla_inference_succeeded=true` 时返回
成功码。这表示“导航—搜索—对齐—模型推理”通过，不表示抓取或伸手通过。
当前 `houseplant` 也只是审核过的搜索/停靠对象，
`manipulation_ready=false`；它用于验证链路和观察模型输出，不是抓取验收物。

2026-07-30 最终实测同一 `application_id=isaac-sim-625387` 完成四阶段，OpenVLA 对最终
实时 RGB 输出
`[0.00382,0.00735,-0.01635,0.03690,0.02869,-0.06776,0.99608]`，模型加载
6.79 s、推理 1.03 s。最终图中的植物只在右上角部分出现，因此这个结果也暴露了
“地图几何对齐不等于操作视角对齐”；当前 handoff 明确把最终帧目标可见性复核列为
未通过的安全门。

## 开放右臂动作前必须补齐

1. 用 G1-D 单右臂示教数据微调 checkpoint，并保存对应反归一化统计。
2. 声明训练动作是 camera、base 还是 tool frame，完成 head/wrist camera 外参和 TF。
3. 在最终 OpenVLA RGB 上重新检测目标，拒绝过小、遮挡或贴边截断的操作视角。
4. 把末端增量交给 G1-D 右臂 IK，输出七个明确关节目标。
5. 对关节限位、速度、加速度、自碰撞、场景碰撞和底盘静止逐项门控。
6. 单独映射 G1-D 多指手；不能把 OpenVLA 的单个 gripper 标量广播到手指关节。
7. 用环境物理事实做 `VERIFY`，而不是把 VLA 自报成功当作抓取成功。
8. 仿真连续闭环通过后，才可在人工 enable、硬急停和 watchdog 下测试实体机器人。
