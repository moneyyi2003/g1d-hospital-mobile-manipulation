# G1-D Family Home OpenVLA-OFT preparation

This project trains OpenVLA-OFT from successful first-person Expert
trajectories. It does not train from a directory of PNG files alone.

## Frozen data contract

- Observation: `640x480` RGB from the ego-centric G1-D head camera.
- Pinhole intrinsics: 50 mm focal length, `fx=1527.0818`, `fy=1527.0819`,
  `cx=320`, `cy=240`; clipping range is explicitly frozen to
  `[0.1 m, 1,000,000 m]` for newly collected trajectories.
- Rate: 10 Hz.
- Training image: `step_NNNN/image.png` only. `third_person.png` is audit
  evidence and is never passed to OpenVLA-OFT.
- Label: synchronized 7-D
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]` action in the world frame.
- Gripper convention: `1=open`, `0=closed`.
- Action normalization key: `g1d_family_home_cup_head`.
- OFT target: eight consecutive 7-D actions from one current RGB image.
- Accepted episodes must pass Expert success, physical execution, at least
  10 cm lift, 30 stable-hold frames, RGB black-region checks, and action-range
  checks.

Trajectories collected with the former 1.0 m near clipping plane are kept in
`expert_demos_head`; new 0.1 m trajectories are isolated in
`expert_demos_head_clip01`. Do not silently merge the two camera domains.

The deployed controller must preserve the same world-frame delta and gripper
conventions when it unnormalizes OFT output. Coordinate conversion belongs in
the execution adapter; the seven values must not be treated as seven G1-D
joint angles.

## Preflight without training

Run:

```bash
cd /data/MaMingyi/robot-vln && ./scripts/prepare_openvla_oft_g1d.sh
```

This command uses only `.conda/envs/openvla-oft`, hides all GPUs, audits the
dataset, validates the local three-shard OpenVLA checkpoint and OFT checkout,
and writes:

- `outputs/family_home_vln/openvla_oft/manifest.json`
- `outputs/family_home_vln/openvla_oft/preflight.json`

It never starts fine-tuning. `software_ready=true` means the training code,
environment, model and accepted data format are usable. `dataset_ready=true`
additionally requires the configurable project gate of 100 successful
episodes and 4000 eight-action samples.

For an end-to-end data-loader smoke test, still without model training:

```bash
cd /data/MaMingyi/robot-vln && CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 .conda/envs/openvla-oft/bin/python scripts/train_openvla_oft_g1d.py --data-smoke
```

## Training boundary

Do not start the training command recorded in `preflight.json` until
`ready_to_train=true`. The preparation script records that command for the
later training stage but does not execute it.
