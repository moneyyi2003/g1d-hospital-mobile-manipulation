#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${1:-outputs/family_home_vln/grasp_demos_integrated}"
OUTPUT_DIR="${2:-checkpoints/openvla-g1d-grasp-lora}"
EPOCHS="${3:-10}"

exec docker run --name openvla-g1d-training \
  --gpus '"device=6,7"' \
  --ipc host \
  --entrypoint /usr/bin/env \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONNOUSERSITE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/workspace:/workspace/envs/openvla/lib/python3.10/site-packages:/workspace/envs/mobilemanibench/lib/python3.10/site-packages \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  isaac-family-home-gui:6.0.1 \
  /workspace/.conda/envs/vln/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node 2 \
  /workspace/scripts/train_openvla_grasp.py \
  --demo-dir "/workspace/${DEMO_DIR}" \
  --model /workspace/checkpoints/openvla-7b \
  --output "/workspace/${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --gpus 1 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --gradient-accumulation-steps 4 \
  --num-workers 2
