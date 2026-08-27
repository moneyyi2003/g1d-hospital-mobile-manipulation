#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFT_PYTHON="${PROJECT_ROOT}/.conda/envs/openvla-oft/bin/python"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-1}"
NPROC="${G1D_OFT_NPROC:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export WANDB_MODE="disabled"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec "${OFT_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC}" \
  "${PROJECT_ROOT}/scripts/train_openvla_oft_g1d.py" \
  "$@"
