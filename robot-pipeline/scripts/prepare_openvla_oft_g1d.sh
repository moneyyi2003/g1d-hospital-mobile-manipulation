#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFT_PYTHON="${PROJECT_ROOT}/.conda/envs/openvla-oft/bin/python"

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec "${OFT_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_openvla_oft_g1d.py" "$@"
