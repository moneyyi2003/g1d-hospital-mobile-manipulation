#!/bin/bash
set -e

# --- Environment setup for g1d-expert ---
export PYTHONPATH="/workspace/g1d-expert-MaChuanhao/g1d-expert:$PYTHONPATH"

echo "[g1d-expert] Running MaChuanhao's expert collection (validation run)"
echo "[g1d-expert] PYTHONPATH=$PYTHONPATH"

cd /workspace/g1d-expert-MaChuanhao/g1d-expert

exec /isaac-sim/python.sh scripts/mock_vln_entry.py \
  --headless \
  --no-webrtc \
  --scene scene/expert_collection_scene.usda \
  --task contracts/minimal_pick_lift_drop_task.json \
  --expert-config contracts/minimal_expert_config.json \
  --control-config contracts/g1d_control_parameters.json \
  --output-dir /workspace/outputs/g1d_expert_validation
