#!/bin/bash
set -e

# --- Environment setup for g1d-expert COFFEE CUP collection ---
export PYTHONPATH="/workspace/g1d-expert-MaChuanhao/g1d-expert:$PYTHONPATH"

echo "[g1d-expert-cup] Running MaChuanhao's expert with COFFEE CUP target"
echo "[g1d-expert-cup] PYTHONPATH=$PYTHONPATH"
echo ""
echo "[g1d-expert-cup] Key expert config changes vs red-block:"
echo "  - grasp_point_z_offset_m: 0.0 (grasp at cup center, hollow object)"
echo "  - pregrasp_clearance_m: 0.18 (taller than 30mm block)"
echo "  - minimum_lift_observed_m: 0.03 (cup is larger)"
echo "  - lift_height_m: 0.20 (higher lift for taller object)"
echo "  - target: /World/OperationTable/CoffeeCup"
echo ""

cd /workspace/g1d-expert-MaChuanhao/g1d-expert

exec /isaac-sim/python.sh scripts/mock_vln_entry.py \
  --headless \
  --no-webrtc \
  --scene scene/expert_collection_scene_cup.usda \
  --task contracts/cup_pick_task.json \
  --expert-config contracts/cup_expert_config.json \
  --control-config contracts/g1d_control_parameters.json \
  --output-dir /workspace/outputs/g1d_cup_expert
