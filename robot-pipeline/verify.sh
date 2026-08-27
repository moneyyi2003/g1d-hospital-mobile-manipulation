#!/bin/bash
set -euo pipefail

pipeline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${pipeline_root}/.conda/envs/vln/bin/python"

required=(
  "family_home_dashboard/index.html"
  "scripts/serve_family_home_dashboard.py"
  "scripts/_entrypoint_family_dashboard.sh"
  "run_g1d_simple_room_vln.py"
  "g1d_dual_brain_agent/executive.py"
  "family_home_vln/task_intent.py"
  "outputs/family_home_vln/lingbot_map/map.yaml"
  "outputs/family_home_vln/lingbot_map/map.pgm"
  "outputs/family_home_vln/places_formal.json"
  "outputs/family_home_vln/objects_formal.json"
  "outputs/family_home_vln/mapping_summary.json"
  "outputs/family_home_vln/map_preview/occupancy.png"
  "outputs/family_home_vln/map_preview/semantic.png"
  "outputs/family_home_vln/map_preview/rgb_pointcloud.png"
  "outputs/family_home_vln/map_preview/region.png"
  "Assets/g1_d_robot/g1_d.usd"
  "Assets/room/IsaacSim/SimpleRoom_flat.usd"
  "checkpoints/florence-2-base-ft/config.json"
  ".env"
)

for path in "${required[@]}"; do
  if [ ! -e "${pipeline_root}/${path}" ]; then
    echo "缺少：${path}" >&2
    exit 1
  fi
done

if ! grep -Eq '^DEEPSEEK_API_KEY=.+$' "${pipeline_root}/.env"; then
  echo "DEEPSEEK_API_KEY 未配置" >&2
  exit 1
fi

PYTHONPATH="${pipeline_root}/lingbot_semantic_nav/src" \
  "$python_bin" -m py_compile \
  "${pipeline_root}/family_home_vln/task_intent.py" \
  "${pipeline_root}/g1d_dual_brain_agent/executive.py" \
  "${pipeline_root}/scripts/serve_family_home_dashboard.py"

if [ "${1:-}" = "--online" ]; then
  cd "$pipeline_root"
  PYTHONPATH="${pipeline_root}/lingbot_semantic_nav/src" \
    "$python_bin" - <<'PY'
import json
from pathlib import Path
from family_home_vln.task_intent import FamilyTaskIntentResolver

root = Path.cwd()
places = json.loads((root / "outputs/family_home_vln/places_formal.json").read_text())
objects = json.loads((root / "outputs/family_home_vln/objects_formal.json").read_text())
result = FamilyTaskIntentResolver(
    places,
    objects,
    allow_rule_fallback=False,
).resolve("请带我去吃饭的地方")
if result.parser != "deepseek" or result.destination_place_id != "dining_area":
    raise SystemExit(f"DeepSeek 检查失败：{result}")
print("DeepSeek 在线检查通过：吃饭的地方 -> dining_area")
PY
fi

echo "Pipeline 文件检查通过"
