#!/bin/bash
set -euo pipefail

# Isaac's RTX renderer needs a real NVIDIA Xorg display on this host.  The
# container supplies a matching Vulkan userspace stack, avoiding the broken
# host ICD while keeping all project data on the bind-mounted workspace.
xorg_conf=/tmp/family-dashboard-xorg.conf
gpu_pci_id="$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | head -n 1)"
gpu_bus_hex="$(printf '%s' "$gpu_pci_id" | cut -d: -f2)"
gpu_bus_dec="$((16#$gpu_bus_hex))"

printf '%s\n' \
  'Section "ServerLayout"' \
  '    Identifier "Layout0"' \
  '    Screen 0 "Screen0" 0 0' \
  'EndSection' \
  'Section "Files"' \
  '    ModulePath "/usr/lib/x86_64-linux-gnu/nvidia/xorg"' \
  '    ModulePath "/usr/lib/xorg/modules"' \
  'EndSection' \
  'Section "Device"' \
  '    Identifier "Card0"' \
  '    Driver "nvidia"' \
  "    BusID \"PCI:${gpu_bus_dec}:0:0\"" \
  '    Option "AllowEmptyInitialConfiguration" "True"' \
  '    Option "UseDisplayDevice" "None"' \
  'EndSection' \
  'Section "Screen"' \
  '    Identifier "Screen0"' \
  '    Device "Card0"' \
  '    DefaultDepth 24' \
  '    SubSection "Display"' \
  '        Depth 24' \
  '        Virtual 1920 1080' \
  '    EndSubSection' \
  'EndSection' > "$xorg_conf"

rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
Xorg :1 -ac -noreset +extension GLX +extension RENDER +extension RANDR \
  -config "$xorg_conf" >/tmp/family-dashboard-xorg.log 2>&1 &

for attempt in $(seq 1 30); do
  if [ -S /tmp/.X11-unix/X1 ]; then
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    tail -80 /tmp/family-dashboard-xorg.log
    exit 1
  fi
  sleep 1
done

export DISPLAY=:1
export G1D_ISAAC_PYTHON=/isaac-sim/python.sh
export G1D_ACTIVE_GPU=0
export G1D_RENDER_GPU_COUNT=1
export G1D_SIDECAR_CUDA_DEVICE=1
export OMNI_KIT_ACCEPT_EULA=YES

exec /workspace/.conda/envs/vln/bin/python \
  /workspace/scripts/serve_family_home_dashboard.py \
  --host 0.0.0.0 --port 6012 \
  --intent-provider deepseek --no-rule-fallback
