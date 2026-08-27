#!/bin/bash
set -euo pipefail

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  xserver-xorg-core x11vnc novnc websockify >/dev/null

gpu_pci_id="$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | head -n 1)"
gpu_bus_hex="$(printf '%s' "$gpu_pci_id" | cut -d: -f2)"
gpu_bus_dec="$((16#$gpu_bus_hex))"
cat > /etc/X11/xorg.conf <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection
Section "Files"
    ModulePath "/usr/lib/x86_64-linux-gnu/nvidia/xorg"
    ModulePath "/usr/lib/xorg/modules"
EndSection
Section "Device"
    Identifier "Card0"
    Driver "nvidia"
    BusID "PCI:${gpu_bus_dec}:0:0"
    Option "AllowEmptyInitialConfiguration" "True"
    Option "UseDisplayDevice" "None"
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Card0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1920 1080
    EndSubSection
EndSection
EOF

rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
x11vnc -storepasswd g1dview /tmp/isaac-vnc.pass
Xorg :1 -ac -noreset +extension GLX +extension RENDER +extension RANDR \
  -config /etc/X11/xorg.conf >/tmp/xorg.log 2>&1 &
for _ in $(seq 1 30); do
  [ -S /tmp/.X11-unix/X1 ] && break
  sleep 1
done
test -S /tmp/.X11-unix/X1
x11vnc -display :1 -geometry 1920x1080 -forever -shared \
  -rfbauth /tmp/isaac-vnc.pass -rfbport 5901 -noxdamage -snapfb \
  >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 6080 localhost:5901 >/tmp/novnc.log 2>&1 &

export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
exec /isaac-sim/python.sh /workspace/run_g1d_simple_room_vln.py \
  --scene-profile cgs-office \
  --output-dir /workspace/outputs/cgs_office_vln \
  --interactive-port 6013 \
  --interactive-host 0.0.0.0 \
  --resolution 960x720 \
  --assisted-motion-scale 2.0 \
  --arrival-hold-seconds 0 \
  > /workspace/outputs/cgs_office_vln_gui.log 2>&1
