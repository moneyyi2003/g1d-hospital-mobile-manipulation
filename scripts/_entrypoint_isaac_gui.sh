#!/bin/bash
set -e

echo "=== Installing VNC stack ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xvfb x11vnc novnc websockify > /dev/null 2>&1
echo "=== VNC stack installed ==="

# Clean up any leftover X lock files
rm -f /tmp/.X*-lock /tmp/.X11-unix/X*

# Start virtual display
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 2
echo "=== Xvfb started on :1 ==="

# Start VNC server (no password)
x11vnc -display :1 -forever -shared -nopw -rfbport 5901 &
sleep 1
echo "=== x11vnc started on 5901 ==="

# Start noVNC websockify bridge
websockify --web=/usr/share/novnc 6080 localhost:5901 &
sleep 1
echo "=== noVNC ready on http://<host>:6080 ==="

export DISPLAY=:1

echo "=== Starting Isaac Sim GUI ==="
exec /isaac-sim/runheadless.native.sh \
  --/app/content/emptyStageOnStart=false \
  --/isaac/startup/create_new_stage=false \
  --exec "open_stage.py file:///workspace/scene_asset/living_room/home_lab.usda"
