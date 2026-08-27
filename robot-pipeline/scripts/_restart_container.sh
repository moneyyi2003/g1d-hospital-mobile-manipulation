#!/bin/bash
# Auto-generated: recreate isaac-family-home container with external port binding
set -e

CONTAINER="isaac-family-home-gpu67"
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  -p 6013:6013 \
  -p 6080:6080 \
  -v /data/MaMingyi/robot-vln:/workspace \
  isaac-family-home-gui:6.0.1 \
  /bin/bash -lc '
set -e
x11vnc -storepasswd g1dview /tmp/isaac-vnc.pass
Xvfb :1 -screen 0 1440x900x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
x11vnc -display :1 -forever -shared -rfbauth /tmp/isaac-vnc.pass -rfbport 5901 >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 6080 localhost:5901 >/tmp/novnc.log 2>&1 &
export DISPLAY=:1
exec /isaac-sim/python.sh run_g1d_simple_room_vln.py \
  --scene-profile family-home \
  --output-dir /workspace/outputs/family_home_vln \
  --map /workspace/outputs/family_home_vln/lingbot_map/map.yaml \
  --places /workspace/outputs/family_home_vln/places_formal.json \
  --objects /workspace/outputs/family_home_vln/objects_formal.json \
  --dual-agent --family-task --openvla --execute-sim-pick \
  --live-search-frames 3 --assisted-motion-scale 2.5 \
  --interactive-port 6013 --interactive-host 0.0.0.0
'

echo "Container $CONTAINER started"
echo "Access: http://14.17.59.253:6013/"
