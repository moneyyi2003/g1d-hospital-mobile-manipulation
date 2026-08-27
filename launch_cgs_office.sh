#!/bin/bash
set -e
docker rm -f isaac-cgs-office 2>/dev/null || true
docker run -d \
  --name isaac-cgs-office \
  --gpus '"device=1"' \
  -e PYTHONUNBUFFERED=1 \
  --health-cmd 'python3 -c "import socket; socket.create_connection((\"127.0.0.1\",6013),2).close()"' \
  --health-interval 10s \
  --health-timeout 3s \
  --health-start-period 120s \
  --health-retries 6 \
  -p 127.0.0.1:6013:6013 \
  -p 127.0.0.1:6080:6080 \
  -v /data/MaMingyi/robot-vln:/workspace \
  -w /workspace \
  isaac-family-home-gui:6.0.1 \
  /workspace/scripts/_entrypoint_cgs_office.sh
echo "Started CGS office interactive session."
echo "  Open http://127.0.0.1:6013/ on the server browser."
echo "  VNC (noVNC): http://127.0.0.1:6080/vnc.html"
echo "  tunnel: ssh -N -L 6013:127.0.0.1:6013 -L 6080:127.0.0.1:6080 -p 10027 MaMingyi@14.17.59.253"
