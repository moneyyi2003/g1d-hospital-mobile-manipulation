#!/bin/bash
set -e
pkill -f 'scripts/tcp_forward.py.*--listen-port 6082' 2>/dev/null || true
docker rm -f isaac-mjpeg-relay 2>/dev/null || true
docker rm -f isaac-family-home-gpu67 2>/dev/null || true
docker run -d \
  --name isaac-family-home-gpu67 \
  --gpus '"device=6,7"' \
  -e PYTHONUNBUFFERED=1 \
  --health-cmd 'python3 -c "import socket; socket.create_connection((\"127.0.0.1\",6013),2).close()"' \
  --health-interval 10s \
  --health-timeout 3s \
  --health-start-period 60s \
  --health-retries 6 \
  -p 127.0.0.1:6013:6013 \
  -p 127.0.0.1:6014:6014 \
  -p 127.0.0.1:6080:6080 \
  -p 127.0.0.1:6082:6082 \
  -v /data/MaMingyi/robot-vln:/workspace \
  -w /workspace \
  isaac-family-home-gui:6.0.1-xorg \
  /workspace/scripts/_entrypoint_family_cup_vla.sh
echo "Started.  Open http://127.0.0.1:6013/ on the server browser,"
echo "  Expert data replay: http://127.0.0.1:6014/"
echo "  tunnel from your own computer: ssh -N -L 6013:127.0.0.1:6013 -L 6014:127.0.0.1:6014 -L 6080:127.0.0.1:6080 -L 6082:127.0.0.1:6082 -p 10027 MaMingyi@14.17.59.253"
echo "  then open http://127.0.0.1:6013/ and smooth view http://127.0.0.1:6082/ locally."
