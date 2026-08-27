#!/bin/bash
# Rebuild container with Xorg (NVIDIA driver) instead of Xvfb
# This fixes Vulkan swapchain creation for Isaac Sim 6.0.1 rendering
set -e

CONTAINER="isaac-family-home-gpu67"
echo "=== Stopping and removing old container ==="
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true

echo "=== Starting container with Xorg-enabled entrypoint ==="
docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  -p 127.0.0.1:6013:6013 \
  -p 127.0.0.1:6080:6080 \
  -v /data/MaMingyi/robot-vln:/workspace \
  isaac-family-home-gui:6.0.1 \
  /workspace/scripts/_entrypoint.sh

echo "=== Container started. Waiting for Isaac Sim initialization (~60s) ==="
echo ""
echo "Check logs:  docker logs -f $CONTAINER"
echo ""
echo "SSH tunnel:"
echo "  ssh -L 16013:127.0.0.1:6013 -L 16080:127.0.0.1:6080 -p 10027 MaMingyi@14.17.59.253"
echo ""
echo "Then open:"
echo "  http://127.0.0.1:16013/  (control page)"
echo "  http://127.0.0.1:16080/vnc.html  (Isaac Sim desktop)"
