#!/bin/bash
set -e

# Stop any old scene viewer container
docker rm -f MaMingyi-scene-viewer 2>/dev/null || true

echo "🚀 Launching living_room scene viewer in Isaac Sim with VNC..."
echo "   GPU: 1 (GeForce RTX 4090)"
echo "   VNC will be on port 6080"
echo ""

docker run -d \
  --name MaMingyi-scene-viewer \
  --entrypoint bash \
  --runtime=nvidia \
  --gpus '"device=GPU-93e638ca-577d-ed18-0b22-5ec1328ab31e"' \
  -p 127.0.0.1:6080:6080 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v /data/IsaacSimWorkspaces/MaMingyi:/workspace \
  -v /data/MaMingyi/robot-vln/scripts/_entrypoint_scene_viewer.sh:/entrypoint.sh \
  -w /workspace \
  isaac-sim:6.0.1-nvenc-fix \
  /entrypoint.sh

echo ""
echo "✅ Container started!"
echo "   Check logs: docker logs -f MaMingyi-scene-viewer"
echo ""
echo "📌 On your Windows machine, run:"
echo "   ssh -N -L 6080:127.0.0.1:6080 -p 10027 MaMingyi@14.17.59.253"
echo ""
echo "   Then open in browser: http://127.0.0.1:6080/vnc.html?autoconnect=true"
echo ""
echo "   (VNC password: g1dview)"
echo ""
echo "📌 To stop: docker rm -f MaMingyi-scene-viewer"
