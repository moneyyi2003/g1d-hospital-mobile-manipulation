#!/bin/bash
set -e

# Stop any old GUI container
docker rm -f MaMingyi-gui 2>/dev/null || true

echo "🚀 Launching Isaac Sim GUI with VNC..."
echo "   GPU: 1 (GeForce RTX 4090)"
echo "   VNC will be on port 6080"
echo ""

docker run -d \
  --name MaMingyi-gui \
  --runtime=nvidia \
  --network=host \
  --gpus '"device=GPU-93e638ca-577d-ed18-0b22-5ec1328ab31e"' \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v /data/IsaacSimWorkspaces/MaMingyi:/workspace \
  -v /data/MaMingyi/robot-vln/scripts/_entrypoint_isaac_gui.sh:/entrypoint.sh \
  -w /workspace \
  isaac-sim:6.0.1-nvenc-fix \
  bash /entrypoint.sh

echo ""
echo "✅ Container started!"
echo "   Check logs: docker logs -f MaMingyi-gui"
echo ""
echo "📌 On your Windows machine, run:"
echo "   ssh -N -L 6080:127.0.0.1:6080 -p 10027 MaMingyi@14.17.59.253"
echo ""
echo "   Then open in browser: http://127.0.0.1:6080/vnc.html?autoconnect=true"
echo ""
echo "📌 To stop: docker rm -f MaMingyi-gui"
