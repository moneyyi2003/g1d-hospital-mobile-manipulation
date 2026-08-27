#!/bin/bash
set -euo pipefail

pipeline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container_name="${PIPELINE_CONTAINER:-robot-pipeline-6012}"
host_port="${PIPELINE_PORT:-6012}"
gpu_devices="${PIPELINE_GPUS:-4,5}"
image_name="${PIPELINE_IMAGE:-isaac-family-home-gui:6.0.1-xorg}"

if ! docker image inspect "$image_name" >/dev/null 2>&1; then
  echo "缺少 Docker 镜像：$image_name" >&2
  exit 1
fi

if docker container inspect "$container_name" >/dev/null 2>&1; then
  docker rm -f "$container_name" >/dev/null
fi

docker run -d \
  --name "$container_name" \
  --gpus "\"device=${gpu_devices}\"" \
  --shm-size 2g \
  -p "127.0.0.1:${host_port}:6012" \
  -v "${pipeline_root}:/workspace:rw" \
  -w /workspace \
  --env-file "${pipeline_root}/.env" \
  -e PRIVACY_CONSENT=Y \
  -e ACCEPT_EULA=Y \
  -e PYTHONUNBUFFERED=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --health-cmd='python3 -c "import socket; socket.create_connection((\"127.0.0.1\",6012),2).close()"' \
  --health-interval=10s \
  --health-timeout=3s \
  --health-start-period=60s \
  --health-retries=6 \
  --entrypoint /bin/bash \
  "$image_name" \
  /workspace/scripts/_entrypoint_family_dashboard.sh >/dev/null

echo "Pipeline 正在启动：http://localhost:${host_port}/"
echo "容器：${container_name}；GPU：${gpu_devices}"
