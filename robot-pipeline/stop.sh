#!/bin/bash
set -euo pipefail

container_name="${PIPELINE_CONTAINER:-robot-pipeline-6012}"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  docker stop "$container_name" >/dev/null
  echo "已停止 ${container_name}"
else
  echo "${container_name} 未创建"
fi
