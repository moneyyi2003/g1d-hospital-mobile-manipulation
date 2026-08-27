#!/usr/bin/env python3
"""Serve a PLY point cloud in browser using viser (the same viewer LingBot-Map uses)."""

import numpy as np
from pathlib import Path
import sys

PLY_PATH = "/root/autodl-tmp/outputs/lingbot_demo_long/pointcloud.ply"
PORT = 8080

print(f"Loading PLY: {PLY_PATH}")

# Parse binary PLY header
with open(PLY_PATH, "rb") as f:
    header_lines = []
    while True:
        line = f.readline().decode("ascii").strip()
        header_lines.append(line)
        if line == "end_header":
            break

    # Parse vertex count
    vertex_count = 0
    for line in header_lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])

    print(f"  Vertex count: {vertex_count:,}")

    # Read binary data
    data = np.fromfile(f, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]), count=vertex_count)

print(f"  Points loaded: {len(data):,}")

# Downsample for smoother browser rendering
MAX_POINTS = 2_000_000
if len(data) > MAX_POINTS:
    step = max(1, len(data) // MAX_POINTS)
    data = data[::step]
    print(f"  Downsampled to: {len(data):,} points")

# Launch viser viewer
import viser
from viser import ViserServer

server = ViserServer(host="0.0.0.0", port=PORT)
print(f"\n{'='*60}")
print(f"3D Viewer ready!")
print(f"Open in browser: http://localhost:{PORT}")
print(f"If on remote server, use SSH tunnel:")
print(f"  ssh -L {PORT}:localhost:{PORT} user@server")
print(f"{'='*60}")

# Add point cloud
rgb_uint8 = np.stack([data["red"], data["green"], data["blue"]], axis=-1).astype(np.uint8)
xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)

server.scene.add_point_cloud(
    name="lingbot_map",
    points=xyz,
    colors=rgb_uint8,
    point_size=0.003,
)

print(f"Point cloud added! {len(data):,} points rendered.")
print("Press Ctrl+C to stop the server.")
print()

# Block forever
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
