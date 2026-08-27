#!/usr/bin/env python3
"""Quick script: run LingBot-Map on a video and export a PLY point cloud."""

import sys
sys.path.insert(0, "/root/autodl-tmp/lingbot_semantic_nav/third_party/lingbot-map")

import numpy as np
import torch
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO = "/root/autodl-tmp/vedio/webcammictest1.mp4"
CHECKPOINT = "/root/autodl-tmp/checkpoints/lingbot-map-long.pt"
OUTPUT_DIR = Path("/root/autodl-tmp/outputs/lingbot_demo_long")
FPS = 10
IMAGE_SIZE = 518
PATCH_SIZE = 14
NUM_SCALE_FRAMES = 8
CONF_THRESHOLD = 1.5
DOWNSAMPLE = 5  # keep every Nth point

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import official demo ──────────────────────────────────────────────────────
import importlib.util
demo_path = Path("/root/autodl-tmp/lingbot_semantic_nav/third_party/lingbot-map/demo.py")
spec = importlib.util.spec_from_file_location("lingbot_demo", demo_path)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

# ── Load video ────────────────────────────────────────────────────────────────
print(f"Loading video: {VIDEO}")
images, paths, image_folder = demo.load_images(
    video_path=VIDEO, fps=FPS,
    image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
)
print(f"  Frames: {images.shape[0]}, shape: {images.shape}")

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda")
print(f"Loading checkpoint: {CHECKPOINT}")
from lingbot_map.models.gct_stream import GCTStream

model = GCTStream(
    img_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
    enable_3d_rope=True, max_frame_num=1024,
    kv_cache_sliding_window=64, kv_cache_scale_frames=8,
    kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
    use_sdpa=True, camera_num_iterations=4,
)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model = model.to(device).eval()
print("  Model loaded.")

# ── Inference ─────────────────────────────────────────────────────────────────
images = images.to(device)
num_scale = min(NUM_SCALE_FRAMES, images.shape[0])
dtype = torch.bfloat16

print(f"Running inference ({images.shape[0]} frames)...")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    predictions = model.inference_streaming(
        images,
        num_scale_frames=num_scale,
        keyframe_interval=1,
        output_device=torch.device("cpu"),
    )

predictions, images_cpu = demo.postprocess(predictions, predictions.get("images", images))
print("  Inference done.")

# ── Build point cloud ─────────────────────────────────────────────────────────
print("Building point cloud...")
all_points = []
all_colors = []

# Unbatch images
if images_cpu.ndim == 5 and images_cpu.shape[0] == 1:
    images_cpu = images_cpu[0]
images_np = images_cpu.detach().cpu().numpy()

world_pts = predictions.get("world_points")
if world_pts is not None:
    if world_pts.ndim == 5 and world_pts.shape[0] == 1:
        world_pts = world_pts[0]
    pts_np = world_pts.detach().cpu().numpy()
else:
    # Fallback: unproject depth ourselves
    print("  No world_points in predictions, unprojecting from depth...")
    depth = predictions["depth"]
    intrinsic = predictions["intrinsic"]
    if depth.ndim == 5: depth = depth[0]
    if intrinsic.ndim == 4: intrinsic = intrinsic[0]
    pts_list = []
    for i in range(depth.shape[0]):
        d = depth[i].detach().cpu().numpy().squeeze()
        K = intrinsic[i].detach().cpu().numpy()
        h, w = d.shape
        rows, cols = np.indices((h, w), dtype=np.float64)
        x = (cols - K[0, 2]) * d / K[0, 0]
        y = (rows - K[1, 2]) * d / K[1, 1]
        cam = np.stack((x, y, d), axis=-1)

        c2w = predictions.get("extrinsic")
        if c2w is not None:
            if c2w.ndim == 4: c2w = c2w[0]
            T = c2w[i].detach().cpu().numpy()
            if T.shape == (3, 4):
                cam_h = cam.reshape(-1, 3)
                world = cam_h @ T[:3, :3].T + T[:3, 3]
                pts_list.append(world.reshape(h, w, 3))
        else:
            pts_list.append(cam)
    pts_np = np.stack(pts_list, axis=0)

# Confidence filter
conf = predictions.get("world_points_conf")
if conf is not None:
    if conf.ndim == 4 and conf.shape[0] == 1:
        conf = conf[0]
    conf_np = conf.detach().cpu().numpy()
else:
    conf_np = np.ones(pts_np.shape[:-1], dtype=np.float32)

print(f"  Points shape: {pts_np.shape}, Images shape: {images_np.shape}")

for i in range(pts_np.shape[0]):
    pts = pts_np[i].reshape(-1, 3)
    confidence = conf_np[i].reshape(-1)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(confidence) & (confidence > 0.01)
    if valid.any():
        thresh = np.quantile(confidence[valid], 0.3)
        valid &= confidence >= thresh

    if valid.sum() == 0:
        continue

    pts = pts[valid][::DOWNSAMPLE]
    rgb = images_np[i].transpose(1, 2, 0).reshape(-1, 3)[valid][::DOWNSAMPLE]
    if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.0:
        rgb = (rgb * 255).clip(0, 255)
    rgb = rgb.astype(np.uint8)

    all_points.append(pts.astype(np.float32))
    all_colors.append(rgb)

if not all_points:
    print("ERROR: No points survived filtering!")
    sys.exit(1)

points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)
print(f"  Total points: {points.shape[0]:,}")

# ── Save PLY ──────────────────────────────────────────────────────────────────
ply_path = OUTPUT_DIR / "pointcloud.ply"
print(f"Saving PLY to: {ply_path}")

vertex = np.empty(points.shape[0], dtype=[
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
vertex["x"], vertex["y"], vertex["z"] = points.T
vertex["red"], vertex["green"], vertex["blue"] = colors.T

header = (
    "ply\nformat binary_little_endian 1.0\n"
    f"element vertex {points.shape[0]}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
).encode("ascii")

with ply_path.open("wb") as f:
    f.write(header)
    vertex.tofile(f)

print(f"  Done! {points.shape[0]:,} points → {ply_path}")
print(f"  File size: {ply_path.stat().st_size / 1024 / 1024:.1f} MB")
