#!/usr/bin/env python3
"""Full pipeline: LingBot-Map + SAM3.1 → Point Cloud + Occupancy + Semantic Maps."""

import sys, os, json, math, time, glob, argparse
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
LINGBOT_MAP_ROOT = "/root/autodl-tmp/lingbot_semantic_nav/third_party/lingbot-map"
sys.path.insert(0, LINGBOT_MAP_ROOT)
sys.path.insert(0, "/root/autodl-tmp/lingbot_semantic_nav/third_party/sam3")

import numpy as np
import torch
import cv2
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO = "/root/autodl-tmp/vedio/webcammictest1.mp4"
CHECKPOINT_LINGBOT = "/root/autodl-tmp/checkpoints/lingbot-map-long.pt"
CHECKPOINT_SAM3 = "/root/autodl-tmp/checkpoints/sam3.1/sam3.1_multiplex.pt"
OUTPUT = Path("/root/autodl-tmp/outputs/full_pipeline")
FPS = 10

# Prompts for SAM3 — general indoor objects
PROMPTS = ["floor", "wall", "chair", "table", "door", "person", "screen", "plant"]

OUTPUT.mkdir(parents=True, exist_ok=True)
PREDS_DIR = OUTPUT / "predictions"
SAM3_DIR = OUTPUT / "sam3_tracks"
MAP_DIR = OUTPUT / "maps"
for d in [PREDS_DIR, SAM3_DIR, MAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    t_total = time.time()

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: LingBot-Map Inference
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Step 1: LingBot-Map inference (RGB → depth + world_pts + poses)")
    print("=" * 60)

    import importlib.util
    demo_path = Path(LINGBOT_MAP_ROOT) / "demo.py"
    spec = importlib.util.spec_from_file_location("lingbot_demo", demo_path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    # Extract frames
    video_name = Path(VIDEO).stem
    frames_dir = OUTPUT / f"{video_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / FPS))
    saved = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = frames_dir / f"{len(saved):06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
        idx += 1
    cap.release()
    print(f"  Extracted {len(saved)} frames from {total_frames} total")

    # Load images
    images, paths, _ = demo.load_images(image_folder=str(frames_dir), image_size=518, patch_size=14)
    print(f"  Preprocessed: {images.shape}")

    # Load model
    device = torch.device("cuda")
    from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=1024, kv_cache_sliding_window=64, kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        use_sdpa=True, camera_num_iterations=4,
    )
    ckpt = torch.load(CHECKPOINT_LINGBOT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = model.to(device).eval()
    print("  Model loaded.")

    # Inference
    images_dev = images.to(device)
    num_scale = min(8, images.shape[0])
    dtype = torch.bfloat16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images_dev, num_scale_frames=num_scale,
            keyframe_interval=1, output_device=torch.device("cpu"),
        )
    predictions, images_cpu = demo.postprocess(predictions, predictions.get("images", images_dev))
    print("  Inference done.")

    # Unbatch
    if images_cpu.ndim == 5 and images_cpu.shape[0] == 1:
        images_cpu = images_cpu[0]
    images_np = images_cpu.detach().cpu().numpy()
    frame_count = images_np.shape[0]

    # Get world_points
    world_pts = predictions.get("world_points")
    if world_pts is not None:
        if world_pts.ndim == 5 and world_pts.shape[0] == 1:
            world_pts = world_pts[0]
        pts_np = world_pts.detach().cpu().numpy()
    else:
        print("  No world_points, unprojecting from depth...")
        depth = predictions["depth"]
        intrinsic = predictions["intrinsic"]
        if depth.ndim == 5: depth = depth[0]
        if intrinsic.ndim == 4: intrinsic = intrinsic[0]
        c2w_all = predictions.get("extrinsic")
        if c2w_all is not None and c2w_all.ndim == 4: c2w_all = c2w_all[0]
        pts_list = []
        for i in range(depth.shape[0]):
            d = depth[i].detach().cpu().numpy().squeeze()
            K = intrinsic[i].detach().cpu().numpy()
            h, w = d.shape
            rows, cols = np.indices((h, w), dtype=np.float64)
            x = (cols - K[0, 2]) * d / K[0, 0]
            y = (rows - K[1, 2]) * d / K[1, 1]
            cam = np.stack((x, y, d), axis=-1)
            if c2w_all is not None:
                T = c2w_all[i].detach().cpu().numpy()
                if T.shape == (3, 4):
                    cam_h = cam.reshape(-1, 3)
                    world = cam_h @ T[:3, :3].T + T[:3, 3]
                    pts_list.append(world.reshape(h, w, 3))
            else:
                pts_list.append(cam)
        pts_np = np.stack(pts_list, axis=0)

    depth_all = predictions.get("depth")
    if depth_all is not None:
        if depth_all.ndim == 5: depth_all = depth_all[0]
        depth_np = depth_all.detach().cpu().numpy()
    intrinsic_all = predictions.get("intrinsic")
    if intrinsic_all is not None:
        if intrinsic_all.ndim == 4: intrinsic_all = intrinsic_all[0]
        intrinsic_np = intrinsic_all.detach().cpu().numpy()
    c2w_all = predictions.get("extrinsic")
    if c2w_all is not None:
        if c2w_all.ndim == 4: c2w_all = c2w_all[0]
        c2w_np = c2w_all.detach().cpu().numpy()
    else:
        c2w_np = np.tile(np.eye(3, 4)[None], (frame_count, 1, 1))

    conf = predictions.get("world_points_conf")
    if conf is not None:
        if conf.ndim == 4 and conf.shape[0] == 1: conf = conf[0]
        conf_np = conf.detach().cpu().numpy()
    else:
        conf_np = np.ones(pts_np.shape[:-1], dtype=np.float32)

    # Save per-frame NPZ
    for i in range(frame_count):
        rgb = images_np[i].transpose(1, 2, 0)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).clip(0, 255)
        rgb = rgb.astype(np.uint8)
        artifact = {
            "images": rgb,
            "world_points": pts_np[i],
            "world_points_conf": conf_np[i],
            "extrinsic": c2w_np[i],
        }
        if depth_np is not None:
            artifact["depth"] = depth_np[i]
        if intrinsic_np is not None:
            artifact["intrinsic"] = intrinsic_np[i]
        np.savez_compressed(PREDS_DIR / f"frame_{i:06d}.npz", **artifact)
    print(f"  Saved {frame_count} frame predictions")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Build Point Cloud PLY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 2: Building point cloud PLY")
    print("=" * 60)

    DOWNSAMPLE = 3
    all_pts, all_cols = [], []
    for i in range(frame_count):
        pts = pts_np[i].reshape(-1, 3)
        c = conf_np[i].reshape(-1)
        valid = np.isfinite(pts).all(axis=1) & np.isfinite(c) & (c > 0.01)
        if valid.any():
            thresh = np.quantile(c[valid], 0.25)
            valid &= c >= thresh
        if valid.sum() == 0:
            continue
        pt = pts[valid][::DOWNSAMPLE]
        rgb = images_np[i].transpose(1, 2, 0).reshape(-1, 3)[valid][::DOWNSAMPLE]
        if rgb.max() <= 1.0: rgb = (rgb * 255).clip(0, 255)
        rgb = rgb.astype(np.uint8)
        all_pts.append(pt.astype(np.float32))
        all_cols.append(rgb)

    points_cloud = np.concatenate(all_pts, axis=0)
    colors_cloud = np.concatenate(all_cols, axis=0)
    print(f"  Point cloud: {points_cloud.shape[0]:,} points")

    # Save PLY
    ply_path = MAP_DIR / "pointcloud.ply"
    vertex = np.empty(points_cloud.shape[0], dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = points_cloud.T
    vertex["red"], vertex["green"], vertex["blue"] = colors_cloud.T
    with ply_path.open("wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {points_cloud.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode("ascii"))
        vertex.tofile(f)
    print(f"  Saved: {ply_path} ({ply_path.stat().st_size/1024/1024:.1f}MB)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Align and estimate scale
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 3: Scale estimation & alignment")
    print("=" * 60)

    # Estimate scale from point cloud extent
    pts_2d = points_cloud[:, :2]
    span = float(np.linalg.norm(np.quantile(pts_2d, 0.95, axis=0) - np.quantile(pts_2d, 0.05, axis=0)))
    if span > 100:
        scale = 0.01
    elif span < 0.5:
        scale = 10.0
    else:
        scale = 1.0
    alignment = np.eye(4, dtype=np.float64)
    print(f"  Scale: {scale:.4f} m/unit, Span: {span:.1f} units")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: Occupancy Map
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 4: Building occupancy map")
    print("=" * 60)

    scaled_pts = points_cloud * scale
    # Auto ground_z as the lowest z value
    ground_z = float(np.quantile(scaled_pts[:, 2], 0.05))
    print(f"  Auto ground_z: {ground_z:.3f}m")

    RES = 0.05
    q = 0.002
    lower = np.quantile(scaled_pts[:, :2], q, axis=0) - 0.5
    upper = np.quantile(scaled_pts[:, :2], 1.0 - q, axis=0) + 0.5
    span_xy = upper - lower
    width, height = int(np.ceil(span_xy[0] / RES)) + 1, int(np.ceil(span_xy[1] / RES)) + 1
    print(f"  Map size: {width}x{height} cells @ {RES}m")

    in_bounds = (scaled_pts[:, 0] >= lower[0]) & (scaled_pts[:, 0] <= upper[0]) & (scaled_pts[:, 1] >= lower[1]) & (scaled_pts[:, 1] <= upper[1])
    pts_filt = scaled_pts[in_bounds]
    col = np.floor((pts_filt[:, 0] - lower[0]) / RES).astype(np.int64)
    row = np.floor((pts_filt[:, 1] - lower[1]) / RES).astype(np.int64)
    flat = row * width + col

    rel_z = pts_filt[:, 2] - ground_z
    ground_mask = np.abs(rel_z) <= 0.10
    obstacle_mask = (rel_z >= 0.12) & (rel_z <= 1.80)
    ground_counts = np.bincount(flat[ground_mask], minlength=width * height)
    obstacle_counts = np.bincount(flat[obstacle_mask], minlength=width * height)

    cells = np.full(width * height, -1, dtype=np.int8)
    cells[ground_counts >= 2] = 0
    cells[obstacle_counts >= 2] = 100
    grid = cells.reshape((height, width))

    # Save PGM/YAML
    pixels = np.full(grid.shape, 205, dtype=np.uint8)
    pixels[grid == 0] = 254
    pixels[grid == 100] = 0
    pixels = np.flipud(pixels)
    pgm_path = MAP_DIR / "map.pgm"
    with pgm_path.open("wb") as f:
        f.write(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
        f.write(pixels.tobytes(order="C"))
    yaml_text = f"image: map.pgm\nresolution: {RES}\norigin: [{lower[0]}, {lower[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n"
    (MAP_DIR / "map.yaml").write_text(yaml_text)
    cell_stats = {"free": int((grid == 0).sum()), "occupied": int((grid == 100).sum()), "unknown": int((grid == -1).sum())}
    print(f"  Free: {cell_stats['free']:,}  Occupied: {cell_stats['occupied']:,}  Unknown: {cell_stats['unknown']:,}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: SAM3.1 Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 5: SAM3.1 video tracking")
    print("=" * 60)

    from sam3.model_builder import build_sam3_predictor
    predictor = build_sam3_predictor(
        version="sam3.1", use_fa3=False, use_rope_real=True,
        compile=False, warm_up=False,
        checkpoint_path=CHECKPOINT_SAM3,
    )
    predictor.model.batched_grounding_batch_size = 1
    predictor.model.postprocess_batch_size = 1
    print("  SAM3.1 loaded.")

    import uuid, re
    def prompt_slug(idx, p):
        return f"{idx:03d}_{re.sub(r'[^a-zA-Z0-9_-]+', '_', p).strip('_')[:48]}"

    config = {"prompt_frame": 0, "probability_threshold": 0.5,
              "propagation_direction": "forward", "offload_video_to_cpu": True,
              "offload_state_to_cpu": False}

    for pi, prompt in enumerate(PROMPTS):
        slug = prompt_slug(pi, prompt)
        prompt_dir = SAM3_DIR / slug
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Init session
        import inspect
        params = inspect.signature(predictor.model.init_state).parameters
        init_kwargs = {"resource_path": str(VIDEO)}
        for k in ["offload_video_to_cpu", "offload_state_to_cpu"]:
            if k in params:
                init_kwargs[k] = config[k]
        state = predictor.model.init_state(**init_kwargs)
        sid = str(uuid.uuid4())
        predictor._all_inference_states[sid] = {"state": state, "session_id": sid, "start_time": time.time(), "last_use_time": time.time()}

        try:
            # Add prompt
            resp = predictor.handle_request({
                "type": "add_prompt", "session_id": sid,
                "frame_index": 0, "text": prompt,
                "output_prob_thresh": config["probability_threshold"],
            })
            fi = int(resp["frame_index"])
            objs = resp["outputs"].get("out_obj_ids", [])
            if len(objs) > 0:
                np.savez_compressed(prompt_dir / f"frame_{fi:06d}.npz",
                    object_ids=np.asarray(objs, dtype=np.int64),
                    track_ids=np.asarray([f"p{pi}:o{int(v)}" for v in objs]),
                    masks=np.asarray(resp["outputs"].get("out_binary_masks", []), dtype=np.uint8),
                    scores=np.asarray(resp["outputs"].get("out_probs", np.ones(len(objs))), dtype=np.float32),
                    boxes_xywh=np.asarray(resp["outputs"].get("out_boxes_xywh", np.zeros((len(objs), 4))), dtype=np.float32),
                )

            # Propagate
            stream = predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "forward", "start_frame_index": 0,
                "output_prob_thresh": config["probability_threshold"],
            })
            frame_objs = {}
            for item in stream:
                fi = int(item["frame_index"])
                objs = item["outputs"].get("out_obj_ids", [])
                frame_objs[fi] = len(objs)
                if len(objs) > 0:
                    np.savez_compressed(prompt_dir / f"frame_{fi:06d}.npz",
                        object_ids=np.asarray(objs, dtype=np.int64),
                        track_ids=np.asarray([f"p{pi}:o{int(v)}" for v in objs]),
                        masks=np.asarray(item["outputs"].get("out_binary_masks", []), dtype=np.uint8),
                        scores=np.asarray(item["outputs"].get("out_probs", np.ones(len(objs))), dtype=np.float32),
                        boxes_xywh=np.asarray(item["outputs"].get("out_boxes_xywh", np.zeros((len(objs), 4))), dtype=np.float32),
                    )
            print(f"  [{prompt}]: {len(frame_objs)} frames, {sum(frame_objs.values())} total detections")
        finally:
            predictor.handle_request({"type": "close_session", "session_id": sid})

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 6: Semantic Map (CLIPSeg)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 6: Building semantic map (CLIPSeg)")
    print("=" * 60)

    DEFAULT_LABELS = ("floor", "wall", "door", "table", "chair", "sofa", "bed", "toilet", "sink", "refrigerator", "television", "plant")

    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        segmenter_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
        segmenter_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        print("  CLIPSeg loaded.")

        labels = DEFAULT_LABELS
        frame_stride = 3
        pixel_stride = 3
        votes = np.zeros((len(labels), height, width), dtype=np.float32)

        pred_files = sorted(PREDS_DIR.glob("frame_*.npz"))[::frame_stride]
        frames_used = 0
        for path in pred_files:
            with np.load(path, allow_pickle=False) as data:
                if "depth" not in data or "intrinsic" not in data or "extrinsic" not in data:
                    continue
                d = np.asarray(data["depth"], dtype=np.float64).squeeze()
                K = np.asarray(data["intrinsic"], dtype=np.float64)
                T = np.asarray(data["extrinsic"], dtype=np.float64)
                img = np.asarray(data["images"])
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                if img.max() <= 1.0: img = (img * 255).clip(0, 255)
                img = img.astype(np.uint8)

            # Unproject
            h_d, w_d = d.shape
            rr_d, cc_d = np.indices((h_d, w_d), dtype=np.float64)
            x = (cc_d - K[0, 2]) * d / K[0, 0]
            y = (rr_d - K[1, 2]) * d / K[1, 1]
            cam = np.stack((x, y, d), axis=-1)
            w2c = np.eye(4); w2c[:3, :4] = T if T.shape == (3, 4) else T
            c2w = np.linalg.inv(w2c)
            pts_world = (cam.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]).reshape(h_d, w_d, 3) * scale

            # CLIPSeg
            try:
                pil_img = Image.fromarray(img)
                inputs = segmenter_proc(
                    text=[f"a photo of {l}" for l in labels],
                    images=[pil_img] * len(labels),
                    padding=True, return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    logits = segmenter_model(**inputs).logits[:, None]
                    logits = torch.nn.functional.interpolate(logits, size=(h_d, w_d), mode="bilinear", align_corners=False)[:, 0]
                scores = logits.sigmoid().float().cpu().numpy()
            except Exception:
                continue

            # Vote
            rr = np.arange(0, h_d, pixel_stride)
            cc = np.arange(0, w_d, pixel_stride)
            img_rows, img_cols = np.meshgrid(rr, cc, indexing="ij")
            sampled_pts = pts_world[img_rows, img_cols]
            sampled_scores = scores[:, img_rows, img_cols]
            cls = sampled_scores.argmax(axis=0)
            confidence = sampled_scores.max(axis=0)
            map_cols = np.floor((sampled_pts[..., 0] - lower[0]) / RES).astype(int)
            map_rows = np.floor((sampled_pts[..., 1] - lower[1]) / RES).astype(int)
            valid = (np.isfinite(sampled_pts).all(axis=-1) & (confidence >= 0.35)
                     & (map_rows >= 0) & (map_rows < height) & (map_cols >= 0) & (map_cols < width))
            for li in range(len(labels)):
                sel = valid & (cls == li)
                np.add.at(votes[li], (map_rows[sel], map_cols[sel]), confidence[sel])
            frames_used += 1

        best = votes.argmax(axis=0)
        strength = votes.max(axis=0)
        semantic_ids = np.where(strength >= 2.0, best + 1, 0).astype(np.uint16)
        semantic_ids[grid == -1] = 0

        # Region map
        from scipy.ndimage import label as scipy_label
        regions, region_count = scipy_label(grid == 0, structure=np.ones((3, 3), dtype=np.uint8))
        regions = regions.astype(np.int32)

        # Save
        np.save(MAP_DIR / "semantic_map.npy", semantic_ids)
        np.save(MAP_DIR / "region_map.npy", regions)
        # Color maps
        for name, ids, count in [("semantic_map", semantic_ids, len(labels)), ("region_map", regions, region_count)]:
            palette = np.zeros((max(count + 1, 2), 3), dtype=np.uint8)
            for i in range(1, len(palette)):
                palette[i] = ((53 * i) % 251, (97 * i) % 241, (193 * i) % 239)
            Image.fromarray(palette[np.clip(ids, 0, len(palette) - 1)].astype(np.uint8)).save(MAP_DIR / f"{name}.png")

        print(f"  Semantic labels: {labels}")
        print(f"  Frames used: {frames_used}")
        print(f"  Regions: {region_count}")
        print(f"  Saved: semantic_map.npy/.png, region_map.npy/.png")
    except ImportError as e:
        print(f"  SKIPPED: {e}")
        print(f"  Install: pip install transformers scipy")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 7: SAM3 → 3D Projection
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Step 7: SAM3 mask → 3D projection")
    print("=" * 60)

    total_obs = 0
    for pi, prompt in enumerate(PROMPTS):
        slug = prompt_slug(pi, prompt)
        prompt_dir = SAM3_DIR / slug
        if not prompt_dir.is_dir():
            continue
        observations = []
        for mask_path in sorted(prompt_dir.glob("frame_*.npz")):
            try:
                fi = int(mask_path.stem.split("_")[-1])
            except ValueError:
                continue
            pred_path = PREDS_DIR / f"frame_{fi:06d}.npz"
            if not pred_path.is_file():
                continue
            with np.load(mask_path, allow_pickle=False) as mdata, np.load(pred_path, allow_pickle=False) as gdata:
                if "world_points" not in gdata:
                    continue
                wpts = np.asarray(gdata["world_points"], dtype=np.float64)
                wconf = gdata.get("world_points_conf")
                for tid, mask, sc in zip(mdata["track_ids"], mdata["masks"], mdata["scores"]):
                    mask_bool = np.asarray(mask, dtype=bool).squeeze()
                    if mask_bool.shape != wpts.shape[:2]:
                        continue
                    valid_mask = mask_bool & np.isfinite(wpts).all(axis=-1)
                    if wconf is not None:
                        wc = np.asarray(wconf).squeeze()
                        valid_mask &= np.isfinite(wc) & (wc > 0.01)
                    selected = wpts[valid_mask] * scale
                    if selected.shape[0] < 30:
                        continue
                    centroid = np.median(selected, axis=0)
                    min_pt = np.quantile(selected, 0.02, axis=0)
                    max_pt = np.quantile(selected, 0.98, axis=0)
                    observations.append({
                        "track_id": str(tid), "prompt": prompt,
                        "frame_index": fi, "score": float(sc),
                        "point_count": int(selected.shape[0]),
                        "centroid_xyz": tuple(float(v) for v in centroid),
                        "min_xyz": tuple(float(v) for v in min_pt),
                        "max_xyz": tuple(float(v) for v in max_pt),
                    })
        if observations:
            (MAP_DIR / f"observations_{slug}.json").write_text(
                json.dumps({"observations": observations}, indent=2))
            print(f"  [{prompt}]: {len(observations)} 3D observations")
            total_obs += len(observations)
    print(f"  Total: {total_obs} 3D observations")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Pipeline complete! ({time.time() - t_total:.0f}s total)")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT}")
    print(f"\nArtifacts:")
    for f in sorted(OUTPUT.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                print(f"  {f.relative_to(OUTPUT)}  ({size/1024/1024:.1f}MB)")
            else:
                print(f"  {f.relative_to(OUTPUT)}  ({size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
