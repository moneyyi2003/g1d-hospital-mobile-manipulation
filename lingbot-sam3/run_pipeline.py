#!/usr/bin/env python3
"""LingBot-SAM3 Pipeline: Video → Point Cloud → Semantic + Occupancy + Region Maps.

Usage:
    # Full pipeline with SAM3 tracking
    python run_pipeline.py \\
        --video input.mp4 \\
        --fps 10 \\
        --lingbot-checkpoint checkpoints/lingbot-map.pt \\
        --sam3-checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \\
        --prompts prompts.txt \\
        --output outputs/my_scene

    # LingBot-only (point cloud + occupancy, no SAM3)
    python run_pipeline.py \\
        --video input.mp4 \\
        --lingbot-checkpoint checkpoints/lingbot-map.pt \\
        --output outputs/my_scene \\
        --skip-sam3

    # From existing RGB frames directory
    python run_pipeline.py \\
        --rgb-dir path/to/frames/ \\
        --lingbot-checkpoint checkpoints/lingbot-map.pt \\
        --output outputs/my_scene \\
        --skip-sam3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def _extract_frames(video_path: str | Path, output_dir: str | Path, fps: int = 10) -> Path:
    """Extract frames from a video file into an image directory."""
    import cv2

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / fps))

    saved = []
    idx = 0
    print(f"Extracting frames from {video_path} (fps={fps}, interval={interval})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = output_dir / f"{len(saved):06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
        idx += 1
    cap.release()
    print(f"  Extracted {len(saved)} frames (from {total_frames} total)")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="LingBot-SAM3: Video → Point Cloud + Semantic + Occupancy Maps"
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Input MP4 video file")
    input_group.add_argument("--rgb-dir", type=str, help="Directory of pre-extracted RGB frames")

    parser.add_argument("--fps", type=int, default=10, help="Frame extraction FPS (default: 10)")
    parser.add_argument("--first-k", type=int, default=None, help="Only use first K frames")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    # ── LingBot-Map ────────────────────────────────────────────────────────────
    parser.add_argument("--lingbot-checkpoint", type=str, required=True,
                        help="Path to LingBot-Map checkpoint (.pt)")
    parser.add_argument("--lingbot-mode", type=str, default="streaming",
                        choices=["streaming", "windowed"])
    parser.add_argument("--no-lingbot", action="store_true",
                        help="Skip LingBot inference (use existing predictions)")

    # ── SAM3 ───────────────────────────────────────────────────────────────────
    parser.add_argument("--sam3-checkpoint", type=str, default=None,
                        help="Path to SAM3.1 checkpoint (.pt)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Text file with one object prompt per line")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="Skip SAM3 tracking (point cloud + occupancy only)")
    parser.add_argument("--sam3-threshold", type=float, default=0.5,
                        help="SAM3 probability threshold (default: 0.5)")

    # ── Map building ───────────────────────────────────────────────────────────
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Occupancy map resolution in meters (default: 0.05)")
    parser.add_argument("--ground-z", type=float, default=0.0,
                        help="Ground plane Z in meters (default: 0.0)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Meters per LingBot unit (auto-estimated if not set)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip CLIPSeg semantic/region mapping")
    parser.add_argument("--semantic-labels", type=str, default=None,
                        help="Comma-separated custom labels for semantic mapping "
                             "(default: floor,wall,door,table,chair,sofa,bed,toilet,sink,"
                             "refrigerator,television,plant)")

    # ── Visualization ──────────────────────────────────────────────────────────
    parser.add_argument("--visualize", action="store_true",
                        help="Launch interactive 3D point cloud viewer")

    args = parser.parse_args()

    # ── Setup output directories ───────────────────────────────────────────────
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rgb_dir = output_root / "rgb_frames"
    lingbot_output = output_root / "lingbot"
    sam3_output = output_root / "sam3"
    map_output = output_root / "maps"
    alignment_file = output_root / "alignment.json"

    timing: dict[str, float] = {}

    # ── Step 0: Prepare RGB frames ─────────────────────────────────────────────
    t0 = time.time()
    if args.rgb_dir:
        rgb_dir = Path(args.rgb_dir).expanduser().resolve()
        print(f"Using existing RGB frames from: {rgb_dir}")
    else:
        _extract_frames(args.video, rgb_dir, fps=args.fps)

    # Apply stride / first_k to create the actual frame set
    import glob
    frame_paths = sorted(glob.glob(str(rgb_dir / "*.jpg")) + glob.glob(str(rgb_dir / "*.png")))
    if args.first_k:
        frame_paths = frame_paths[:args.first_k]
    if args.stride > 1:
        frame_paths = frame_paths[::args.stride]

    # Create a sub-directory with the filtered frames if needed
    if args.first_k or args.stride > 1:
        filtered_dir = output_root / "rgb_frames_filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for i, src in enumerate(frame_paths):
            ext = Path(src).suffix
            shutil.copy2(src, filtered_dir / f"{i:06d}{ext}")
        rgb_dir = filtered_dir

    timing["prepare_frames"] = time.time() - t0
    print(f"RGB frames ready: {len(frame_paths)} images in {rgb_dir}")

    # ── Step 1: LingBot-Map inference ──────────────────────────────────────────
    if not args.no_lingbot:
        print("\n" + "=" * 60)
        print("Step 1: LingBot-Map inference (RGB → depth + point cloud + poses)")
        print("=" * 60)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline.lingbot_backend import LingBotInferenceConfig, run_lingbot_map

        t0 = time.time()
        lingbot_config = LingBotInferenceConfig(mode=args.lingbot_mode)
        manifest = run_lingbot_map(
            rgb_directory=rgb_dir,
            checkpoint=args.lingbot_checkpoint,
            output_directory=lingbot_output,
            config=lingbot_config,
        )
        timing["lingbot_inference"] = time.time() - t0
        print(f"  Done in {timing['lingbot_inference']:.1f}s")
        print(f"  Predictions saved to: {lingbot_output / 'predictions'}")
    else:
        print("\nSkipping LingBot inference (--no-lingbot)")
        # Use existing predictions
        if not (lingbot_output / "predictions").is_dir():
            print(f"  WARNING: No existing predictions found at {lingbot_output / 'predictions'}")

    # ── Step 2: Auto-estimate scale + alignment ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Scale estimation and alignment")
    print("=" * 60)

    from pipeline.alignment import build_pose_anchored_alignment

    t0 = time.time()
    predictions_dir = lingbot_output / "predictions"
    if predictions_dir.is_dir():
        # Since we don't have a survey manifest in standalone mode,
        # we build a self-consistent alignment using LingBot's own camera trajectory
        from pipeline.pointcloud import _numpy
        np = _numpy()

        # Estimate scale from the point cloud extent
        frame_files = sorted(predictions_dir.glob("frame_*.npz"))
        if frame_files:
            all_points = []
            for f in frame_files[:min(20, len(frame_files))]:  # Use first 20 frames
                with np.load(f, allow_pickle=False) as data:
                    if "world_points" in data:
                        pts = np.asarray(data["world_points"]).reshape(-1, 3)
                        all_points.append(pts[np.isfinite(pts).all(axis=1)])

            if all_points:
                combined = np.concatenate(all_points, axis=0)
                extent = np.quantile(combined[:, :2], 0.95) - np.quantile(combined[:, :2], 0.05)
                scene_span = float(np.linalg.norm(extent))

                if args.scale:
                    scale_m_per_unit = args.scale
                    print(f"  Using user-specified scale: {scale_m_per_unit} m/unit")
                else:
                    # Heuristic: typical indoor scene is ~5-15m across
                    if scene_span > 50:
                        scale_m_per_unit = 0.01  # large arbitrary units → cm
                    elif scene_span < 0.5:
                        scale_m_per_unit = 10.0  # tiny units → likely dm
                    else:
                        scale_m_per_unit = 1.0  # already in meters
                    print(f"  Auto-estimated scale: {scale_m_per_unit:.4f} m/unit "
                          f"(scene span={scene_span:.1f} units)")

                # Build identity alignment matrix (world = LingBot coordinates * scale)
                alignment_matrix = np.eye(4, dtype=np.float64)
                alignment_payload = {
                    "schema_version": 1,
                    "artifact_type": "lingbot_to_metric_survey_sim3",
                    "matrix": alignment_matrix.tolist(),
                    "scale_m_per_unit": scale_m_per_unit,
                    "scale_method": "auto_estimated_from_point_cloud_extent",
                }
                alignment_file.parent.mkdir(parents=True, exist_ok=True)
                alignment_file.write_text(json.dumps(alignment_payload, indent=2) + "\n")
                print(f"  Alignment saved to: {alignment_file}")
            else:
                scale_m_per_unit = args.scale or 1.0
                print(f"  WARNING: No valid points found, using scale={scale_m_per_unit}")
        else:
            scale_m_per_unit = args.scale or 1.0
            print(f"  WARNING: No predictions found, using scale={scale_m_per_unit}")
    else:
        scale_m_per_unit = args.scale or 1.0
        print(f"  Using default scale: {scale_m_per_unit}")

    timing["alignment"] = time.time() - t0

    # ── Step 3: Build point cloud ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Building point cloud (PLY)")
    print("=" * 60)

    t0 = time.time()
    from pipeline.pointcloud import (
        PointCloudBuildConfig,
        load_alignment_matrix,
        load_lingbot_points,
        write_binary_ply,
    )

    pointcloud_written = False
    if predictions_dir.is_dir() and list(predictions_dir.glob("frame_*.npz")):
        try:
            matrix = load_alignment_matrix(alignment_file)
            pc_config = PointCloudBuildConfig(
                scale_m_per_unit=scale_m_per_unit,
                alignment_matrix=matrix,
            )
            points, colors, pc_stats = load_lingbot_points(predictions_dir, pc_config)
            ply_path = map_output / "pointcloud.ply"
            write_binary_ply(ply_path, points, colors)
            pointcloud_written = True
            print(f"  Point cloud: {pc_stats.kept_points:,} points → {ply_path}")
            print(f"  Stats: {pc_stats.frames_used}/{pc_stats.frames_seen} frames used, "
                  f"{pc_stats.raw_points:,} raw points")
        except Exception as exc:
            print(f"  WARNING: Point cloud build failed: {exc}")
    else:
        print("  No predictions available, skipping point cloud")

    timing["pointcloud"] = time.time() - t0

    # ── Step 4: Build occupancy map ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Building occupancy map (ROS PGM/YAML)")
    print("=" * 60)

    t0 = time.time()
    if pointcloud_written:
        from pipeline.occupancy import (
            OccupancyBuildConfig,
            build_occupancy,
            write_ros_map,
        )

        occ_config = OccupancyBuildConfig(
            resolution=args.resolution,
            ground_z=args.ground_z,
        )
        grid = build_occupancy(points, occ_config)
        ros_result = write_ros_map(map_output, grid)
        print(f"  Occupancy map: {ros_result['width']}x{ros_result['height']} cells")
        print(f"  Free: {ros_result['cell_counts']['free']:,}, "
              f"Occupied: {ros_result['cell_counts']['occupied']:,}, "
              f"Unknown: {ros_result['cell_counts']['unknown']:,}")
        print(f"  Saved to: {map_output}/map.pgm, {map_output}/map.yaml")
    else:
        print("  Skipping (no point cloud available)")

    timing["occupancy"] = time.time() - t0

    # ── Step 5: SAM3 tracking ──────────────────────────────────────────────────
    if not args.skip_sam3 and args.sam3_checkpoint and args.prompts:
        print("\n" + "=" * 60)
        print("Step 5: SAM3 video tracking (video → segmentation masks)")
        print("=" * 60)

        from pipeline.sam3_backend import Sam3TrackConfig, run_sam3_tracking

        with open(args.prompts, "r", encoding="utf-8") as f:
            prompt_list = [line.strip() for line in f if line.strip()]
        print(f"  Prompts: {prompt_list}")

        t0 = time.time()
        sam3_config = Sam3TrackConfig(probability_threshold=args.sam3_threshold)

        # Use the video file directly if available, else the preprocessed RGB dir
        video_resource = args.video if args.video else str(rgb_dir)
        try:
            sam3_manifest = run_sam3_tracking(
                video_resource=video_resource,
                prompts=prompt_list,
                output_directory=sam3_output,
                checkpoint=args.sam3_checkpoint,
                config=sam3_config,
            )
            timing["sam3_tracking"] = time.time() - t0
            print(f"  Done in {timing['sam3_tracking']:.1f}s")
            for record in sam3_manifest["prompts"]:
                print(f"  '{record['prompt']}': {record['frames']} frames, "
                      f"{record['detections']} detections")
        except Exception as exc:
            print(f"  WARNING: SAM3 tracking failed: {exc}")
            print(f"  Continuing without SAM3 results...")

    elif not args.skip_sam3:
        print("\nSkipping SAM3 (--sam3-checkpoint and --prompts required)")

    timing["sam3"] = time.time() - t0 if "sam3_tracking" not in timing else 0

    # ── Step 6: SAM3 mask → 3D projection ──────────────────────────────────────
    if not args.skip_sam3 and sam3_output.is_dir():
        print("\n" + "=" * 60)
        print("Step 6: Projecting SAM3 masks into 3D map frame")
        print("=" * 60)

        from pipeline.mask_projection import build_track_observations

        t0 = time.time()
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompt_list = [line.strip() for line in f if line.strip()]

        total_obs = 0
        for prompt in prompt_list:
            try:
                obs = build_track_observations(
                    lingbot_predictions=predictions_dir,
                    sam3_artifacts=sam3_output,
                    alignment_file=alignment_file,
                    output_file=map_output / f"observations_{prompt}.json",
                    prompt=prompt,
                    scale_m_per_unit=scale_m_per_unit,
                )
                total_obs += len(obs)
                print(f"  '{prompt}': {len(obs)} 3D observations")
            except Exception as exc:
                print(f"  WARNING: Projection for '{prompt}' failed: {exc}")

        timing["mask_projection"] = time.time() - t0
        print(f"  Total: {total_obs} 3D observations")

    # ── Step 7: Semantic + Region maps ─────────────────────────────────────────
    if not args.no_semantic and pointcloud_written:
        print("\n" + "=" * 60)
        print("Step 7: Building semantic + region maps (CLIPSeg)")
        print("=" * 60)

        t0 = time.time()
        try:
            from pipeline.semantic_map import (
                ClipSegSegmenter,
                SemanticMapConfig,
                build_semantic_maps,
            )

            if args.semantic_labels:
                labels = tuple(l.strip() for l in args.semantic_labels.split(","))
            else:
                from pipeline.semantic_map import DEFAULT_LABELS
                labels = DEFAULT_LABELS
            print(f"  Labels: {labels}")

            segmenter = ClipSegSegmenter()
            sem_config = SemanticMapConfig(labels=labels)
            sem_metadata = build_semantic_maps(
                prediction_dir=predictions_dir,
                output_dir=map_output,
                grid=grid,
                scale_m_per_unit=scale_m_per_unit,
                alignment_matrix=load_alignment_matrix(alignment_file),
                segmenter=segmenter,
                config=sem_config,
            )
            print(f"  Semantic map: {sem_metadata['shape'][0]}x{sem_metadata['shape'][1]} cells")
            print(f"  Regions: {sem_metadata['region_count']}")
            print(f"  Place candidates: {sem_metadata['place_candidates_generated']}")
            print(f"  Frames used: {sem_metadata['frames_used']}")
            print(f"  Saved to: {map_output}/semantic_map.npy, "
                  f"{map_output}/region_map.npy")
        except ImportError as exc:
            print(f"  WARNING: Semantic mapping requires additional dependencies: {exc}")
            print(f"  Install with: pip install transformers torch scipy pillow")
        except Exception as exc:
            print(f"  WARNING: Semantic mapping failed: {exc}")

        timing["semantic"] = time.time() - t0

    # ── Step 8: Visualization ──────────────────────────────────────────────────
    if args.visualize and pointcloud_written:
        print("\n" + "=" * 60)
        print("Step 8: Launching 3D viewer")
        print("=" * 60)
        try:
            from lingbot_map.vis import PointCloudViewer

            viewer = PointCloudViewer(
                pred_dict={"world_points": points, "images": colors},
                port=8080,
            )
            print(f"  3D viewer at http://localhost:8080")
            viewer.run()
        except ImportError:
            print("  viser not installed. Install with: pip install viser")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_root}")
    print(f"\nGenerated artifacts:")
    if (lingbot_output / "predictions").is_dir():
        print(f"  LingBot predictions:  {lingbot_output / 'predictions'}/")
    if (lingbot_output / "preprocessed_rgb").is_dir():
        print(f"  Preprocessed RGB:     {lingbot_output / 'preprocessed_rgb'}/")
    if map_output.is_dir():
        for f in sorted(map_output.iterdir()):
            print(f"  Map artifact:         {f}")
    if sam3_output.is_dir():
        print(f"  SAM3 tracks:          {sam3_output}/")

    print(f"\nTiming summary:")
    for step, elapsed in timing.items():
        print(f"  {step:25s} {elapsed:6.1f}s")
    total = sum(timing.values())
    print(f"  {'total':25s} {total:6.1f}s")


if __name__ == "__main__":
    main()
