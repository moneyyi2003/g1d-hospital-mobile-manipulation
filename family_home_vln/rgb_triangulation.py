"""Metric object localization from reviewed SAM3 masks and survey camera poses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _camera_ray(
    robot_pose: dict[str, Any],
    *,
    pixel_uv: tuple[float, float],
    image_size: tuple[int, int],
    focal_xy: tuple[float, float],
    camera_height_m: float,
    camera_forward_offset_m: float,
    downward_pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    u, v = pixel_uv
    fx, fy = focal_xy
    yaw = float(robot_pose["yaw"])
    origin = np.asarray(
        [
            float(robot_pose["x"]) + camera_forward_offset_m * math.cos(yaw),
            float(robot_pose["y"]) + camera_forward_offset_m * math.sin(yaw),
            camera_height_m,
        ],
        dtype=np.float64,
    )
    # Isaac camera convention used by the survey: local +X forward, +Z up.
    local = np.asarray(
        [1.0, -(u - width / 2.0) / fx, -(v - height / 2.0) / fy],
        dtype=np.float64,
    )
    local /= np.linalg.norm(local)
    pitch = math.radians(downward_pitch_deg)
    rotation_y = np.asarray(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    direction = rotation_z @ rotation_y @ local
    direction /= np.linalg.norm(direction)
    return origin, direction


def triangulate_rays(
    origins: list[np.ndarray],
    directions: list[np.ndarray],
) -> tuple[np.ndarray, list[float], float]:
    """Return the least-squares point nearest all 3-D rays."""

    if len(origins) != len(directions) or len(origins) < 3:
        raise ValueError("triangulation needs at least three paired rays")
    identity = np.eye(3, dtype=np.float64)
    projectors = [
        identity - np.outer(direction, direction)
        for direction in directions
    ]
    system = sum(projectors)
    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > 1.0e5:
        raise ValueError(
            f"triangulation ray geometry is ill-conditioned: {condition:.1f}"
        )
    rhs = sum(
        projector @ origin
        for projector, origin in zip(projectors, origins)
    )
    point = np.linalg.solve(system, rhs)
    residuals = [
        float(np.linalg.norm(np.cross(point - origin, direction)))
        for origin, direction in zip(origins, directions)
    ]
    return point, residuals, condition


def triangulate_reviewed_sam3_track(
    survey_manifest_path: Path,
    artifact_directory: Path,
    *,
    start_frame: int,
    end_frame_exclusive: int,
    downward_pitch_deg: float = 25.0,
    camera_forward_offset_m: float = 0.18,
    minimum_camera_baseline_m: float = 0.08,
    maximum_median_ray_error_m: float = 0.05,
) -> dict[str, Any]:
    """Triangulate one visually reviewed, temporally bounded SAM3 track."""

    if start_frame < 0 or end_frame_exclusive <= start_frame:
        raise ValueError("invalid reviewed triangulation frame window")
    manifest = json.loads(survey_manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    camera = manifest.get("camera", {})
    source_width, source_height = (
        int(value) for value in camera["resolution"]
    )
    intrinsics = camera["intrinsics"]
    source_fx = float(intrinsics[0][0])
    source_fy = float(intrinsics[1][1])
    camera_height = float(camera["height_above_floor_m"])

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    used_frames: list[dict[str, Any]] = []
    mask_size: tuple[int, int] | None = None
    for frame_index in range(start_frame, end_frame_exclusive):
        path = artifact_directory / f"frame_{frame_index:06d}.npz"
        if not path.is_file() or frame_index >= len(frames):
            continue
        values = np.load(path)
        masks = np.asarray(values["masks"])
        scores = np.asarray(values["scores"])
        if masks.shape[0] == 0:
            continue
        mask_index = int(
            max(
                range(masks.shape[0]),
                key=lambda index: int(np.count_nonzero(masks[index])),
            )
        )
        mask = masks[mask_index].astype(bool)
        rows, columns = np.nonzero(mask)
        if columns.size < 30:
            continue
        height, width = mask.shape
        mask_size = (width, height)
        focal = (
            source_fx * width / source_width,
            source_fy * height / source_height,
        )
        pixel = (float(np.median(columns)), float(np.median(rows)))
        origin, direction = _camera_ray(
            frames[frame_index]["robot_pose"],
            pixel_uv=pixel,
            image_size=mask_size,
            focal_xy=focal,
            camera_height_m=camera_height,
            camera_forward_offset_m=camera_forward_offset_m,
            downward_pitch_deg=downward_pitch_deg,
        )
        origins.append(origin)
        directions.append(direction)
        used_frames.append(
            {
                "frame_index": frame_index,
                "pixel_uv": [pixel[0], pixel[1]],
                "mask_pixels": int(columns.size),
                "score": float(scores[mask_index]),
                "camera_origin_map_m": origin.tolist(),
                "ray_direction_map": direction.tolist(),
            }
        )
    if len(origins) < 3:
        raise ValueError(
            f"reviewed SAM3 window yielded only {len(origins)} usable masks"
        )
    baseline = max(
        float(np.linalg.norm(left - right))
        for left in origins
        for right in origins
    )
    if baseline < minimum_camera_baseline_m:
        raise ValueError(
            f"camera baseline {baseline:.3f} m is below "
            f"{minimum_camera_baseline_m:.3f} m"
        )
    point, residuals, condition = triangulate_rays(origins, directions)
    median_error = float(np.median(residuals))
    if median_error > maximum_median_ray_error_m:
        raise ValueError(
            f"median triangulation ray error {median_error:.3f} m exceeds "
            f"{maximum_median_ray_error_m:.3f} m"
        )
    return {
        "schema_version": 1,
        "method": "reviewed_sam3_mask_centroid_multiview_ray_triangulation",
        "frame_id": "map",
        "point_xyz_m": point.tolist(),
        "reviewed_frame_window": [start_frame, end_frame_exclusive],
        "used_frame_count": len(used_frames),
        "camera_baseline_m": baseline,
        "median_ray_error_m": median_error,
        "maximum_ray_error_m": max(residuals),
        "normal_system_condition": condition,
        "camera_calibration": {
            "source_resolution": [source_width, source_height],
            "mask_resolution": list(mask_size or (0, 0)),
            "downward_pitch_deg": downward_pitch_deg,
            "height_above_floor_m": camera_height,
            "forward_offset_m": camera_forward_offset_m,
        },
        "frames": used_frames,
        "ground_truth_boundary": {
            "rgb_and_sam3_masks_used": True,
            "survey_camera_pose_used_after_model_inference": True,
            "usd_semantics_read": False,
            "scene_object_coordinates_read": False,
        },
    }


__all__ = [
    "triangulate_rays",
    "triangulate_reviewed_sam3_track",
]
