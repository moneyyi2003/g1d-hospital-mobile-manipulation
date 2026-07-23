"""Build a local occupancy map from RGB-only LingBot-Map predictions.

This module intentionally has no Habitat imports and accepts no reference poses,
navmesh, depth sensor files, or ground-truth alignment matrices.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from .occupancy import OccupancyBuildConfig, build_occupancy, write_ros_map
from .pointcloud import (
    PointCloudBuildConfig,
    _numpy,
    load_lingbot_points,
    write_binary_ply,
)


RGB_ONLY_AXIS_MATRIX = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_sequence_digest(root: Path) -> tuple[str, int]:
    files = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if len(files) < 2:
        raise ConfigurationError(f"RGB-only input needs at least two images: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest(), len(files)


def estimate_ground_level(points: Any) -> float:
    """Estimate the dominant low horizontal surface in LingBot local units."""
    np = _numpy()
    z = np.asarray(points, dtype=np.float64)[:, 2]
    z = z[np.isfinite(z)]
    if z.size < 1000:
        raise ConfigurationError("Ground estimation needs at least 1000 finite points")
    low, high = np.quantile(z, (0.002, 0.45))
    if not float(high) > float(low):
        raise ConfigurationError("Cannot estimate a ground plane from degenerate points")
    counts, edges = np.histogram(z[(z >= low) & (z <= high)], bins=256, range=(low, high))
    index = int(counts.argmax())
    return float((edges[index] + edges[index + 1]) * 0.5)


def _predicted_camera_centres(prediction_files: list[Path], alignment: Any):
    """Camera trajectory in the LingBot-local map frame (prediction data only)."""
    np = _numpy()
    centres = []
    for path in prediction_files:
        with np.load(path, allow_pickle=False) as data:
            if "camera_to_world" in data:
                camera_to_world = np.eye(4, dtype=np.float64)
                camera_to_world[:3, :4] = np.asarray(
                    data["camera_to_world"], dtype=np.float64
                )
            elif "extrinsic" in data:
                # Legacy artifacts stored world-to-camera under this ambiguous
                # key. New official-adapter artifacts use camera_to_world.
                world_to_camera = np.eye(4, dtype=np.float64)
                world_to_camera[:3, :4] = np.asarray(
                    data["extrinsic"], dtype=np.float64
                )
                camera_to_world = np.linalg.inv(world_to_camera)
            else:
                raise ConfigurationError(
                    f"{path.name} lacks LingBot predicted camera transform"
                )
        centre = np.asarray(alignment, dtype=np.float64) @ camera_to_world[:, 3]
        centres.append(centre[:3])
    return np.asarray(centres, dtype=np.float64)


def _mark_predicted_trajectory_free(
    grid,
    centres: Any,
    radius: float = 0.24,
    obstacle_clear_radius: float = 0.10,
    maximum_segment_length: float = 0.60,
) -> dict[str, Any]:
    """Sweep a free corridor along LingBot-predicted camera motion.

    A reconstructed obstacle at the predicted optical centre contradicts the
    same model's motion estimate: the camera could not have occupied that cell.
    Unknown cells are cleared across the sensor footprint, while occupied cells
    are corrected only inside a narrow camera/body core.  Long pose jumps are
    not interpolated, avoiding artificial tunnels across relocalization gaps.
    Habitat poses, depth, navmesh, and semantics are neither accepted nor used.
    """
    np = _numpy()
    if not (
        radius > 0
        and 0 < obstacle_clear_radius <= radius
        and maximum_segment_length > 0
    ):
        raise ConfigurationError("Invalid LingBot trajectory corridor settings")
    radius_cells = max(1, int(math.ceil(radius / grid.resolution)))
    changed_unknown = 0
    changed_occupied = 0
    skipped_discontinuities = 0
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] < 2 or not np.isfinite(centres).all():
        raise ConfigurationError("LingBot camera centres must be finite [N, 3] values")
    samples = []
    for index, current in enumerate(centres):
        if index == 0:
            samples.append(current)
            continue
        previous = centres[index - 1]
        distance = float(np.linalg.norm(current[:2] - previous[:2]))
        if distance > maximum_segment_length:
            samples.append(current)
            skipped_discontinuities += 1
            continue
        steps = max(1, int(math.ceil(distance / max(grid.resolution, radius * 0.25))))
        samples.extend(
            previous + (current - previous) * (step / steps)
            for step in range(1, steps + 1)
        )
    for x, y, _ in samples:
        col = int(math.floor((x - grid.origin_x) / grid.resolution))
        row = int(math.floor((y - grid.origin_y) / grid.resolution))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr, cc = row + dr, col + dc
                if (
                    0 <= rr < grid.cells.shape[0]
                    and 0 <= cc < grid.cells.shape[1]
                ):
                    distance = math.hypot(dr, dc) * grid.resolution
                    value = int(grid.cells[rr, cc])
                    if value == -1 and distance <= radius:
                        grid.cells[rr, cc] = 0
                        changed_unknown += 1
                    elif value == 100 and distance <= obstacle_clear_radius:
                        grid.cells[rr, cc] = 0
                        changed_occupied += 1
    return {
        "source": "lingbot_predicted_extrinsics_only",
        "unknown_cells_cleared": changed_unknown,
        "occupied_cells_corrected": changed_occupied,
        "free_cells_added": changed_unknown + changed_occupied,
        "sensor_footprint_radius": radius,
        "occupied_core_radius": obstacle_clear_radius,
        "maximum_interpolated_segment": maximum_segment_length,
        "skipped_pose_discontinuities": skipped_discontinuities,
        "habitat_ground_truth_used": False,
    }


def build_rgb_only_map(
    predictions: str | Path,
    rgb_source: str | Path,
    output_dir: str | Path,
    *,
    resolution: float = 0.03,
    confidence_quantile: float = 0.50,
    frame_stride: int = 1,
) -> dict[str, Any]:
    np = _numpy()
    prediction_root = Path(predictions).resolve()
    rgb_root = Path(rgb_source).resolve()
    output = Path(output_dir).resolve()
    if not prediction_root.is_dir() or not rgb_root.is_dir():
        raise ConfigurationError("RGB-only predictions and RGB source must be directories")
    rgb_digest, rgb_frames = _rgb_sequence_digest(rgb_root)
    prediction_files = sorted(prediction_root.glob("frame_*.npz"))
    if len(prediction_files) != rgb_frames:
        raise ConfigurationError(
            f"Prediction/RGB frame mismatch: {len(prediction_files)} != {rgb_frames}"
        )

    base_alignment = np.asarray(RGB_ONLY_AXIS_MATRIX, dtype=np.float64)
    point_config = PointCloudBuildConfig(
        scale_m_per_unit=1.0,
        alignment_matrix=base_alignment,
        confidence_quantile=confidence_quantile,
        frame_stride=frame_stride,
    )
    points, colors, stats = load_lingbot_points(prediction_root, point_config)
    ground_level = estimate_ground_level(points)
    points[:, 2] -= ground_level
    local_alignment = base_alignment.copy()
    local_alignment[2, 3] = -ground_level

    output.mkdir(parents=True, exist_ok=True)
    pointcloud_path = output / "lingbot_local.ply"
    write_binary_ply(pointcloud_path, points, colors)
    occupancy_config = OccupancyBuildConfig(
        resolution=resolution,
        ground_z=0.0,
        ground_band=0.06,
        obstacle_min_height=0.08,
        obstacle_max_height=1.20,
        padding=0.30,
        minimum_points_per_cell=2,
        bounds_quantile=0.002,
    )
    grid = build_occupancy(points, occupancy_config)
    camera_centres = _predicted_camera_centres(prediction_files, local_alignment)
    trajectory_correction = _mark_predicted_trajectory_free(grid, camera_centres)
    map_artifacts = write_ros_map(output, grid)
    alignment_path = output / "lingbot_local_frame.json"
    alignment_payload = {
        "matrix": local_alignment.tolist(),
        "scale_lingbot_units_per_input_unit": 1.0,
        "unit": "lingbot_local_unit_not_meters",
        "ground_level_before_translation": ground_level,
        "source": "fixed_lingbot_axis_convention_plus_pointcloud_ground_histogram",
        "forbidden_inputs": [
            "Habitat poses",
            "Habitat depth sensor",
            "Habitat navmesh",
            "Habitat semantic annotations",
        ],
    }
    alignment_path.write_text(
        json.dumps(alignment_payload, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "pipeline": "lingbot_map_rgb_only_blind",
        "unit": "lingbot_local_unit_not_meters",
        "inputs": {
            "rgb_directory": str(rgb_root),
            "rgb_frames": rgb_frames,
            "rgb_sequence_sha256": rgb_digest,
            "predictions": str(prediction_root),
            "prediction_frames": len(prediction_files),
        },
        "prohibited_ground_truth_inputs": {
            "habitat_poses": False,
            "habitat_depth": False,
            "habitat_navmesh": False,
            "habitat_semantics": False,
        },
        "self_calibration": alignment_payload,
        "pointcloud_stats": stats.to_dict(),
        "predicted_camera_trajectory": {
            "frames": int(len(camera_centres)),
            "start": camera_centres[0].tolist(),
            "end": camera_centres[-1].tolist(),
            "free_cells_added": trajectory_correction["free_cells_added"],
            "occupancy_self_consistency": trajectory_correction,
            "source": "lingbot_predicted_extrinsics_only",
        },
        "occupancy_config": asdict(occupancy_config),
        "artifacts": {
            "pointcloud": str(pointcloud_path),
            "pointcloud_sha256": _sha256(pointcloud_path),
            "map_pgm": map_artifacts["pgm"],
            "map_yaml": map_artifacts["yaml"],
            "local_frame": str(alignment_path),
        },
        "map": {
            "width": map_artifacts["width"],
            "height": map_artifacts["height"],
            "resolution_lingbot_units": resolution,
            "origin": map_artifacts["origin"],
            "cell_counts": map_artifacts["cell_counts"],
        },
    }
    manifest_path = output / "rgb_only_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["RGB_ONLY_AXIS_MATRIX", "build_rgb_only_map", "estimate_ground_level"]
