"""Build a metric ROS occupancy map from official LingBot prediction artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .occupancy import (
    OccupancyBuildConfig,
    build_occupancy,
    clear_traversed_footprints,
    write_ros_map,
)
from .pointcloud import (
    PointCloudBuildConfig,
    PoseAnchoredPointCloudBuildConfig,
    load_alignment_matrix,
    load_lingbot_points,
    load_pose_anchored_lingbot_points,
    write_binary_ply,
)


def build_metric_occupancy_map(
    predictions: str | Path,
    alignment_file: str | Path,
    output_directory: str | Path,
    *,
    scale_m_per_unit: float,
    resolution_m: float = 0.05,
    ground_z_m: float = 0.0,
    robot_obstacle_min_height_m: float = 0.12,
    robot_obstacle_max_height_m: float = 1.80,
    survey_manifest: str | Path | None = None,
    traversed_footprint_clearance_m: float = 0.40,
) -> dict[str, Any]:
    output = Path(output_directory).expanduser().resolve()
    alignment_path = Path(alignment_file).expanduser().resolve()
    alignment_payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    pose_anchored = (
        alignment_payload.get("artifact_type")
        == "lingbot_depth_to_metric_survey_pose_anchor"
    )
    if pose_anchored:
        manifest_path = survey_manifest or alignment_payload.get("inputs", {}).get(
            "survey_manifest"
        )
        if not manifest_path:
            raise ValueError("Pose-anchored map building requires a survey manifest")
        point_config = PoseAnchoredPointCloudBuildConfig(
            scale_m_per_unit=scale_m_per_unit,
        )
        points, colors, point_stats = load_pose_anchored_lingbot_points(
            predictions, manifest_path, point_config
        )
    else:
        matrix = load_alignment_matrix(alignment_path)
        point_config = PointCloudBuildConfig(
            scale_m_per_unit=scale_m_per_unit,
            alignment_matrix=matrix,
        )
        points, colors, point_stats = load_lingbot_points(predictions, point_config)
    output.mkdir(parents=True, exist_ok=True)
    pointcloud = output / "lingbot_map_metric.ply"
    write_binary_ply(pointcloud, points, colors)
    occupancy_config = OccupancyBuildConfig(
        resolution=resolution_m,
        ground_z=ground_z_m,
        obstacle_min_height=robot_obstacle_min_height_m,
        obstacle_max_height=robot_obstacle_max_height_m,
    )
    grid = build_occupancy(points, occupancy_config)
    footprint_clearance_m = None
    if pose_anchored:
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        robot_positions = [
            [frame["robot_pose"]["x"], frame["robot_pose"]["y"]]
            for frame in manifest_payload["frames"]
        ]
        footprint_clearance_m = traversed_footprint_clearance_m
        grid = clear_traversed_footprints(
            grid, robot_positions, radius_m=footprint_clearance_m
        )
    map_artifacts = write_ros_map(output, grid)
    manifest = {
        "schema_version": 1,
        "pipeline": "lingbot_map_metric_ros_occupancy",
        "inputs": {
            "predictions": str(Path(predictions).expanduser().resolve()),
            "alignment": str(alignment_path),
            "scale_m_per_unit": scale_m_per_unit,
        },
        "pointcloud": {"path": str(pointcloud), **point_stats.to_dict()},
        "occupancy_config": asdict(occupancy_config),
        "map": map_artifacts,
        "ground_truth_inputs": {
            "habitat_depth": False,
            "habitat_pose": pose_anchored,
            "habitat_semantics": False,
            "habitat_navmesh": False,
        },
        "geometry_fusion": (
            "offline_survey_pose_anchored" if pose_anchored else "global_sim3"
        ),
        "traversed_footprint_clearance_m": footprint_clearance_m,
    }
    (output / "metric_map_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["build_metric_occupancy_map"]
