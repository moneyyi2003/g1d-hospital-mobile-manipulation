"""LingBot-SAM3 pipeline: video → point cloud → semantic + occupancy maps."""

from .pointcloud import (
    load_lingbot_points,
    load_pose_anchored_lingbot_points,
    write_binary_ply,
    PointCloudBuildConfig,
    PoseAnchoredPointCloudBuildConfig,
    load_alignment_matrix,
)
from .occupancy import (
    build_occupancy,
    clear_traversed_footprints,
    write_ros_map,
    OccupancyBuildConfig,
    OccupancyGrid,
)
from .semantic_map import (
    build_semantic_maps,
    SemanticMapConfig,
)
from .mask_projection import (
    project_mask_to_map,
    project_pose_anchored_mask_to_map,
    build_track_observations,
    TrackObservation3D,
)
from .alignment import (
    align_lingbot_to_survey,
    build_pose_anchored_alignment,
    fit_similarity,
)

__all__ = [
    "load_lingbot_points",
    "load_pose_anchored_lingbot_points",
    "write_binary_ply",
    "PointCloudBuildConfig",
    "PoseAnchoredPointCloudBuildConfig",
    "load_alignment_matrix",
    "build_occupancy",
    "clear_traversed_footprints",
    "write_ros_map",
    "OccupancyBuildConfig",
    "OccupancyGrid",
    "build_semantic_maps",
    "SemanticMapConfig",
    "project_mask_to_map",
    "project_pose_anchored_mask_to_map",
    "build_track_observations",
    "TrackObservation3D",
    "align_lingbot_to_survey",
    "build_pose_anchored_alignment",
    "fit_similarity",
]
