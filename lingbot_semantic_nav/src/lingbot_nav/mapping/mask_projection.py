"""Project SAM3 masks through LingBot-Map geometry into the ROS map frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from .pointcloud import _numpy, load_alignment_matrix


@dataclass(frozen=True)
class TrackObservation3D:
    track_id: str
    prompt: str
    frame_index: int
    score: float
    point_count: int
    centroid_xyz: tuple[float, float, float]
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_mask_to_map(
    mask: Any,
    world_points: Any,
    alignment_matrix: Any,
    *,
    scale_m_per_unit: float,
    confidence: Any | None = None,
    confidence_quantile: float = 0.25,
):
    np = _numpy()
    mask_array = np.asarray(mask, dtype=bool).squeeze()
    points = np.asarray(world_points, dtype=np.float64)
    if mask_array.ndim != 2 or points.shape != (*mask_array.shape, 3):
        raise ConfigurationError(
            f"SAM3 mask {mask_array.shape} and LingBot points {points.shape} are not pixel-aligned"
        )
    if not np.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0:
        raise ConfigurationError("LingBot-to-map scale must be explicitly positive")
    valid = mask_array & np.isfinite(points).all(axis=-1)
    if confidence is not None:
        confidence_array = np.asarray(confidence).squeeze()
        if confidence_array.shape != mask_array.shape:
            raise ConfigurationError("LingBot point confidence is not pixel-aligned")
        finite = np.isfinite(confidence_array)
        valid &= finite
        if valid.any():
            threshold = np.quantile(confidence_array[valid], confidence_quantile)
            valid &= confidence_array >= threshold
    selected = points[valid] * scale_m_per_unit
    if selected.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    matrix = np.asarray(alignment_matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ConfigurationError("LingBot-to-map alignment must be a finite 4x4 matrix")
    homogeneous = np.concatenate(
        (selected, np.ones((selected.shape[0], 1), dtype=np.float64)), axis=1
    )
    return (matrix @ homogeneous.T).T[:, :3].astype(np.float32)


def build_track_observations(
    lingbot_predictions: str | Path,
    sam3_artifacts: str | Path,
    alignment_file: str | Path,
    output_file: str | Path,
    *,
    prompt: str,
    scale_m_per_unit: float,
    minimum_points: int = 30,
) -> list[TrackObservation3D]:
    np = _numpy()
    prediction_root = Path(lingbot_predictions)
    sam_root = Path(sam3_artifacts)
    alignment = load_alignment_matrix(alignment_file)
    observations: list[TrackObservation3D] = []
    for mask_path in sorted(sam_root.glob("frame_*.npz")):
        try:
            frame_index = int(mask_path.stem.split("_")[-1])
        except ValueError as exc:
            raise ConfigurationError(f"Invalid SAM3 frame artifact name: {mask_path.name}") from exc
        prediction_path = prediction_root / f"frame_{frame_index:06d}.npz"
        if not prediction_path.is_file():
            raise ConfigurationError(f"Missing LingBot geometry for SAM3 frame {frame_index}")
        with np.load(mask_path, allow_pickle=False) as masks, np.load(
            prediction_path, allow_pickle=False
        ) as geometry:
            if "world_points" not in geometry:
                raise ConfigurationError(f"{prediction_path.name} lacks world_points")
            confidence = geometry.get("world_points_conf")
            for track_id, mask, score in zip(
                masks["track_ids"], masks["masks"], masks["scores"]
            ):
                projected = project_mask_to_map(
                    mask,
                    geometry["world_points"],
                    alignment,
                    scale_m_per_unit=scale_m_per_unit,
                    confidence=confidence,
                )
                if projected.shape[0] < minimum_points:
                    continue
                centroid = np.median(projected, axis=0)
                minimum = np.quantile(projected, 0.02, axis=0)
                maximum = np.quantile(projected, 0.98, axis=0)
                observations.append(TrackObservation3D(
                    str(track_id),
                    prompt,
                    frame_index,
                    float(score),
                    int(projected.shape[0]),
                    tuple(float(item) for item in centroid),
                    tuple(float(item) for item in minimum),
                    tuple(float(item) for item in maximum),
                ))
    payload = {
        "schema_version": 1,
        "frame_id": "map",
        "geometry_source": "lingbot_map_rgb_only",
        "mask_source": "sam3_text_video_tracking",
        "observations": [item.to_dict() for item in observations],
    }
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return observations


__all__ = ["TrackObservation3D", "build_track_observations", "project_mask_to_map"]
