"""Fit LingBot's predicted camera trajectory to a metric RGB survey.

The simulator poses are never model inputs.  They are consumed only after
RGB-only inference to establish the metric map frame used by occupancy and
evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from .pointcloud import (
    _CV_TO_SURVEY_CAMERA,
    _camera_points_from_depth,
    _numpy,
    _quaternion_wxyz_to_rotation,
)


def fit_similarity(source, target):
    """Return ``scale, rotation, translation`` for target ~= s R source + t."""

    np = _numpy()
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ConfigurationError("Similarity correspondences must both have shape [N, 3]")
    if source.shape[0] < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ConfigurationError("Similarity fitting needs at least three finite correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.sum(source_centered * source_centered) / source.shape[0])
    if variance <= 1e-12:
        raise ConfigurationError("LingBot camera trajectory is degenerate")
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0.0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_t
    scale = float(np.sum(singular * signs) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ConfigurationError("Similarity fitting produced an invalid scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _predicted_camera_centres(prediction_dir: Path):
    np = _numpy()
    files = sorted(prediction_dir.glob("frame_*.npz"))
    if not files:
        raise ConfigurationError(f"No LingBot predictions found in {prediction_dir}")
    centres = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            key = "camera_to_world" if "camera_to_world" in data else "extrinsic"
            if key not in data:
                raise ConfigurationError(f"{path.name} has no predicted camera transform")
            transform = np.asarray(data[key], dtype=np.float64)
            if transform.shape == (3, 4):
                centres.append(transform[:, 3])
            elif transform.shape == (4, 4):
                centres.append(transform[:3, 3])
            else:
                raise ConfigurationError(
                    f"Unexpected camera transform shape in {path.name}: {transform.shape}"
                )
    return files, np.asarray(centres, dtype=np.float64)


def _survey_camera_centres(manifest_path: Path):
    np = _numpy()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = payload["frames"]
        centres = np.asarray(
            [frame["camera_pose"]["position"] for frame in frames], dtype=np.float64
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read RGB survey manifest {manifest_path}: {exc}") from exc
    if centres.ndim != 2 or centres.shape[1] != 3 or not np.isfinite(centres).all():
        raise ConfigurationError("Survey camera positions must have shape [N, 3]")
    return payload, centres


def align_lingbot_to_survey(
    predictions: str | Path,
    survey_manifest: str | Path,
    output_file: str | Path,
    *,
    inlier_threshold_m: float = 0.45,
    maximum_iterations: int = 6,
) -> dict[str, Any]:
    """Robustly align predicted camera centres to corresponding survey frames."""

    np = _numpy()
    prediction_dir = Path(predictions).expanduser().resolve()
    manifest_path = Path(survey_manifest).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()
    if not math.isfinite(inlier_threshold_m) or inlier_threshold_m <= 0.0:
        raise ConfigurationError("Alignment inlier threshold must be positive")
    files, source = _predicted_camera_centres(prediction_dir)
    manifest, target = _survey_camera_centres(manifest_path)
    if len(source) != len(target):
        raise ConfigurationError(
            f"LingBot/survey frame counts differ: {len(source)} != {len(target)}"
        )
    inliers = np.ones(len(source), dtype=bool)
    minimum_inliers = max(3, int(math.ceil(0.50 * len(source))))
    for _ in range(maximum_iterations):
        scale, rotation, translation = fit_similarity(source[inliers], target[inliers])
        predicted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target, axis=1)
        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        robust_limit = median + 2.5 * max(1.4826 * mad, 0.02)
        candidate = residuals <= min(inlier_threshold_m, robust_limit)
        if int(candidate.sum()) < minimum_inliers:
            raise ConfigurationError(
                "LingBot global trajectory cannot be aligned reliably: "
                f"only {int(candidate.sum())}/{len(source)} correspondences are within "
                f"{inlier_threshold_m:.3f} m"
            )
        if np.array_equal(candidate, inliers):
            break
        inliers = candidate
    scale, rotation, translation = fit_similarity(source[inliers], target[inliers])
    predicted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    result = {
        "schema_version": 1,
        "artifact_type": "lingbot_to_metric_survey_sim3",
        "matrix": matrix.tolist(),
        "scale_m_per_unit": scale,
        "correspondences": len(source),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "rmse_m": float(np.sqrt(np.mean(residuals[inliers] ** 2))),
        "max_inlier_error_m": float(residuals[inliers].max()),
        "inlier_threshold_m": inlier_threshold_m,
        "inlier_frame_indices": np.flatnonzero(inliers).astype(int).tolist(),
        "inputs": {
            "predictions": str(prediction_dir),
            "survey_manifest": str(manifest_path),
            "prediction_frames": [path.name for path in files],
        },
        "ground_truth_boundary": {
            "rgb_is_only_lingbot_input": bool(manifest.get("rgb_is_only_model_input")),
            "survey_pose_used_for_model_inference": False,
            "survey_pose_used_for_offline_metric_alignment": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_pose_anchored_alignment(
    predictions: str | Path,
    survey_manifest: str | Path,
    output_file: str | Path,
    *,
    floor_z_m: float = 0.001,
    confidence_quantile: float = 0.50,
) -> dict[str, Any]:
    """Estimate depth scale from the floor for offline survey-pose fusion."""

    np = _numpy()
    prediction_dir = Path(predictions).expanduser().resolve()
    manifest_path = Path(survey_manifest).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()
    files = sorted(prediction_dir.glob("frame_*.npz"))
    manifest, _ = _survey_camera_centres(manifest_path)
    frames = manifest["frames"]
    if len(files) != len(frames):
        raise ConfigurationError(
            f"LingBot/survey frame counts differ: {len(files)} != {len(frames)}"
        )
    if not 0.0 <= confidence_quantile < 1.0:
        raise ConfigurationError("Floor confidence quantile must be in [0, 1)")

    cv_to_survey = np.asarray(_CV_TO_SURVEY_CAMERA, dtype=np.float64)
    frame_scales = []
    frame_indices = []
    for frame_index, (path, frame) in enumerate(zip(files, frames)):
        with np.load(path, allow_pickle=False) as data:
            if not all(key in data for key in ("depth", "intrinsic")):
                raise ConfigurationError(
                    f"{path.name} needs depth and intrinsic for floor scale estimation"
                )
            camera_points = _camera_points_from_depth(data["depth"], data["intrinsic"])
            target_shape = camera_points.shape[:2]
            confidence = np.asarray(
                data.get("depth_conf", np.ones(target_shape)), dtype=np.float64
            ).squeeze()
        if confidence.shape != target_shape:
            raise ConfigurationError(f"Confidence shape mismatch in {path}: {confidence.shape}")
        camera_pose = frame.get("camera_pose", {})
        position = np.asarray(camera_pose.get("position"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ConfigurationError(f"Survey frame {frame_index} position is invalid")
        rotation = _quaternion_wxyz_to_rotation(
            camera_pose.get("orientation_wxyz")
        ) @ cv_to_survey
        vertical_offsets = camera_points @ rotation[2, :]
        lower_start = int(0.45 * target_shape[0])
        values = vertical_offsets[lower_start:]
        conf = confidence[lower_start:]
        finite_conf = conf[np.isfinite(conf)]
        if finite_conf.size == 0:
            continue
        threshold = np.quantile(finite_conf, confidence_quantile)
        valid = np.isfinite(values) & np.isfinite(conf) & (conf >= threshold) & (values < -0.05)
        values = values[valid]
        if values.size < 100:
            continue
        low, high = np.quantile(values, [0.02, 0.98])
        if not high > low:
            continue
        histogram, edges = np.histogram(values, bins=80, range=(low, high))
        peak = int(histogram.argmax())
        predicted_floor_distance = -0.5 * (edges[peak] + edges[peak + 1])
        metric_camera_height = float(position[2] - floor_z_m)
        if predicted_floor_distance > 0.0 and metric_camera_height > 0.0:
            frame_scales.append(metric_camera_height / predicted_floor_distance)
            frame_indices.append(frame_index)

    if len(frame_scales) < max(3, len(files) // 2):
        raise ConfigurationError(
            f"Too few frames expose a stable floor for scale estimation: {len(frame_scales)}"
        )
    values = np.asarray(frame_scales, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    limit = max(3.5 * 1.4826 * mad, 0.15 * median)
    inliers = np.abs(values - median) <= limit
    if int(inliers.sum()) < max(3, len(files) // 2):
        raise ConfigurationError("Floor-derived LingBot scales are not stable enough")
    scale = float(np.median(values[inliers]))
    inlier_values = values[inliers]
    result = {
        "schema_version": 1,
        "artifact_type": "lingbot_depth_to_metric_survey_pose_anchor",
        "scale_m_per_unit": scale,
        "scale_method": "known_camera_height_and_predicted_floor_mode",
        "floor_z_m": floor_z_m,
        "frames_considered": len(files),
        "floor_frames": len(values),
        "inliers": int(inliers.sum()),
        "scale_p10": float(np.quantile(inlier_values, 0.10)),
        "scale_p90": float(np.quantile(inlier_values, 0.90)),
        "inputs": {
            "predictions": str(prediction_dir),
            "survey_manifest": str(manifest_path),
            "inlier_frame_indices": np.asarray(frame_indices)[inliers].astype(int).tolist(),
        },
        "ground_truth_boundary": {
            "rgb_is_only_lingbot_input": bool(manifest.get("rgb_is_only_model_input")),
            "survey_pose_used_for_model_inference": False,
            "survey_pose_used_for_offline_per_frame_geometry_fusion": True,
            "known_camera_height_used_for_metric_scale": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = [
    "align_lingbot_to_survey",
    "build_pose_anchored_alignment",
    "fit_similarity",
]
