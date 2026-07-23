"""Convert saved LingBot-Map per-frame predictions into an aligned RGB point cloud."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "Mapping commands require NumPy. Install with: pip install -e '.[mapping]'"
        ) from exc
    return np


@dataclass(frozen=True)
class PointCloudBuildConfig:
    scale_m_per_unit: float
    alignment_matrix: Any
    confidence_quantile: float = 0.50
    frame_stride: int = 1
    max_points_per_frame: int = 50_000
    max_total_points: int = 3_000_000

    def validate(self) -> None:
        np = _numpy()
        matrix = np.asarray(self.alignment_matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ConfigurationError("Alignment matrix must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
            raise ConfigurationError("Alignment matrix last row must be [0, 0, 0, 1]")
        if not math.isfinite(self.scale_m_per_unit) or self.scale_m_per_unit <= 0:
            raise ConfigurationError("scale_m_per_unit must be explicitly set and positive")
        if not 0.0 <= self.confidence_quantile < 1.0:
            raise ConfigurationError("confidence_quantile must be in [0, 1)")
        if min(self.frame_stride, self.max_points_per_frame, self.max_total_points) < 1:
            raise ConfigurationError("Point-cloud sampling options must be positive")


@dataclass(frozen=True)
class PointCloudStats:
    frames_seen: int
    frames_used: int
    raw_points: int
    kept_points: int
    scale_m_per_unit: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PoseAnchoredPointCloudBuildConfig:
    scale_m_per_unit: float
    confidence_quantile: float = 0.50
    frame_stride: int = 1
    max_points_per_frame: int = 50_000
    max_total_points: int = 3_000_000

    def validate(self) -> None:
        if not math.isfinite(self.scale_m_per_unit) or self.scale_m_per_unit <= 0:
            raise ConfigurationError("scale_m_per_unit must be explicitly set and positive")
        if not 0.0 <= self.confidence_quantile < 1.0:
            raise ConfigurationError("confidence_quantile must be in [0, 1)")
        if min(self.frame_stride, self.max_points_per_frame, self.max_total_points) < 1:
            raise ConfigurationError("Point-cloud sampling options must be positive")


def load_alignment_matrix(path: str | Path):
    np = _numpy()
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        matrix = payload["matrix"] if isinstance(payload, dict) else payload
        result = np.asarray(matrix, dtype=np.float64)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read alignment matrix {source}: {exc}") from exc
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ConfigurationError("Alignment matrix must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7):
        raise ConfigurationError("Alignment matrix last row must be [0, 0, 0, 1]")
    return result


def _normalize_colors(images, target_shape):
    np = _numpy()
    colors = np.asarray(images)
    if colors.ndim == 3 and colors.shape[0] == 3:
        colors = colors.transpose(1, 2, 0)
    if colors.shape != (*target_shape, 3):
        raise ConfigurationError(
            f"LingBot images shape {colors.shape} does not match points {target_shape}"
        )
    if np.issubdtype(colors.dtype, np.floating):
        if float(np.nanmax(colors)) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255)
    return colors.astype(np.uint8, copy=False)


def _unproject_depth(depth, intrinsic, extrinsic=None, camera_to_world=None):
    """Reconstruct world points from LingBot's compact RGB-D prediction."""
    np = _numpy()
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if camera_to_world is not None:
        c2w = np.asarray(camera_to_world, dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w_h = np.eye(4, dtype=np.float64)
            c2w_h[:3, :4] = c2w
            c2w = c2w_h
    else:
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        if extrinsic.shape == (3, 4):
            w2c[:3, :4] = extrinsic
            c2w = np.linalg.inv(w2c)
        else:
            c2w = np.empty((0, 0))
    if depth.ndim != 2 or intrinsic.shape != (3, 3) or c2w.shape != (4, 4):
        raise ConfigurationError(
            "LingBot depth/intrinsic/camera transform shapes are invalid"
        )
    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float64)
    x = (cols - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    camera = np.stack((x, y, depth), axis=-1)
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def _camera_points_from_depth(depth, intrinsic):
    np = _numpy()
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if depth.ndim != 2 or intrinsic.shape != (3, 3):
        raise ConfigurationError("LingBot depth/intrinsic shapes are invalid")
    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float64)
    x = (cols - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    return np.stack((x, y, depth), axis=-1)


def _quaternion_wxyz_to_rotation(quaternion):
    np = _numpy()
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ConfigurationError("Survey camera quaternion must be finite wxyz")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ConfigurationError("Survey camera quaternion has zero length")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# OpenCV camera coordinates are +X right, +Y down, +Z forward.  The survey
# manifest stores camera orientation for +X forward, +Y left, +Z up.
_CV_TO_SURVEY_CAMERA = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def load_pose_anchored_lingbot_points(
    prediction_dir: str | Path,
    survey_manifest: str | Path,
    config: PoseAnchoredPointCloudBuildConfig,
):
    """Fuse RGB-only LingBot depth using offline survey camera poses."""

    np = _numpy()
    config.validate()
    root = Path(prediction_dir)
    frame_files = sorted(root.glob("frame_*.npz"))
    if not frame_files:
        raise ConfigurationError(f"No frame_*.npz files found in {root}")
    try:
        manifest = json.loads(Path(survey_manifest).read_text(encoding="utf-8"))
        frames = manifest["frames"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigurationError(f"Cannot read survey manifest {survey_manifest}: {exc}") from exc
    if len(frames) != len(frame_files):
        raise ConfigurationError(
            f"LingBot/survey frame counts differ: {len(frame_files)} != {len(frames)}"
        )

    cv_to_survey = np.asarray(_CV_TO_SURVEY_CAMERA, dtype=np.float64)
    point_chunks = []
    color_chunks = []
    raw_points = 0
    used = 0
    for frame_index, (path, frame) in enumerate(zip(frame_files, frames)):
        if frame_index % config.frame_stride:
            continue
        with np.load(path, allow_pickle=False) as data:
            if not all(key in data for key in ("images", "depth", "intrinsic")):
                raise ConfigurationError(
                    f"{path.name} needs images, depth and intrinsic for pose-anchored fusion"
                )
            camera_points = _camera_points_from_depth(data["depth"], data["intrinsic"])
            target_shape = camera_points.shape[:2]
            colors_map = _normalize_colors(data["images"], target_shape)
            confidence = np.asarray(data.get("depth_conf", np.ones(target_shape))).squeeze()
            if confidence.shape != target_shape:
                raise ConfigurationError(f"Confidence shape mismatch in {path}: {confidence.shape}")

            try:
                camera_pose = frame["camera_pose"]
                position = np.asarray(camera_pose["position"], dtype=np.float64)
                survey_rotation = _quaternion_wxyz_to_rotation(
                    camera_pose["orientation_wxyz"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Survey frame {frame_index} has an invalid camera pose: {exc}"
                ) from exc
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ConfigurationError(f"Survey frame {frame_index} position is invalid")
            camera_to_hospital = survey_rotation @ cv_to_survey

            raw_points += int(camera_points.shape[0] * camera_points.shape[1])
            points = camera_points.reshape(-1, 3)
            colors = colors_map.reshape(-1, 3)
            conf = confidence.reshape(-1)
            valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
            if valid.any() and config.confidence_quantile > 0:
                threshold = np.quantile(conf[valid], config.confidence_quantile)
                valid &= conf >= threshold
            indices = np.flatnonzero(valid)
            if indices.size > config.max_points_per_frame:
                step = math.ceil(indices.size / config.max_points_per_frame)
                indices = indices[::step][: config.max_points_per_frame]
            if indices.size == 0:
                continue

            selected = points[indices] * config.scale_m_per_unit
            anchored = selected @ camera_to_hospital.T + position
            point_chunks.append(anchored.astype(np.float32))
            color_chunks.append(colors[indices])
            used += 1

    if not point_chunks:
        raise ConfigurationError("No valid LingBot points survived confidence filtering")
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if points.shape[0] > config.max_total_points:
        step = math.ceil(points.shape[0] / config.max_total_points)
        points = points[::step][: config.max_total_points]
        colors = colors[::step][: config.max_total_points]
    stats = PointCloudStats(
        frames_seen=len(frame_files),
        frames_used=used,
        raw_points=raw_points,
        kept_points=int(points.shape[0]),
        scale_m_per_unit=config.scale_m_per_unit,
    )
    return points, colors, stats


def load_lingbot_points(prediction_dir: str | Path, config: PointCloudBuildConfig):
    """Return (points_xyz_m, colors_rgb_u8, stats) from a saved NPZ directory."""
    np = _numpy()
    config.validate()
    root = Path(prediction_dir)
    frame_files = sorted(root.glob("frame_*.npz"))
    if not frame_files:
        raise ConfigurationError(f"No frame_*.npz files found in {root}")

    point_chunks = []
    color_chunks = []
    raw_points = 0
    used = 0
    matrix = np.asarray(config.alignment_matrix, dtype=np.float64)
    for frame_index, path in enumerate(frame_files):
        if frame_index % config.frame_stride:
            continue
        with np.load(path, allow_pickle=False) as data:
            if "images" not in data:
                raise ConfigurationError(
                    f"{path.name} needs images; rerun LingBot with --save_predictions"
                )
            if "world_points" in data:
                points_map = np.asarray(data["world_points"])
            elif (
                all(key in data for key in ("depth", "intrinsic"))
                and ("camera_to_world" in data or "extrinsic" in data)
            ):
                points_map = _unproject_depth(
                    data["depth"],
                    data["intrinsic"],
                    data["extrinsic"] if "extrinsic" in data else None,
                    data["camera_to_world"] if "camera_to_world" in data else None,
                )
            else:
                raise ConfigurationError(
                    f"{path.name} needs world_points or depth/intrinsic/camera transform"
                )
            if points_map.ndim != 3 or points_map.shape[-1] != 3:
                raise ConfigurationError(f"Unexpected world_points shape in {path}: {points_map.shape}")
            target_shape = points_map.shape[:2]
            colors_map = _normalize_colors(data["images"], target_shape)
            confidence = None
            for key in ("world_points_conf", "depth_conf", "confidence"):
                if key in data:
                    confidence = np.asarray(data[key]).squeeze()
                    break
            if confidence is None:
                confidence = np.ones(target_shape, dtype=np.float32)
            if confidence.shape != target_shape:
                raise ConfigurationError(f"Confidence shape mismatch in {path}: {confidence.shape}")

            raw_points += int(points_map.shape[0] * points_map.shape[1])
            points = points_map.reshape(-1, 3).astype(np.float64, copy=False)
            colors = colors_map.reshape(-1, 3)
            conf = confidence.reshape(-1)
            valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
            if valid.any() and config.confidence_quantile > 0:
                threshold = np.quantile(conf[valid], config.confidence_quantile)
                valid &= conf >= threshold
            indices = np.flatnonzero(valid)
            if indices.size > config.max_points_per_frame:
                step = math.ceil(indices.size / config.max_points_per_frame)
                indices = indices[::step][: config.max_points_per_frame]
            if indices.size == 0:
                continue

            selected = points[indices] * config.scale_m_per_unit
            homogeneous = np.concatenate(
                [selected, np.ones((selected.shape[0], 1), dtype=np.float64)], axis=1
            )
            aligned = (matrix @ homogeneous.T).T[:, :3]
            point_chunks.append(aligned.astype(np.float32))
            color_chunks.append(colors[indices])
            used += 1

    if not point_chunks:
        raise ConfigurationError("No valid LingBot points survived confidence filtering")
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if points.shape[0] > config.max_total_points:
        step = math.ceil(points.shape[0] / config.max_total_points)
        points = points[::step][: config.max_total_points]
        colors = colors[::step][: config.max_total_points]
    stats = PointCloudStats(
        frames_seen=len(frame_files),
        frames_used=used,
        raw_points=raw_points,
        kept_points=int(points.shape[0]),
        scale_m_per_unit=config.scale_m_per_unit,
    )
    return points, colors, stats


def write_binary_ply(path: str | Path, points, colors) -> None:
    np = _numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ConfigurationError("PLY points and colors must both have shape [N, 3]")
    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with target.open("wb") as stream:
        stream.write(header)
        vertex.tofile(stream)
