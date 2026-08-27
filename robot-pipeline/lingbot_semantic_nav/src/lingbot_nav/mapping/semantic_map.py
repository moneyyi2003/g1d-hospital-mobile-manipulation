"""Open-vocabulary 2D-to-3D fusion into semantic and region BEV maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from ..errors import ConfigurationError
from .occupancy import OccupancyGrid
from .pointcloud import _normalize_colors, _numpy, _unproject_depth


DEFAULT_LABELS = (
    "floor", "wall", "door", "table", "chair", "sofa", "bed",
    "toilet", "sink", "refrigerator", "television", "plant",
)

LABEL_ALIASES = {
    "door": ("门", "出入口", "door"),
    "table": ("桌子", "餐桌", "table"),
    "chair": ("椅子", "座位", "chair"),
    "sofa": ("沙发", "休息区", "lounge", "sofa"),
    "bed": ("床", "卧室", "bedroom", "bed"),
    "toilet": ("卫生间", "厕所", "洗手间", "toilet", "restroom"),
    "sink": ("洗手池", "水槽", "sink"),
    "refrigerator": ("冰箱", "厨房", "refrigerator", "kitchen"),
    "television": ("电视", "电视区", "television", "tv"),
    "plant": ("绿植", "植物", "plant"),
}


def load_ros_occupancy(map_yaml: str | Path) -> OccupancyGrid:
    """Load a ROS trinary PGM into the internal bottom-left-origin grid."""
    np = _numpy()
    from ..map_validation import load_map_metadata, load_pgm

    metadata = load_map_metadata(map_yaml)
    width, height, raw = load_pgm(metadata.image)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
    occupancy = pixels / 255.0 if metadata.negate else (255 - pixels) / 255.0
    cells = np.full((height, width), -1, dtype=np.int8)
    cells[occupancy < metadata.free_thresh] = 0
    cells[occupancy > metadata.occupied_thresh] = 100
    cells = np.flipud(cells)
    from .occupancy import OccupancyBuildConfig
    return OccupancyGrid(
        cells, metadata.resolution, metadata.origin_x, metadata.origin_y,
        OccupancyBuildConfig(resolution=metadata.resolution),
    )


@dataclass(frozen=True)
class SemanticMapConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    frame_stride: int = 3
    pixel_stride: int = 3
    score_threshold: float = 0.35
    min_votes: float = 2.0
    robot_radius: float = 0.22
    max_goal_distance: float = 1.25
    export_unverified_places: bool = False

    def validate(self) -> None:
        if not self.labels or any(not str(label).strip() for label in self.labels):
            raise ConfigurationError("Semantic labels must not be empty")
        if self.frame_stride < 1 or self.pixel_stride < 1:
            raise ConfigurationError("Semantic frame/pixel strides must be positive")
        if not math.isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ConfigurationError("score_threshold must be in [0, 1]")
        if not math.isfinite(self.min_votes) or self.min_votes <= 0.0:
            raise ConfigurationError("min_votes must be positive")
        if not math.isfinite(self.robot_radius) or self.robot_radius <= 0.0:
            raise ConfigurationError("robot_radius must be positive")
        if not math.isfinite(self.max_goal_distance) or self.max_goal_distance <= 0.0:
            raise ConfigurationError("max_goal_distance must be positive")


class ClipSegSegmenter:
    """Text-conditioned pixel segmenter; labels are supplied at runtime."""

    def __init__(self, model_id: str = "CIDAS/clipseg-rd64-refined", device: str = "cuda"):
        try:
            import torch
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        except ImportError as exc:
            raise ConfigurationError("CLIPSeg needs torch and transformers") from exc
        self.torch = torch
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, image, labels: Sequence[str]):
        torch = self.torch
        from PIL import Image
        import torch.nn.functional as functional

        rgb = Image.fromarray(image)
        inputs = self.processor(
            text=[f"a photo of {label}" for label in labels],
            images=[rgb] * len(labels),
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[:, None]
            logits = functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False
            )[:, 0]
        return logits.sigmoid().float().cpu().numpy()


def _aligned_points(data, scale: float, alignment):
    np = _numpy()
    points = _unproject_depth(data["depth"], data["intrinsic"], data["extrinsic"])
    flat = points.reshape(-1, 3) * scale
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    return (np.asarray(alignment) @ homogeneous.T).T[:, :3].reshape(points.shape)


def _write_color_map(path: Path, ids, count: int) -> None:
    np = _numpy()
    from PIL import Image

    palette = np.zeros((max(count + 1, 2), 3), dtype=np.uint8)
    for idx in range(1, len(palette)):
        palette[idx] = ((53 * idx) % 251, (97 * idx) % 241, (193 * idx) % 239)
    Image.fromarray(palette[np.clip(ids, 0, len(palette) - 1)]).save(path)


def _regions_from_free(cells):
    np = _numpy()
    try:
        from scipy.ndimage import label
    except ImportError as exc:
        raise ConfigurationError("Region extraction needs scipy") from exc
    regions, count = label(cells == 0, structure=np.ones((3, 3), dtype=np.uint8))
    return regions.astype(np.int32), int(count)


def _places_from_semantics(semantic_ids, votes, grid: OccupancyGrid, labels, config):
    np = _numpy()
    from scipy.ndimage import distance_transform_edt

    free = grid.cells == 0
    clearance = distance_transform_edt(free) * grid.resolution
    safe_rows, safe_cols = np.where(free & (clearance >= config.robot_radius))
    places = []
    for class_index, name in enumerate(labels, start=1):
        aliases = LABEL_ALIASES.get(name)
        if not aliases:
            continue
        rows, cols = np.where(semantic_ids == class_index)
        if len(rows) < 3 or not len(safe_rows):
            continue
        target_row, target_col = float(np.median(rows)), float(np.median(cols))
        distance = np.hypot(safe_rows - target_row, safe_cols - target_col) * grid.resolution
        valid = distance <= config.max_goal_distance
        if not valid.any():
            continue
        candidates = np.flatnonzero(valid)
        best = candidates[np.argmin(distance[candidates] - 0.1 * clearance[safe_rows[candidates], safe_cols[candidates]])]
        row, col = int(safe_rows[best]), int(safe_cols[best])
        x = grid.origin_x + (col + 0.5) * grid.resolution
        y = grid.origin_y + (row + 0.5) * grid.resolution
        places.append({
            "id": name,
            "name": aliases[0],
            "aliases": list(aliases),
            "entrance_pose": {"x": x, "y": y, "yaw": math.atan2(target_row - row, target_col - col)},
            "region": f"region_{0}",
            "metadata": {
                "source": "open_vocabulary_semantic_map",
                "semantic_label": name,
                "support_cells": int(len(rows)),
                "mean_vote": float(votes[class_index - 1, rows, cols].mean()),
            },
        })
    return places


def build_semantic_maps(
    prediction_dir: str | Path,
    output_dir: str | Path,
    grid: OccupancyGrid,
    *,
    scale_m_per_unit: float,
    alignment_matrix,
    segmenter: Callable,
    config: SemanticMapConfig = SemanticMapConfig(),
) -> dict:
    np = _numpy()
    config.validate()
    if not math.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0.0:
        raise ConfigurationError("scale_m_per_unit must be positive")
    alignment = np.asarray(alignment_matrix, dtype=np.float64)
    if alignment.shape != (4, 4) or not np.isfinite(alignment).all():
        raise ConfigurationError("alignment_matrix must be a finite 4x4 matrix")
    root, output = Path(prediction_dir), Path(output_dir)
    files = sorted(root.glob("frame_*.npz"))[:: config.frame_stride]
    if not files:
        raise ConfigurationError(f"No LingBot prediction frames under {root}")
    height, width = grid.cells.shape
    votes = np.zeros((len(config.labels), height, width), dtype=np.float32)
    observations = np.zeros((height, width), dtype=np.uint32)
    frames_used = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = ("depth", "intrinsic", "extrinsic", "images")
            if not all(key in data for key in required):
                raise ConfigurationError(f"{path.name} lacks compact LingBot RGB-D fields")
            depth_shape = np.asarray(data["depth"]).squeeze().shape
            image = _normalize_colors(data["images"], depth_shape)
            scores = np.asarray(segmenter(image, config.labels), dtype=np.float32)
            if scores.shape != (len(config.labels), *depth_shape):
                raise ConfigurationError(f"Segmenter returned unexpected shape {scores.shape}")
            points = _aligned_points(data, scale_m_per_unit, alignment)
        rr = np.arange(0, depth_shape[0], config.pixel_stride)
        cc = np.arange(0, depth_shape[1], config.pixel_stride)
        image_rows, image_cols = np.meshgrid(rr, cc, indexing="ij")
        pts = points[image_rows, image_cols]
        sampled = scores[:, image_rows, image_cols]
        cls = sampled.argmax(axis=0)
        confidence = sampled.max(axis=0)
        cols = np.floor((pts[..., 0] - grid.origin_x) / grid.resolution).astype(int)
        rows = np.floor((pts[..., 1] - grid.origin_y) / grid.resolution).astype(int)
        valid = (
            np.isfinite(pts).all(axis=-1) & (confidence >= config.score_threshold)
            & (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        )
        np.add.at(observations, (rows[valid], cols[valid]), 1)
        for index in range(len(config.labels)):
            selected = valid & (cls == index)
            np.add.at(votes[index], (rows[selected], cols[selected]), confidence[selected])
        frames_used += 1

    best = votes.argmax(axis=0)
    strength = votes.max(axis=0)
    semantic_ids = np.where(strength >= config.min_votes, best + 1, 0).astype(np.uint16)
    semantic_ids[grid.cells == -1] = 0
    regions, region_count = _regions_from_free(grid.cells)
    places = _places_from_semantics(semantic_ids, votes, grid, config.labels, config)
    for place in places:
        pose = place["entrance_pose"]
        row = int((pose["y"] - grid.origin_y) / grid.resolution)
        col = int((pose["x"] - grid.origin_x) / grid.resolution)
        place["region"] = f"region_{int(regions[row, col])}"

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "semantic_map.npy", semantic_ids)
    np.save(output / "semantic_votes.npy", votes)
    np.save(output / "region_map.npy", regions)
    _write_color_map(output / "semantic_map.png", semantic_ids, len(config.labels))
    _write_color_map(output / "region_map.png", regions, region_count)
    legacy_candidates = {
        "schema_version": 1,
        "artifact_type": "legacy_unverified_class_candidates",
        "frame_id": "map",
        "warning": (
            "CLIPSeg class heatmaps do not identify object instances or verify semantics; "
            "these candidates must not be used for online navigation without independent review"
        ),
        "candidates": places,
    }
    (output / "legacy_place_candidates.json").write_text(
        json.dumps(legacy_candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if config.export_unverified_places:
        places_payload = {"schema_version": 1, "frame_id": "map", "places": places}
        (output / "places.json").write_text(
            json.dumps(places_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    metadata = {
        "schema_version": 1,
        "backend": "open_vocabulary",
        "labels": {str(i + 1): label for i, label in enumerate(config.labels)},
        "origin": [grid.origin_x, grid.origin_y],
        "resolution": grid.resolution,
        "shape": [height, width],
        "frames_used": frames_used,
        "region_count": region_count,
        "place_candidates_generated": len(places),
        "unverified_places_exported": bool(config.export_unverified_places),
        "observed_cells": int((observations > 0).sum()),
    }
    (output / "semantic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
