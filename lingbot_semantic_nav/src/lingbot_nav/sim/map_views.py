"""Browser previews for aligned RGB point-cloud and 2D mapping artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..map_validation import load_map_metadata, load_pgm


LAYER_LABELS = {
    "rgb_pointcloud": "RGB 点云",
    "semantic": "Semantic",
    "instance": "Instance",
    "region": "Region",
    "occupancy": "Occupancy",
    "habitat_gt": "Habitat GT",
}


def _imports():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Map previews require NumPy and Pillow") from exc
    return np, Image


def _palette(values, np):
    maximum = int(values.max()) if values.size else 0
    palette = np.zeros((max(maximum + 1, 2), 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    for index in range(1, len(palette)):
        palette[index] = (
            45 + (53 * index) % 190,
            55 + (97 * index) % 180,
            65 + (193 * index) % 170,
        )
    return palette[np.clip(values, 0, len(palette) - 1)]


def _write_overlay(values, target: Path, occupancy, alpha: float, np, Image) -> None:
    base = np.asarray(occupancy.convert("RGB"), dtype=np.float32)
    colored = _palette(values, np).astype(np.float32)
    mask = values > 0
    base[mask] = base[mask] * (1.0 - alpha) + colored[mask] * alpha
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(target)


def _reproject_habitat_occupancy(reference, reference_meta, target_meta, np):
    """Sample Habitat topdown navigability into the LingBot map grid."""
    width = int(target_meta["width"])
    height = int(target_meta["height"])
    bounds = target_meta["bounds"]
    source_bounds = reference_meta["bounds"]
    xs = float(bounds["min_x"]) + (
        np.arange(width, dtype=np.float64) + 0.5
    ) * (float(bounds["max_x"]) - float(bounds["min_x"])) / width
    zs = float(bounds["min_z"]) + (
        np.arange(height, dtype=np.float64) + 0.5
    ) * (float(bounds["max_z"]) - float(bounds["min_z"])) / height
    cols = np.floor(
        (xs - float(source_bounds["min_x"]))
        / (float(source_bounds["max_x"]) - float(source_bounds["min_x"]))
        * reference.width
    ).astype(int)
    rows = np.floor(
        (zs - float(source_bounds["min_z"]))
        / (float(source_bounds["max_z"]) - float(source_bounds["min_z"]))
        * reference.height
    ).astype(int)
    valid_cols = (cols >= 0) & (cols < reference.width)
    valid_rows = (rows >= 0) & (rows < reference.height)
    source = np.asarray(reference.convert("L"))
    result = np.zeros((height, width), dtype=np.uint8)
    valid = valid_rows[:, None] & valid_cols[None, :]
    sampled = source[
        np.clip(rows, 0, reference.height - 1)[:, None],
        np.clip(cols, 0, reference.width - 1)[None, :],
    ]
    result[valid] = sampled[valid]
    return result


def _boundary(mask, np):
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def _add_gt_boundary(path: Path, gt_values, np, Image) -> None:
    image = np.asarray(Image.open(path).convert("RGB")).copy()
    image[_boundary(gt_values > 100, np)] = (255, 62, 170)
    Image.fromarray(image, mode="RGB").save(path)


def _write_rgb_pointcloud(
    source: Path,
    target: Path,
    *,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    flip_output: bool,
    np,
    Image,
) -> None:
    """Rasterize the upper visible RGB samples from this project's binary PLY."""
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    with source.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 16_384:
                raise ValueError(f"Invalid PLY header: {source}")
            header.extend(line)
        header_text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in header_text:
            raise ValueError(f"Only binary little-endian RGB PLY is supported: {source}")
        vertex_line = next(
            (line for line in header_text.splitlines() if line.startswith("element vertex ")),
            None,
        )
        if vertex_line is None:
            raise ValueError(f"PLY has no vertex count: {source}")
        count = int(vertex_line.rsplit(" ", 1)[1])
        points = np.fromfile(stream, dtype=dtype, count=count)
    if len(points) != count:
        raise ValueError(f"PLY vertex data is truncated: {source}")

    cols = np.floor((points["x"] - origin_x) / resolution).astype(np.int64)
    rows = np.floor((points["y"] - origin_y) / resolution).astype(np.int64)
    valid = (
        np.isfinite(points["x"])
        & np.isfinite(points["y"])
        & np.isfinite(points["z"])
        & (cols >= 0)
        & (cols < width)
        & (rows >= 0)
        & (rows < height)
        & (points["z"] >= -0.25)
        & (points["z"] <= 2.5)
    )
    cols, rows, cloud = cols[valid], rows[valid], points[valid]
    flat = rows * width + cols
    max_height = np.full(width * height, -np.inf, dtype=np.float32)
    np.maximum.at(max_height, flat, cloud["z"])
    top = cloud["z"] >= max_height[flat] - 0.08
    flat, cloud = flat[top], cloud[top]
    samples = np.bincount(flat, minlength=width * height)
    image = np.full((height * width, 3), (7, 16, 22), dtype=np.uint8)
    occupied = samples > 0
    for channel, name in enumerate(("red", "green", "blue")):
        totals = np.bincount(
            flat, weights=cloud[name].astype(np.float64), minlength=width * height
        )
        image[occupied, channel] = np.clip(
            totals[occupied] / samples[occupied], 0, 255
        ).astype(np.uint8)
    image = image.reshape((height, width, 3))
    if flip_output:
        image = np.flipud(image)
    Image.fromarray(image, mode="RGB").save(target)


def render_mapping_views(
    map_yaml: Path,
    output: Path,
    *,
    pointcloud_path: Path | None,
    semantic_map_path: Path | None = None,
    instance_map_path: Path | None = None,
    region_map_path: Path | None = None,
    reference_meta: dict[str, Any] | None = None,
    reference_occupancy_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    np, Image = _imports()
    metadata = load_map_metadata(map_yaml)
    width, height, pixels = load_pgm(metadata.image)
    source_shape = (height, width)
    output.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}

    target_meta = {
        "asset": "/asset/map/occupancy.png",
        "width": width,
        "height": height,
        "resolution": metadata.resolution,
        "flip_y": True,
        "bounds": {
            "min_x": metadata.origin_x,
            "max_x": metadata.origin_x + width * metadata.resolution,
            "min_z": metadata.origin_y,
            "max_z": metadata.origin_y + height * metadata.resolution,
        },
    }
    occupancy_path = output / "occupancy.png"
    occupancy = Image.frombytes("L", (width, height), pixels).convert("RGB")
    occupancy.save(occupancy_path)
    assets["occupancy"] = occupancy_path
    gt_display = None
    if reference_meta is not None and reference_occupancy_path is not None:
        reference = Image.open(reference_occupancy_path).convert("L")
        gt_logical = _reproject_habitat_occupancy(
            reference, reference_meta, target_meta, np
        )
        gt_display = np.flipud(gt_logical)
        gt_path = output / "habitat_gt.png"
        Image.fromarray(gt_display, mode="L").convert("RGB").save(gt_path)
        assets["habitat_gt"] = gt_path
    for layer_id, source in (
        ("semantic", semantic_map_path),
        ("instance", instance_map_path),
        ("region", region_map_path),
    ):
        if source is not None:
            values = np.load(source, allow_pickle=False)
            if values.shape != source_shape:
                raise ValueError(
                    f"{source.name} shape {values.shape} does not match occupancy "
                    f"shape {source_shape}"
                )
            values = np.flipud(values)
            target = output / f"{layer_id}.png"
            _write_overlay(
                values,
                target,
                occupancy,
                0.82 if layer_id in {"semantic", "instance"} else 0.58,
                np,
                Image,
            )
            if gt_display is not None:
                _add_gt_boundary(target, gt_display, np, Image)
            assets[layer_id] = target
    if pointcloud_path is not None:
        target = output / "rgb_pointcloud.png"
        target_bounds = target_meta["bounds"]
        _write_rgb_pointcloud(
            pointcloud_path,
            target,
            width=int(target_meta["width"]),
            height=int(target_meta["height"]),
            resolution=float(target_meta["resolution"]),
            origin_x=float(target_bounds["min_x"]),
            origin_y=float(target_bounds["min_z"]),
            flip_output=bool(target_meta.get("flip_y", False)),
            np=np,
            Image=Image,
        )
        assets["rgb_pointcloud"] = target

    if gt_display is not None:
        _add_gt_boundary(occupancy_path, gt_display, np, Image)

    comparison_suffix = "；粉色线为 Habitat GT 边界" if gt_display is not None else "；不含 Habitat GT"
    descriptions = {
        "rgb_pointcloud": "LingBot-Map RGB 点云俯视投影",
        "semantic": "LingBot 几何 + OWLv2/SAM2" + comparison_suffix,
        "instance": "OWLv2/SAM2 跨帧三维实例融合" + comparison_suffix,
        "region": "RGB 开放词汇语义区域 + LingBot-Map 几何" + comparison_suffix,
        "occupancy": "LingBot-Map occupancy" + comparison_suffix,
        "habitat_gt": "Habitat navmesh 真值，仅用于对照与验收",
    }
    order = ("rgb_pointcloud", "semantic", "instance", "region", "occupancy", "habitat_gt")
    layers = [
        {"id": key, "label": LAYER_LABELS[key], "asset": f"/asset/map/{key}.png", "description": descriptions[key]}
        for key in order if key in assets
    ]
    target_meta["layers"] = layers
    return target_meta, assets


def summarize_region_map(region_map_path: Path, map_yaml: Path) -> list[dict[str, Any]]:
    """Return connected-region sizes and map-frame centroids for the dashboard."""
    np, _Image = _imports()
    metadata = load_map_metadata(map_yaml)
    width, height, _pixels = load_pgm(metadata.image)
    values = np.load(region_map_path, allow_pickle=False)
    if values.shape != (height, width):
        raise ValueError(
            f"{region_map_path.name} shape {values.shape} does not match occupancy "
            f"shape {(height, width)}"
        )
    regions: list[dict[str, Any]] = []
    for region_id in sorted(int(item) for item in np.unique(values) if int(item) > 0):
        rows, cols = np.nonzero(values == region_id)
        if not len(rows):
            continue
        regions.append(
            {
                "id": region_id,
                "name": f"区域 {region_id:02d}",
                "x": metadata.origin_x + (float(cols.mean()) + 0.5) * metadata.resolution,
                "y": metadata.origin_y + (float(rows.mean()) + 0.5) * metadata.resolution,
                "cells": int(len(rows)),
                "area_m2": float(len(rows) * metadata.resolution**2),
            }
        )
    return regions


__all__ = ["render_mapping_views", "summarize_region_map"]
