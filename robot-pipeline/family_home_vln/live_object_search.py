"""Live RGB object search against the reviewed scan-derived object memory."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from .discovery import (
    create_florence_backend,
    normalize_label,
    run_object_discovery,
    sample_frame_indices,
    validate_survey,
)


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _aggregate_grounded_instances(
    raw: list[dict[str, Any]], *, image_size: tuple[int, int], phrase: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep separately grounded instances instead of merging equal labels."""

    width, height = image_size
    image_area = float(width * height)
    clusters: list[list[dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for item in raw:
        bbox = item["bbox"]
        area_ratio = (
            max(0.0, bbox[2] - bbox[0])
            * max(0.0, bbox[3] - bbox[1])
            / image_area
        )
        if not 0.002 <= area_ratio <= 0.70:
            rejected.append({**item, "reason": "bbox_area_outside_gate"})
            continue
        item = {**item, "area_ratio": area_ratio}
        best_index = -1
        best_iou = 0.0
        for index, cluster in enumerate(clusters):
            overlap = max(_bbox_iou(bbox, other["bbox"]) for other in cluster)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0 and best_iou >= 0.30:
            clusters[best_index].append(item)
        else:
            clusters.append([item])

    objects = []
    for instance_index, cluster in enumerate(clusters):
        frames = {int(item["frame_index"]) for item in cluster}
        example = max(cluster, key=lambda item: item["area_ratio"])
        areas = sorted(float(item["area_ratio"]) for item in cluster)
        objects.append(
            {
                "label": phrase,
                "sam3_prompt": phrase,
                "grounding_phrase": phrase,
                "grounded_instance": instance_index,
                "prompt_frame": min(frames),
                "frame_occurrences": len(frames),
                "raw_detection_count": len(cluster),
                "median_area_ratio": areas[len(areas) // 2],
                "example": {
                    "frame_index": int(example["frame_index"]),
                    "bbox": list(example["bbox"]),
                    "task": "<CAPTION_TO_PHRASE_GROUNDING>",
                },
            }
        )
    objects.sort(
        key=lambda item: (-item["frame_occurrences"], item["grounded_instance"])
    )
    return objects, rejected


def load_reviewed_object(catalog_path: Path, query: str) -> dict[str, Any]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    query_key = normalize_label(query)
    matches = []
    for item in payload.get("objects", []):
        if item.get("status") != "approved":
            continue
        keys = {
            normalize_label(str(item.get("object_id", ""))),
            normalize_label(str(item.get("source_label", ""))),
            *(normalize_label(str(alias)) for alias in item.get("aliases", [])),
        }
        if query_key in keys:
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(
            f"reviewed object query {query!r} resolved to {len(matches)} objects"
        )
    return matches[0]


def match_live_discovery(
    target: dict[str, Any], discovery: dict[str, Any]
) -> list[dict[str, Any]]:
    """Match after inference; target categories are never given to the model."""

    aliases = {
        normalize_label(str(target.get("source_label", ""))),
        *(normalize_label(str(alias)) for alias in target.get("aliases", [])),
    }
    # Scan-time open-vocabulary aliases may include broad container classes.
    # They are useful for recall during mapping but unsafe for manipulation:
    # a table panel or robot body labelled "box" must not authorize a cup
    # grasp.  Keep only cup-specific names for an audited cup target.
    cup_names = {"coffee cup", "cup", "mug", "杯子", "水杯", "咖啡杯"}
    cup_target = bool(aliases & cup_names)
    if cup_target:
        aliases = aliases & cup_names
    matches = []
    for candidate in discovery.get("objects", []):
        label = normalize_label(str(candidate.get("label", "")))
        if any(
            label == alias
            or (cup_target and alias in label)
            or (len(alias) >= 4 and alias in label)
            or (len(label) >= 4 and label in alias)
            for alias in aliases
        ):
            matches.append(candidate)
    return matches


def search_live_rgb(
    manifest_path: Path,
    rgb_dir: Path,
    catalog_path: Path,
    target_query: str,
    output_file: Path,
    *,
    model_path: Path,
    maximum_frames: int = 12,
    infer: Callable[[Path, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run category-free discovery, then RGB target grounding if it misses."""

    target = load_reviewed_object(catalog_path, target_query)
    backend = infer or create_florence_backend(model_path)
    discovery_path = output_file.parent / "category_free_discovery.json"
    discovery = run_object_discovery(
        manifest_path,
        rgb_dir,
        discovery_path,
        model_path=model_path,
        maximum_frames=maximum_frames,
        min_frame_occurrences=1,
        max_objects=32,
        infer=backend,
    )
    matches = match_live_discovery(target, discovery)
    grounding_path = output_file.parent / "target_phrase_grounding.json"
    grounding_used = False
    if not matches:
        grounding_used = True
        manifest, rgb = validate_survey(manifest_path, rgb_dir)
        indices = sample_frame_indices(len(rgb), maximum_frames)
        phrase = normalize_label(str(target.get("source_label", "")))
        task = f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}"
        raw = []
        for frame_index in indices:
            for item in backend(rgb[frame_index], task):
                raw.append(
                    {
                        "frame_index": frame_index,
                        "image": rgb[frame_index].name,
                        "task": "<CAPTION_TO_PHRASE_GROUNDING>",
                        "label": str(item.get("label") or phrase),
                        "bbox": [float(value) for value in item["bbox"]],
                        "view": str(item.get("view", "full")),
                    }
                )
        grounded, rejected = _aggregate_grounded_instances(
            raw,
            image_size=tuple(manifest["camera"]["resolution"]),
            phrase=phrase,
        )
        # Florence sometimes returns the input phrase without repeating it in
        # its parsed label.  These boxes are already conditioned on the single
        # reviewed target phrase, so make the provenance explicit and retain
        # them as that target rather than applying another string-name gate.
        matches = grounded
        grounding_payload = {
            "schema_version": 1,
            "artifact_type": "live_rgb_target_phrase_grounding",
            "target_phrase": phrase,
            "rgb_only": True,
            "usd_semantics_read": False,
            "scene_object_coordinates_read": False,
            "objects": grounded,
            "raw_detections": raw,
            "rejected": rejected,
        }
        grounding_path.write_text(
            json.dumps(grounding_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "artifact_type": "live_category_free_rgb_object_confirmation",
        "success": bool(matches),
        "target": {
            "object_id": target["object_id"],
            "source_label": target["source_label"],
            "aliases": target.get("aliases", []),
            "map_position": target["map_position"],
        },
        "live_matches": matches,
        "inference": {
            "category_list_supplied_to_model": False,
            "target_used_only_after_inference_for_matching": not grounding_used,
            "discovery_artifact": str(discovery_path),
            "target_phrase_grounding_used": grounding_used,
            "target_phrase_grounding_artifact": (
                str(grounding_path) if grounding_used else ""
            ),
            "usd_semantics_read": False,
            "scene_object_coordinates_read": False,
        },
        "failure_code": "" if matches else "target_not_found",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def manipulation_view_gate(
    search_result: dict[str, Any],
    capture_manifest: dict[str, Any],
    *,
    image_size: tuple[int, int],
    edge_margin_ratio: float = 0.08,
    minimum_area_ratio: float = 0.004,
    maximum_area_ratio: float = 0.55,
) -> dict[str, Any]:
    """Select a manipulation-safe live view from category-free detections.

    The gate uses the bounding boxes produced before target-name matching.  It
    does not consume simulator coordinates or USD semantics.  A target touching
    the image edge is rejected even when SEARCH_OBJECT found it, because such a
    view is unsuitable for a downstream VLA policy.
    """

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= edge_margin_ratio < 0.5:
        raise ValueError("edge_margin_ratio must be in [0, 0.5)")
    if not 0.0 < minimum_area_ratio < maximum_area_ratio <= 1.0:
        raise ValueError("invalid manipulation area-ratio bounds")

    frames = {
        int(item.get("frame", -1)): item
        for item in capture_manifest.get("frames", [])
        if isinstance(item, dict)
    }
    candidates: list[dict[str, Any]] = []
    margin_x = width * edge_margin_ratio
    margin_y = height * edge_margin_ratio
    image_area = float(width * height)
    for match in search_result.get("live_matches", []):
        example = match.get("example", {})
        bbox = [float(value) for value in example.get("bbox", [])]
        frame_index = int(example.get("frame_index", -1))
        if len(bbox) != 4 or frame_index not in frames:
            continue
        x1, y1, x2, y2 = bbox
        area_ratio = (
            max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        )
        edge_clear = (
            x1 >= margin_x
            and y1 >= margin_y
            and x2 <= width - margin_x
            and y2 <= height - margin_y
        )
        area_clear = minimum_area_ratio <= area_ratio <= maximum_area_ratio
        center_error = math.hypot(
            ((x1 + x2) / 2.0 - width / 2.0) / width,
            ((y1 + y2) / 2.0 - height / 2.0) / height,
        )
        candidates.append(
            {
                "frame_index": frame_index,
                "bbox": bbox,
                "area_ratio": area_ratio,
                "edge_clear": edge_clear,
                "area_clear": area_clear,
                "center_error": center_error,
                "robot_pose": frames[frame_index].get("robot_pose", {}),
                "camera_downward_pitch_deg": frames[frame_index].get(
                    "camera_downward_pitch_deg"
                ),
            }
        )
    safe = [
        item
        for item in candidates
        if item["edge_clear"] and item["area_clear"]
    ]
    safe.sort(key=lambda item: (item["center_error"], -item["area_ratio"]))
    selected = safe[0] if safe else None
    return {
        "ready": selected is not None,
        "failure_code": "" if selected is not None else "bad_viewpoint",
        "reason": (
            "target_bbox_inside_manipulation_view_gate"
            if selected is not None
            else "no_live_target_bbox_clears_edge_and_scale_gates"
        ),
        "edge_margin_ratio": edge_margin_ratio,
        "minimum_area_ratio": minimum_area_ratio,
        "maximum_area_ratio": maximum_area_ratio,
        "selected": selected,
        "candidates": candidates,
    }


__all__ = [
    "load_reviewed_object",
    "manipulation_view_gate",
    "match_live_discovery",
    "search_live_rgb",
]
