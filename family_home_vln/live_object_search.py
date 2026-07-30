"""Live RGB object search against the reviewed scan-derived object memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .discovery import normalize_label, run_object_discovery


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
    matches = []
    for candidate in discovery.get("objects", []):
        label = normalize_label(str(candidate.get("label", "")))
        if any(
            label == alias
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
    """Run category-free Florence discovery, then match a reviewed object."""

    target = load_reviewed_object(catalog_path, target_query)
    discovery_path = output_file.parent / "category_free_discovery.json"
    discovery = run_object_discovery(
        manifest_path,
        rgb_dir,
        discovery_path,
        model_path=model_path,
        maximum_frames=maximum_frames,
        min_frame_occurrences=1,
        max_objects=32,
        infer=infer,
    )
    matches = match_live_discovery(target, discovery)
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
            "target_used_only_after_inference_for_matching": True,
            "discovery_artifact": str(discovery_path),
        },
        "failure_code": "" if matches else "target_not_found",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "load_reviewed_object",
    "match_live_discovery",
    "search_live_rgb",
]
