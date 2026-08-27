"""Build candidate catalogs and explicitly promote reviewed docking poses."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .errors import ConfigurationError
from .map_validation import load_map_metadata
from .mapping.docking import TraversabilityGrid, generate_docking_candidates
from .models import Pose2D


def map_bundle_sha256(map_yaml: str | Path) -> str:
    yaml_path = Path(map_yaml).expanduser().resolve()
    metadata = load_map_metadata(yaml_path)
    digest = hashlib.sha256()
    for path in (yaml_path, metadata.image):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _place_id(track_id: str, prompt: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{prompt}_{track_id}").strip("_").lower()
    return value[:96] or hashlib.sha256(f"{prompt}:{track_id}".encode()).hexdigest()[:16]


def _fuse_observations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for value in payload.get("observations", []):
        groups.setdefault(str(value["track_id"]), []).append(value)
    fused = []
    for track_id, observations in groups.items():
        weights = [max(1, int(item["point_count"])) * max(0.01, float(item["score"])) for item in observations]
        total = sum(weights)
        centroid = tuple(
            sum(float(item["centroid_xyz"][axis]) * weight for item, weight in zip(observations, weights)) / total
            for axis in range(3)
        )
        fused.append({
            "track_id": track_id,
            "prompt": str(observations[0]["prompt"]),
            "centroid_xyz": centroid,
            "observation_count": len(observations),
            "frame_indices": sorted({int(item["frame_index"]) for item in observations}),
        })
    return fused


def build_candidate_catalog(
    observations_file: str | Path,
    map_yaml: str | Path,
    output_file: str | Path,
    *,
    map_id: str,
    reachability_start: Pose2D,
    footprint_radius_m: float,
) -> dict[str, Any]:
    source = Path(observations_file)
    try:
        observations = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read 3D track observations: {exc}") from exc
    if observations.get("frame_id") != "map":
        raise ConfigurationError("3D track observations must already be in the map frame")
    if not map_id.strip():
        raise ConfigurationError("Map id must not be empty")
    grid = TraversabilityGrid(map_yaml, footprint_radius_m)
    reachable = grid.reachable_cells(reachability_start)
    places = []
    for instance in _fuse_observations(observations):
        x, y, z = instance["centroid_xyz"]
        candidates = generate_docking_candidates(grid, x, y, reachable)
        if not candidates:
            continue
        place_id = _place_id(instance["track_id"], instance["prompt"])
        places.append({
            "id": place_id,
            "name": instance["prompt"],
            "aliases": [instance["prompt"], place_id],
            "status": "candidate",
            "target": {"type": "sam3_instance", "source_id": instance["track_id"]},
            "region": "",
            "docking_candidates": [item.to_dict() for item in candidates],
            "selected_docking_candidate": "",
            "metadata": {
                "instance_center_xyz": {"x": x, "y": y, "z": z},
                "observation_count": instance["observation_count"],
                "frame_indices": instance["frame_indices"],
                "geometry_backend": "lingbot-map",
                "semantic_backend": "sam3",
                "ground_truth_inputs": False,
            },
        })
    result = {
        "schema_version": 2,
        "map": {
            "id": map_id,
            "sha256": map_bundle_sha256(map_yaml),
            "frame_id": "map",
        },
        "places": places,
    }
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def approve_place(
    catalog_file: str | Path,
    map_yaml: str | Path,
    *,
    place_id: str,
    candidate_id: str,
    reviewer: str,
    evidence: list[str],
) -> dict[str, Any]:
    if not reviewer.strip() or not evidence:
        raise ConfigurationError("Approval requires a reviewer and non-empty evidence")
    source = Path(catalog_file)
    try:
        catalog = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read place catalog: {exc}") from exc
    current_hash = map_bundle_sha256(map_yaml)
    if catalog.get("map", {}).get("sha256") != current_hash:
        raise ConfigurationError("Cannot approve a candidate against a different map bundle")
    place = next((item for item in catalog.get("places", []) if item.get("id") == place_id), None)
    if place is None:
        raise ConfigurationError(f"Unknown candidate place: {place_id}")
    candidate = next(
        (item for item in place.get("docking_candidates", []) if item.get("id") == candidate_id),
        None,
    )
    if candidate is None:
        raise ConfigurationError(f"Unknown docking candidate: {candidate_id}")
    checks = candidate.get("checks", {})
    if checks.get("occupancy_status") != "free" or checks.get("reachable") is not True:
        raise ConfigurationError("Cannot approve an unsafe or unreachable docking candidate")
    grid = TraversabilityGrid(map_yaml, float(checks["footprint_radius_m"]))
    pose = Pose2D.from_mapping(candidate["pose"], "map")
    if not grid.footprint_is_free(*grid.world_to_cell(pose.x, pose.y)):
        raise ConfigurationError("Docking candidate no longer passes the current footprint check")
    candidate["review"] = {
        "status": "accepted",
        "reviewer": reviewer.strip(),
        "evidence": list(evidence),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    place["selected_docking_candidate"] = candidate_id
    place["status"] = "approved"
    source.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return place


__all__ = ["approve_place", "build_candidate_catalog", "map_bundle_sha256"]

