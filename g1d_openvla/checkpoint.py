"""Cheap local completeness checks for sharded OpenVLA checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointStatus:
    ready: bool
    expected_bytes: int
    actual_bytes: int
    missing_files: tuple[str, ...]

    def require_ready(self) -> None:
        if self.ready:
            return
        raise RuntimeError(
            "OpenVLA checkpoint is incomplete: "
            f"{self.actual_bytes}/{self.expected_bytes} bytes; "
            f"missing={list(self.missing_files)}"
        )


def inspect_checkpoint(model_dir: Path) -> CheckpointStatus:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return CheckpointStatus(False, 0, 0, (index_path.name,))
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    shards = tuple(sorted(set(payload.get("weight_map", {}).values())))
    expected = int(payload.get("metadata", {}).get("total_size", 0))
    missing_items = [
        name for name in shards if not (model_dir / name).is_file()
    ]
    missing_items.extend(
        f"{name}.aria2"
        for name in shards
        if (model_dir / f"{name}.aria2").exists()
    )
    missing = tuple(missing_items)
    actual = sum(
        (model_dir / name).stat().st_size
        for name in shards
        if (model_dir / name).is_file()
    )
    # ``metadata.total_size`` counts tensor payload bytes. A complete
    # safetensors shard is slightly larger because its JSON header is part of
    # the file too, so exact equality rejects valid official checkpoints.
    return CheckpointStatus(
        ready=bool(shards) and expected > 0 and not missing and actual >= expected,
        expected_bytes=expected,
        actual_bytes=actual,
        missing_files=missing,
    )


__all__ = ["CheckpointStatus", "inspect_checkpoint"]
