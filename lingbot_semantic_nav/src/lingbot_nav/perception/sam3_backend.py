"""Thin video tracking adapter for the locked official SAM3 repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

from ..errors import ConfigurationError
from ..upstreams import require_upstream


@dataclass(frozen=True)
class Sam3TrackConfig:
    prompt_frame: int = 0
    probability_threshold: float = 0.5
    propagation_direction: str = "forward"
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = False

    def validate(self) -> None:
        if self.prompt_frame < 0:
            raise ConfigurationError("SAM3 prompt frame must be non-negative")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ConfigurationError("SAM3 probability threshold must be in [0, 1]")
        if self.propagation_direction not in {"forward", "backward", "both"}:
            raise ConfigurationError("Unsupported SAM3 propagation direction")


def _prompt_slug(index: int, prompt: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt).strip("_")
    return f"{index:03d}_{normalized[:48] or 'prompt'}"


def _save_frame(path: Path, prompt_index: int, outputs: dict[str, Any]) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError("SAM3 artifact serialization requires NumPy") from exc
    object_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=np.uint8)
    scores = np.asarray(outputs.get("out_probs", np.ones(len(object_ids))), dtype=np.float32)
    boxes = np.asarray(outputs.get("out_boxes_xywh", np.zeros((len(object_ids), 4))), dtype=np.float32)
    if masks.ndim != 3 or masks.shape[0] != len(object_ids):
        raise ConfigurationError(f"Unexpected SAM3 mask output shape: {masks.shape}")
    track_ids = np.asarray([f"p{prompt_index}:o{int(item)}" for item in object_ids])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        object_ids=object_ids,
        track_ids=track_ids,
        masks=masks,
        scores=scores,
        boxes_xywh=boxes,
    )
    return len(object_ids)


def run_sam3_tracking(
    video_resource: str | Path,
    prompts: Iterable[str],
    output_directory: str | Path,
    *,
    checkpoint: str | Path | None = None,
    config: Sam3TrackConfig | None = None,
) -> dict[str, Any]:
    """Run one official text-tracking session per concept prompt.

    Separate sessions preserve SAM3's native object IDs without assuming that
    repeated text prompts accumulate in one session. IDs are namespaced by
    prompt in the serialized contract.
    """
    config = config or Sam3TrackConfig()
    config.validate()
    prompt_list = tuple(item.strip() for item in prompts if item.strip())
    if not prompt_list:
        raise ConfigurationError("At least one SAM3 text prompt is required")
    resource = Path(video_resource).expanduser().resolve()
    if not resource.exists():
        raise ConfigurationError(f"SAM3 video resource does not exist: {resource}")
    output_root = Path(output_directory).expanduser().resolve()
    upstream = require_upstream("sam3")
    checkout_text = str(upstream.checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise ConfigurationError(
            "SAM3 must run in the official Python 3.12/PyTorch environment"
        ) from exc

    build_kwargs: dict[str, Any] = {}
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ConfigurationError(f"SAM3 checkpoint does not exist: {checkpoint_path}")
        build_kwargs["checkpoint_path"] = str(checkpoint_path)
    predictor = build_sam3_video_predictor(**build_kwargs)
    prompt_records = []
    for prompt_index, prompt in enumerate(prompt_list):
        slug = _prompt_slug(prompt_index, prompt)
        prompt_root = output_root / slug
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(resource),
            "offload_video_to_cpu": config.offload_video_to_cpu,
            "offload_state_to_cpu": config.offload_state_to_cpu,
        })
        session_id = response["session_id"]
        frames: dict[int, int] = {}
        try:
            response = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": config.prompt_frame,
                "text": prompt,
                "output_prob_thresh": config.probability_threshold,
            })
            frame_index = int(response["frame_index"])
            frames[frame_index] = _save_frame(
                prompt_root / f"frame_{frame_index:06d}.npz",
                prompt_index,
                response["outputs"],
            )
            stream = predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": config.propagation_direction,
                "start_frame_index": config.prompt_frame,
                "output_prob_thresh": config.probability_threshold,
            })
            for item in stream:
                frame_index = int(item["frame_index"])
                frames[frame_index] = _save_frame(
                    prompt_root / f"frame_{frame_index:06d}.npz",
                    prompt_index,
                    item["outputs"],
                )
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        prompt_records.append({
            "prompt": prompt,
            "prompt_index": prompt_index,
            "artifact_directory": str(prompt_root),
            "frames": len(frames),
            "detections": sum(frames.values()),
        })

    manifest = {
        "schema_version": 1,
        "pipeline": "official_sam3_text_video_tracking",
        "upstream": {"url": upstream.url, "commit": upstream.commit, "release": "SAM 3.1"},
        "video_resource": str(resource),
        "config": asdict(config),
        "prompts": prompt_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sam3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["Sam3TrackConfig", "run_sam3_tracking"]

