"""Thin artifact adapter around the official LingBot-Map implementation.

No model architecture is copied here. The adapter imports the locked official
demo helpers, runs RGB-only inference, and serializes the resulting geometry in
a stable per-frame contract consumed by occupancy and SAM3 projection code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..errors import ConfigurationError
from ..upstreams import require_upstream


@dataclass(frozen=True)
class LingBotInferenceConfig:
    mode: str = "streaming"
    image_size: int = 518
    patch_size: int = 14
    num_scale_frames: int = 8
    keyframe_interval: int = 1
    max_frame_num: int = 1024
    kv_cache_sliding_window: int = 64
    camera_num_iterations: int = 4
    window_size: int = 64
    overlap_size: int = 16

    def validate(self) -> None:
        if self.mode not in {"streaming", "windowed"}:
            raise ConfigurationError("LingBot mode must be streaming or windowed")
        if min(
            self.image_size,
            self.patch_size,
            self.num_scale_frames,
            self.keyframe_interval,
            self.max_frame_num,
        ) < 1:
            raise ConfigurationError("LingBot inference settings must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unbatch_sequence(value: Any, *, name: str, unbatched_ndim: int):
    """Normalize a single-sequence model output and reject ambiguous shapes."""

    ndim = getattr(value, "ndim", None)
    shape = getattr(value, "shape", ())
    if ndim == unbatched_ndim + 1 and shape[0] == 1:
        value = value[0]
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", ())
    if ndim != unbatched_ndim:
        raise ConfigurationError(
            f"LingBot {name} has unexpected shape {tuple(shape)}; "
            f"expected {unbatched_ndim} dimensions after removing a single batch"
        )
    return value


def _load_official_demo(checkout: Path):
    demo_path = checkout / "demo.py"
    spec = importlib.util.spec_from_file_location("lingbot_map_official_demo", demo_path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"Cannot import official LingBot demo: {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_lingbot_map(
    rgb_directory: str | Path,
    checkpoint: str | Path,
    output_directory: str | Path,
    config: LingBotInferenceConfig | None = None,
) -> dict[str, Any]:
    """Run the official model and emit aligned RGB/geometry frame artifacts."""
    config = config or LingBotInferenceConfig()
    config.validate()
    upstream = require_upstream("lingbot-map")
    rgb_root = Path(rgb_directory).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_root = Path(output_directory).expanduser().resolve()
    if not rgb_root.is_dir():
        raise ConfigurationError(f"RGB directory does not exist: {rgb_root}")
    if not checkpoint_path.is_file():
        raise ConfigurationError(f"LingBot checkpoint does not exist: {checkpoint_path}")

    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ConfigurationError(
            "LingBot inference needs its separate official PyTorch environment"
        ) from exc
    if not torch.cuda.is_available():
        raise ConfigurationError("Official LingBot-Map inference requires a CUDA GPU")

    # Import from the locked checkout rather than a potentially different pip package.
    import sys

    checkout_text = str(upstream.checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    official = _load_official_demo(upstream.checkout)
    images, source_paths, _ = official.load_images(
        image_folder=str(rgb_root),
        image_size=config.image_size,
        patch_size=config.patch_size,
    )
    if images.shape[0] < 2:
        raise ConfigurationError("LingBot-Map requires at least two RGB frames")
    args = SimpleNamespace(
        mode=config.mode,
        image_size=config.image_size,
        patch_size=config.patch_size,
        enable_3d_rope=True,
        max_frame_num=config.max_frame_num,
        kv_cache_sliding_window=config.kv_cache_sliding_window,
        num_scale_frames=min(config.num_scale_frames, int(images.shape[0])),
        keyframe_interval=config.keyframe_interval,
        camera_num_iterations=config.camera_num_iterations,
        use_sdpa=True,
        model_path=str(checkpoint_path),
    )
    device = torch.device("cuda")
    model = official.load_model(args, device)
    images = images.to(device)
    output_device = torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if config.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=config.keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=config.window_size,
                overlap_size=config.overlap_size,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=config.keyframe_interval,
                output_device=output_device,
            )
    images_for_post = predictions.get("images", images)
    predictions, images_cpu = official.postprocess(predictions, images_for_post)
    # Streaming inference keeps RGB as [1, frames, channels, height, width],
    # while official postprocess unbatches prediction tensors only.  Normalize
    # that last model-owned batch dimension at the adapter boundary.
    images_cpu = _unbatch_sequence(
        images_cpu, name="postprocessed RGB", unbatched_ndim=4
    )

    frame_count = int(images_cpu.shape[0])
    prediction_root = output_root / "predictions"
    preprocessed_root = output_root / "preprocessed_rgb"
    prediction_root.mkdir(parents=True, exist_ok=True)
    preprocessed_root.mkdir(parents=True, exist_ok=True)
    # A forced rerun may contain fewer frames than a previous survey.  Remove
    # only adapter-owned frame artifacts so stale tails cannot be mistaken for
    # current predictions during alignment.
    for stale in prediction_root.glob("frame_*.npz"):
        stale.unlink()
    for stale in preprocessed_root.glob("*.png"):
        stale.unlink()
    images_numpy = images_cpu.detach().cpu().numpy()
    for frame_index in range(frame_count):
        rgb = images_numpy[frame_index].transpose(1, 2, 0)
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rgb_u8).save(preprocessed_root / f"{frame_index:06d}.png")
        artifact: dict[str, Any] = {"images": rgb_u8}
        for key in (
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
            "intrinsic",
        ):
            value = predictions.get(key)
            if value is not None and hasattr(value, "shape") and value.shape[0] == frame_count:
                artifact[key] = value[frame_index].detach().cpu().numpy() if hasattr(value, "detach") else value[frame_index]
        # Official postprocess names this value extrinsic after converting it
        # from world-to-camera to camera-to-world. Rename it at our boundary so
        # downstream code cannot silently invert it a second time.
        camera_to_world = predictions.get("extrinsic")
        if camera_to_world is not None and camera_to_world.shape[0] == frame_count:
            value = camera_to_world[frame_index]
            artifact["camera_to_world"] = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        np.savez_compressed(prediction_root / f"frame_{frame_index:06d}.npz", **artifact)

    manifest = {
        "schema_version": 1,
        "pipeline": "official_lingbot_map_rgb_only",
        "upstream": {"url": upstream.url, "commit": upstream.commit},
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "input": {
            "rgb_directory": str(rgb_root),
            "source_frames": [str(Path(item).name) for item in source_paths],
        },
        "outputs": {
            "frame_count": frame_count,
            "predictions": str(prediction_root),
            "preprocessed_rgb": str(preprocessed_root),
        },
        "config": asdict(config),
        "ground_truth_inputs": {
            "habitat_depth": False,
            "habitat_pose": False,
            "habitat_semantics": False,
            "habitat_navmesh": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "lingbot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["LingBotInferenceConfig", "run_lingbot_map"]
