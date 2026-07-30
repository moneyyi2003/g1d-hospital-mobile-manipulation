#!/usr/bin/env python3
"""Run one real RGB + instruction inference with an OpenVLA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1d_openvla import inspect_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    for name in ("model", "image", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    inspect_checkpoint(args.model).require_ready()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.instruction.strip():
        raise ValueError("--instruction cannot be empty")

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device)
    dataset_statistics = args.model / "dataset_statistics.json"
    if dataset_statistics.is_file():
        model.norm_stats = json.loads(
            dataset_statistics.read_text(encoding="utf-8")
        )
    model.eval()
    loaded_at = time.monotonic()

    prompt = (
        "In: What action should the robot take to "
        f"{args.instruction.strip().lower()}?\nOut:"
    )
    image = Image.open(args.image).convert("RGB")
    inputs = processor(prompt, image).to(args.device, dtype=dtype)
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=args.unnorm_key,
            do_sample=False,
        )
    finished = time.monotonic()
    payload = {
        "schema_version": 1,
        "artifact_type": "openvla_single_frame_action_inference",
        "status": "succeeded",
        "success": True,
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": _sha256(args.model / "config.json"),
            "unnorm_key": args.unnorm_key,
            "dtype": args.dtype,
            "device": args.device,
            "attention": args.attn_implementation,
            "action_statistics": (
                str(dataset_statistics.resolve())
                if dataset_statistics.is_file()
                else "checkpoint_config.norm_stats"
            ),
        },
        "observation": {
            "image": str(args.image.resolve()),
            "image_sha256": _sha256(args.image),
            "instruction": args.instruction.strip(),
            "prompt": prompt,
        },
        "action": [float(value) for value in action.tolist()],
        "action_semantics": [
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_roll",
            "delta_pitch",
            "delta_yaw",
            "gripper",
        ],
        "timing_sec": {
            "model_load": loaded_at - started,
            "inference": finished - loaded_at,
            "total": finished - started,
        },
        "execution": {
            "performed": False,
            "reason": (
                "Inference sidecar never writes robot joints; "
                "the G1-D safety adapter decides whether execution is allowed."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
