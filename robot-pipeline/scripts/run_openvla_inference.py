#!/usr/bin/env python3
"""Run one real RGB + instruction inference with an OpenVLA checkpoint."""

from __future__ import annotations

import os as _os
# GPU isolation is selected by the dashboard/runner.  Do not overwrite the
# parent's CUDA_VISIBLE_DEVICES here: on the two-GPU Docker deployment the
# second exposed GPU can already be occupied by an unrelated host process.
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="optional PEFT/LoRA adapter trained for the G1-D grasp domain",
    )
    parser.add_argument(
        "--action-head",
        type=Path,
        default=None,
        help="optional OpenVLA-OFT L1 continuous action-head checkpoint",
    )
    parser.add_argument(
        "--dataset-statistics",
        type=Path,
        default=None,
        help="normalization statistics produced by the OFT training run",
    )
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
    for name in (
        "model",
        "image",
        "output",
        "adapter",
        "action_head",
        "dataset_statistics",
    ):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            setattr(args, name, ROOT / path)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    inspect_checkpoint(args.model).require_ready()
    if args.adapter is not None and not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"invalid OpenVLA adapter: {args.adapter}")
    if args.action_head is not None and not args.action_head.is_file():
        raise FileNotFoundError(f"invalid OpenVLA-OFT action head: {args.action_head}")
    if args.dataset_statistics is not None and not args.dataset_statistics.is_file():
        raise FileNotFoundError(
            f"invalid OpenVLA-OFT dataset statistics: {args.dataset_statistics}"
        )
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
    )
    dataset_statistics = args.dataset_statistics or (
        args.adapter / "dataset_statistics.json"
        if args.adapter is not None
        and (args.adapter / "dataset_statistics.json").is_file()
        else args.model / "dataset_statistics.json"
    )
    selected_unnorm_key = args.unnorm_key
    if dataset_statistics.is_file():
        model.norm_stats = json.loads(
            dataset_statistics.read_text(encoding="utf-8")
        )
        if selected_unnorm_key not in model.norm_stats:
            if args.adapter is not None and len(model.norm_stats) == 1:
                selected_unnorm_key = next(iter(model.norm_stats))
            else:
                raise KeyError(
                    f"unnorm key {selected_unnorm_key!r} is unavailable; "
                    f"choose one of {sorted(model.norm_stats)}"
                )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            is_trainable=False,
        )
    action_head = None
    if args.action_head is not None:
        from collections import OrderedDict

        from prismatic.models.action_heads import L1RegressionActionHead

        llm_dim = int(model.llm_dim)
        action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=7,
        )
        state = torch.load(args.action_head, map_location="cpu", weights_only=True)
        # OFT saves the DDP-wrapped module, hence the leading ``module.``.
        state = OrderedDict(
            (
                key.removeprefix("module."),
                value,
            )
            for key, value in state.items()
        )
        action_head.load_state_dict(state, strict=True)
        action_head = action_head.to(args.device, dtype=dtype)
        action_head.eval()
    model = model.to(args.device)
    model.eval()
    loaded_at = time.monotonic()

    prompt = (
        "In: What action should the robot take to "
        f"{args.instruction.strip().lower()}?\nOut:"
    )
    image = Image.open(args.image).convert("RGB")
    inputs = processor(prompt, image).to(args.device, dtype=dtype)
    with torch.inference_mode():
        prediction = model.predict_action(
            **inputs,
            unnorm_key=selected_unnorm_key,
            do_sample=False,
            action_head=action_head,
        )
    # Vanilla OpenVLA returns one 7-D action. OpenVLA-OFT returns
    # ``(action_chunk, hidden_states)`` with an 8x7 chunk even when the
    # continuous OFT action head is not supplied. Keep the complete chunk in
    # the artifact and hand the first receding-horizon action to the existing
    # G1-D safety/expert bridge.
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    action_array = prediction
    if hasattr(action_array, "detach"):
        action_array = action_array.detach().float().cpu().numpy()
    import numpy as np

    action_array = np.asarray(action_array, dtype=np.float32)
    action_chunk = (
        action_array.reshape(-1, 7)
        if action_array.size % 7 == 0
        else action_array.reshape(1, -1)
    )
    if action_chunk.shape[1] != 7:
        raise RuntimeError(f"OpenVLA returned invalid action shape {action_array.shape}")
    action = action_chunk[0]
    finished = time.monotonic()
    payload = {
        "schema_version": 1,
        "artifact_type": "openvla_single_frame_action_inference",
        "status": "succeeded",
        "success": True,
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": _sha256(args.model / "config.json"),
            "adapter": (
                {
                    "path": str(args.adapter.resolve()),
                    "config_sha256": _sha256(args.adapter / "adapter_config.json"),
                }
                if args.adapter is not None
                else None
            ),
            "action_head": (
                {
                    "path": str(args.action_head.resolve()),
                    "sha256": _sha256(args.action_head),
                    "kind": "openvla_oft_l1_regression",
                }
                if args.action_head is not None
                else None
            ),
            "unnorm_key": selected_unnorm_key,
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
        "action_chunk": [
            [float(value) for value in row.tolist()] for row in action_chunk
        ],
        "action_chunk_length": int(action_chunk.shape[0]),
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
