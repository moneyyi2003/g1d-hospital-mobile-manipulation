#!/usr/bin/env python3
"""Run inference with a LoRA-fine-tuned OpenVLA-7B grasp model.

Usage::

    python scripts/infer_openvla_grasp.py \
        --model checkpoints/openvla-7b \
        --lora checkpoints/openvla-g1d-grasp-lora \
        --image path/to/head_rgb.png \
        --instruction "move the robot hand toward the coffee cup" \
        --output outputs/grasp_inference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTION_LABELS = ("dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="base OpenVLA-7B checkpoint dir")
    p.add_argument("--lora", type=Path, required=True, help="LoRA adapter directory")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unnorm-key", default="g1d_cup_grasp")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for name in ("model", "lora", "image", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, (ROOT / path).resolve())

    if not args.model.is_dir():
        raise FileNotFoundError(f"--model not found: {args.model}")
    if not args.lora.is_dir():
        raise FileNotFoundError(f"--lora not found: {args.lora}")
    if not args.image.is_file():
        raise FileNotFoundError(f"--image not found: {args.image}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # ---- Load base model + processor ----------------------------------
    print(f"Loading base model from {args.model} ...")
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ---- Load LoRA adapter --------------------------------------------
    print(f"Loading LoRA adapter from {args.lora} ...")
    model.language_model = PeftModel.from_pretrained(
        model.language_model, args.lora,
    )

    # ---- Load custom norm_stats ---------------------------------------
    stats_path = args.lora / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"  Loaded norm_stats: unnorm_key={args.unnorm_key}")
        if args.unnorm_key in model.norm_stats:
            action_stats = model.norm_stats[args.unnorm_key]["action"]
            for d, lbl in enumerate(ACTION_LABELS):
                print(f"    {lbl}: q01={action_stats['q01'][d]:.4f}  q99={action_stats['q99'][d]:.4f}")
    else:
        print("  WARNING: No dataset_statistics.json found; using checkpoint default norm_stats")

    model.to(device)
    model.eval()

    # ---- Run inference ------------------------------------------------
    prompt = (
        "In: What action should the robot take to "
        f"{args.instruction.strip().lower()}?\nOut:"
    )
    image = Image.open(args.image).convert("RGB")
    inputs = processor(prompt, image).to(device, dtype=dtype)

    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

    action_list = [float(v) for v in action.tolist()]

    print("\nPredicted action (7-D delta):")
    for lbl, val in zip(ACTION_LABELS, action_list):
        print(f"  {lbl:>12s}: {val:+.6f}")

    # ---- Save output -------------------------------------------------
    payload = {
        "schema_version": 1,
        "artifact_type": "openvla_lora_grasp_inference",
        "model": {
            "base": str(args.model.resolve()),
            "lora_adapter": str(args.lora.resolve()),
            "unnorm_key": args.unnorm_key,
            "dtype": args.dtype,
            "device": args.device,
        },
        "observation": {
            "image": str(args.image.resolve()),
            "instruction": args.instruction.strip(),
            "prompt": prompt,
        },
        "action": action_list,
        "action_semantics": list(ACTION_LABELS),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)

    print(f"\nSaved inference to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
