#!/usr/bin/env python3
"""Run category-free object discovery on the G1-D family-home RGB survey."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from family_home_vln.discovery import run_object_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "outputs/family_home_vln",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints/florence-2-base-ft",
    )
    parser.add_argument("--maximum-frames", type=int, default=80)
    parser.add_argument("--min-frame-occurrences", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = args.artifacts.resolve()
    if not (args.model / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Florence-2 权重不存在：{args.model}；请先准备自主发现模型。"
        )
    run_object_discovery(
        artifacts / "survey/capture_manifest.json",
        artifacts / "survey/rgb",
        artifacts / "discovery/object_discovery.json",
        model_path=args.model,
        maximum_frames=args.maximum_frames,
        min_frame_occurrences=args.min_frame_occurrences,
        max_objects=args.max_objects,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
