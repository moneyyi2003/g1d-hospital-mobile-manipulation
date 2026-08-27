#!/usr/bin/env python3
"""Category-free live RGB search sidecar for a running Isaac mission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from family_home_vln.live_object_search import search_live_rgb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "checkpoints/florence-2-base-ft",
    )
    parser.add_argument("--maximum-frames", type=int, default=12)
    args = parser.parse_args()
    result = search_live_rgb(
        args.manifest.resolve(),
        args.rgb_dir.resolve(),
        args.catalog.resolve(),
        args.target,
        args.output.resolve(),
        model_path=args.model.resolve(),
        maximum_frames=args.maximum_frames,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
