#!/usr/bin/env python3
"""Create a relocatable ZIP consumed by the family-home Web dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/family_home_vln"
DEFAULT_OUTPUT = ROOT / "outputs/family_home_web/family_home_map_bundle.zip"

DIRECTORIES = ("lingbot_map", "map_preview", "semantic")
FILES = (
    "mapping_summary.json",
    "places_formal.json",
    "objects_formal.json",
    "lingbot_to_family_home.json",
    "survey/capture_manifest.json",
    "discovery/object_discovery.json",
    "sam3/sam3_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    required = [source / name for name in FILES[:3]]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required map files: " + ", ".join(map(str, missing)))

    summary = json.loads((source / "mapping_summary.json").read_text(encoding="utf-8"))
    summary["assets"] = {
        key: f"map_preview/{key}.png"
        for key in ("rgb_pointcloud", "semantic", "occupancy", "region")
    }
    summary["inputs"] = {
        "survey_manifest": "survey/capture_manifest.json",
        "discovery": "discovery/object_discovery.json",
        "alignment": "lingbot_to_family_home.json",
        "sam3": "sam3/sam3_manifest.json",
        "semantic_observations": "semantic/sam3_observations.json",
        "places": "places_formal.json",
        "objects": "objects_formal.json",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="family-home-map-") as directory:
        rewritten = Path(directory) / "mapping_summary.json"
        rewritten.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(rewritten, "family_home_map/mapping_summary.json")
            for name in FILES[1:]:
                path = source / name
                if path.is_file():
                    bundle.write(path, f"family_home_map/{name}")
            for dirname in DIRECTORIES:
                root = source / dirname
                if not root.is_dir():
                    continue
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        bundle.write(
                            path,
                            "family_home_map/" + path.relative_to(source).as_posix(),
                        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
