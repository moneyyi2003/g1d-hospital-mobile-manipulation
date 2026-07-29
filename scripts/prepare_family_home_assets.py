#!/usr/bin/env python3
"""Convert local ReplicaCAD household GLBs into Isaac-ready USD assets."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


args = parse_args()

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": args.headless})

import omni.kit.asset_converter  # noqa: E402

from family_home_vln.household_objects import (  # noqa: E402
    HOUSEHOLD_OBJECTS,
    PREPARED_ASSET_ROOT,
    asset_manifest,
)


def convert(source: Path, target: Path) -> None:
    if target.is_file() and not args.force:
        print(f"[Home assets] reuse {target}")
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animation = True
    context.ignore_cameras = True
    context.single_mesh = False
    context.smooth_normals = True
    context.preview_surface = True
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = False
    converter = omni.kit.asset_converter.get_instance()
    task = converter.create_converter_task(
        str(source.resolve()),
        str(target.resolve()),
        lambda *_args: None,
        context,
    )
    future = asyncio.ensure_future(task.wait_until_finished())
    while not future.done():
        simulation_app.update()
    if not future.result():
        raise RuntimeError(
            f"asset conversion failed for {source}: "
            f"{task.get_status()} {task.get_error_message()}"
        )
    print(f"[Home assets] converted {source.name} -> {target}")


def main() -> int:
    try:
        for item in HOUSEHOLD_OBJECTS:
            convert(item.source_path, item.prepared_usd)
        PREPARED_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
        manifest = PREPARED_ASSET_ROOT / "asset_manifest.json"
        manifest.write_text(
            json.dumps(asset_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Home assets] manifest: {manifest}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
