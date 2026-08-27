"""Audit an Isaac USD scene before attaching the G1-D navigation runtime.

This script intentionally composes a scene as a reference on a fresh stage.
That is the same loading path used by the navigation runners and avoids the
Isaac Sim 6.0 crash observed when an old 4.1 HTTP asset is opened as the root
stage directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="Local path or HTTP(S) URL to a USD scene")
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Include top-level prim bounds when its name contains this text",
    )
    return parser.parse_args()


args = parse_args()
if args.settle_steps < 1:
    raise SystemExit("--settle-steps must be positive")

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "disable_viewport_updates": True,
    }
)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from pxr import UsdGeom, UsdPhysics


def finite_bounds(
    cache: UsdGeom.BBoxCache, prim
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if value.IsEmpty():
        return None
    minimum = tuple(float(item) for item in value.GetMin())
    maximum = tuple(float(item) for item in value.GetMax())
    if (
        not all(math.isfinite(item) for item in (*minimum, *maximum))
        or any(left > right for left, right in zip(minimum, maximum))
    ):
        return None
    return minimum, maximum


def main() -> int:
    stage_utils.create_new_stage()
    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage_utils.add_reference_to_stage(args.scene, "/World/Scene")
    app_utils.update_app(steps=args.settle_steps)

    stage = stage_utils.get_current_stage()
    scene = stage.GetPrimAtPath("/World/Scene")
    if not scene.IsValid() or not scene.GetChildren():
        raise RuntimeError(f"scene did not compose any prims: {args.scene}")

    cache = UsdGeom.BBoxCache(
        0.0,
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )
    prims = list(stage.Traverse())
    scene_bounds = finite_bounds(cache, scene)
    if scene_bounds is None:
        raise RuntimeError("scene bounds are not finite")

    filters = tuple(value.casefold() for value in args.name_contains)
    selected_bounds = []
    if filters:
        for child in scene.GetChildren():
            if not any(value in child.GetName().casefold() for value in filters):
                continue
            bounds = finite_bounds(cache, child)
            if bounds is None:
                continue
            selected_bounds.append(
                {
                    "path": str(child.GetPath()),
                    "type": child.GetTypeName(),
                    "minimum": [round(item, 6) for item in bounds[0]],
                    "maximum": [round(item, 6) for item in bounds[1]],
                }
            )

    result = {
        "schema_version": 1,
        "scene": args.scene,
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "prim_count": len(prims),
        "mesh_count": sum(prim.IsA(UsdGeom.Mesh) for prim in prims),
        "collision_prim_count": sum(
            prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims
        ),
        "bounds": {
            "minimum": list(scene_bounds[0]),
            "maximum": list(scene_bounds[1]),
        },
        "selected_top_level_bounds": selected_bounds,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


try:
    exit_code = main()
finally:
    simulation_app.close()

raise SystemExit(exit_code)
