#!/usr/bin/env python3
"""Extract a G1-D-footprint occupancy grid from home_lab's PhysX mesh.

This is a bootstrap safety product only.  It decides where RGB survey poses
may be sampled; the formal VLN map is still produced later from G1-D RGB by
LingBot-Map.  The source USD is opened read-only and never rewritten.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
USD = ROOT / "scene_asset/living_room/home_lab.usda"
OUTPUT = ROOT / "outputs/living_room_vln/bootstrap_collision_grid.json"
RESOLUTION_M = 0.10
ROBOT_RADIUS_M = 0.40
BODY_MIN_Z_M = 0.13
BODY_MAX_Z_M = 1.35


def main() -> int:
    app = SimulationApp({"headless": True, "active_gpu": 0, "multi_gpu": False})
    import omni.usd
    from pxr import UsdGeom

    context = omni.usd.get_context()
    if not context.open_stage(f"file://{USD}"):
        raise RuntimeError(f"cannot open {USD}")
    for _ in range(360):
        app.update()
    stage = context.get_stage()
    meshes = [
        prim for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and prim.GetAttribute("physics:collisionEnabled").Get()
    ]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one collision mesh, found {[str(p.GetPath()) for p in meshes]}")
    mesh = UsdGeom.Mesh(meshes[0])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(meshes[0])
    # Gf.Matrix4d.Transform in a Python loop takes several minutes for this
    # 260k-vertex mesh.  USD uses row-vector multiplication, so the same
    # transform is safely vectorized here.
    affine = np.asarray([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    transformed = (np.c_[points, np.ones(len(points))] @ affine)[:, :3]
    triangles = transformed[faces]
    zmin = triangles[:, :, 2].min(axis=1)
    zmax = triangles[:, :, 2].max(axis=1)
    # Exclude the floor, retain walls/cabinets/bed/table surfaces intersecting
    # the robot body.  Mesh vertices are at 2.5 cm density, so raster +
    # footprint dilation is conservative at a 10 cm planning resolution.
    selected = triangles[(zmax > BODY_MIN_Z_M) & (zmin < BODY_MAX_Z_M)]
    xy = selected[:, :, :2].reshape(-1, 2)
    bounds_min = transformed[:, :2].min(axis=0) - 0.20
    bounds_max = transformed[:, :2].max(axis=0) + 0.20
    width, height = np.ceil((bounds_max - bounds_min) / RESOLUTION_M).astype(int)
    occupied = np.zeros((int(height), int(width)), dtype=bool)
    ij = np.floor((xy - bounds_min) / RESOLUTION_M).astype(int)
    valid = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
    occupied[ij[valid, 1], ij[valid, 0]] = True
    radius_cells = int(np.ceil(ROBOT_RADIUS_M / RESOLUTION_M))
    yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    footprint = (xx * xx + yy * yy) <= radius_cells * radius_cells
    blocked = binary_dilation(occupied, structure=footprint)
    payload = {
        "schema_version": 1,
        "source": str(USD),
        "source_collision_mesh": str(meshes[0].GetPath()),
        "frame_id": "home_lab_world",
        "resolution_m": RESOLUTION_M,
        "origin_xy_m": bounds_min.tolist(),
        "shape_rows_cols": [int(height), int(width)],
        "robot_radius_m": ROBOT_RADIUS_M,
        "body_height_range_m": [BODY_MIN_Z_M, BODY_MAX_Z_M],
        "free": (~blocked).astype(np.uint8).tolist(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "free"}, indent=2), flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
