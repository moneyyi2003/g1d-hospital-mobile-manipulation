"""Build the Isaac USD scene bundle for a CGS scanned office.

Inputs (already in the metric map frame, no transform needed):

    <input-dir>/pointcloud.ply   RGB point cloud
    <input-dir>/map.pgm          trinary occupancy
    <input-dir>/map.yaml

Outputs:

    <output-dir>/visual_usd/cgs_office_points.usd   UsdGeom.Points
    <output-dir>/collision_usda/cgs_office.collision.usda
    <output-dir>/cgs_office.usda                    wrapper assembly
    <output-dir>/scene_asset_metadata.json

Run inside the Isaac container (has pxr + numpy):

    export PYTHONPATH=/isaac-sim/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312
    export LD_LIBRARY_PATH=/isaac-sim/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312/bin:/isaac-sim/kit/python/lib
    python3 scripts/build_cgs_office_scene.py [--input-dir ...] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs/CGS/forma_8181_pipeline/maps"
DEFAULT_OUTPUT_DIR = ROOT / "scene_asset/cgs_office"

# Occupied cells become low collision boxes; 0.6 m stops the G1-D wheeled base
# plus lower torso while staying inside the scanned point cloud height band.
OBSTACLE_HEIGHT_M = 0.6
FLOOR_THICKNESS_M = 0.02
POINT_WIDTH_M = 0.012

# Configured per run (module-level for the read/write helpers below).
PLY = MAP_YAML = MAP_PGM = OUT = None
VISUAL_USD = COLLISION_USDA = WRAPPER_USDA = None


def configure(input_dir: Path, output_dir: Path) -> None:
    global PLY, MAP_YAML, MAP_PGM, OUT, VISUAL_USD, COLLISION_USDA, WRAPPER_USDA
    PLY = input_dir / "pointcloud.ply"
    MAP_YAML = input_dir / "map.yaml"
    MAP_PGM = input_dir / "map.pgm"
    OUT = output_dir
    VISUAL_USD = OUT / "visual_usd/cgs_office_points.usd"
    COLLISION_USDA = OUT / "collision_usda/cgs_office.collision.usda"
    WRAPPER_USDA = OUT / "cgs_office.usda"


def read_ply_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary-little-endian PLY with xyz float + rgb uchar vertices."""
    with open(path, "rb") as stream:
        header_lines: list[bytes] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY has no end_header: {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        vertex_count = None
        for line in header_lines:
            if line.startswith(b"element vertex"):
                vertex_count = int(line.split()[-1])
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        payload = np.frombuffer(stream.read(), dtype=np.uint8)
    # 3 float32 + 3 uint8 per vertex; header offset handled above.
    expected = vertex_count * (3 * 4 + 3)
    if payload.size < expected:
        raise ValueError(f"truncated PLY payload: {payload.size} < {expected}")
    stride = 3 * 4 + 3
    xyz = payload[:expected].reshape(vertex_count, stride)[:, :12].view(np.float32)
    rgb = payload[:expected].reshape(vertex_count, stride)[:, 12:15]
    return np.ascontiguousarray(xyz), rgb.copy()


def read_occupancy_grid() -> tuple[np.ndarray, float, tuple[float, float]]:
    """Return (grid, resolution, origin) with grid[0] at minimum Y (map frame).

    grid value: 0 free, 1 occupied, -1 unknown — decoded from the PGM trinary
    encoding written by lingbot-sam3 (free->254, unknown->205, occupied->0).
    """
    text = MAP_YAML.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip("'\"")
    resolution = float(fields["resolution"])
    import re

    origin_values = [float(item) for item in re.findall(r"[-+0-9.eE]+", fields["origin"])]
    origin_x, origin_y = origin_values[0], origin_values[1]
    with open(MAP_PGM, "rb") as stream:
        tokens: list[bytes] = []
        while len(tokens) < 4:
            line = stream.readline()
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        width, height = int(tokens[1]), int(tokens[2])
        pixels = np.frombuffer(stream.read(width * height), dtype=np.uint8).reshape(height, width)
    # PGM row zero is max-Y; flip to min-Y-first to match map frame.
    pixels = pixels[::-1]
    grid = np.zeros((height, width), dtype=np.int8)
    grid[pixels == 205] = -1
    grid[pixels == 0] = 1
    return grid, resolution, (origin_x, origin_y)


def merge_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge True cells row-wise into (row_min, row_max, col_min, col_max) rects."""
    rects: list[tuple[int, int, int, int]] = []
    height, width = mask.shape
    for row in range(height):
        col = 0
        while col < width:
            if not mask[row, col]:
                col += 1
                continue
            col_end = col
            while col_end < width and mask[row, col_end]:
                col_end += 1
            # Extend downward over identical spans.
            row_end = row + 1
            while row_end < height and mask[row_end, col:col_end].all():
                row_end += 1
            rects.append((row, row_end - 1, col, col_end - 1))
            col = col_end
    return rects


def write_visual_usd(xyz: np.ndarray, rgb: np.ndarray) -> None:
    VISUAL_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(VISUAL_USD))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    # The referenced root prim must own child prims: this USD build silently
    # drops references whose target prim is a leaf (a bare Points root never
    # instantiates under the wrapper).
    root = UsdGeom.Xform.Define(stage, "/cgs_office_visual")
    points_prim = UsdGeom.Points.Define(stage, "/cgs_office_visual/points")
    # Usd.Attribute.Set accepts numpy arrays directly (pxr numpy adapter).
    points_prim.GetPointsAttr().Set(xyz.astype(np.float32))
    points_prim.GetWidthsAttr().Set(np.full(len(xyz), POINT_WIDTH_M, dtype=np.float32))
    points_prim.CreateDisplayColorAttr().Set((rgb.astype(np.float32) / 255.0))
    points_prim.CreateDoubleSidedAttr().Set(True)
    # Leaf-prim guard: this USD build also drops leaf children during
    # reference composition, so give the Points prim an inert child.
    UsdGeom.Scope.Define(stage, "/cgs_office_visual/points/_guard")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    print(f"[cgs-scene] visual points={len(xyz)} -> {VISUAL_USD}")


def write_collision_usda(grid: np.ndarray, resolution: float, origin_x: float, origin_y: float) -> None:
    COLLISION_USDA.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(COLLISION_USDA))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Xform.Define(stage, "/collision")
    height, width = grid.shape
    world_width = width * resolution
    world_height = height * resolution
    # Floor slab: thin box, top surface at z=0, covering the full map bounds.
    floor = UsdGeom.Cube.Define(stage, "/collision/floor")
    floor.CreateSizeAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor_xform = UsdGeom.Xformable(floor)
    floor_xform.AddTranslateOp().Set(
        Gf.Vec3d(
            origin_x + world_width / 2.0,
            origin_y + world_height / 2.0,
            -FLOOR_THICKNESS_M / 2.0,
        )
    )
    floor_xform.AddScaleOp().Set(
        Gf.Vec3f(world_width + 2.0, world_height + 2.0, FLOOR_THICKNESS_M)
    )
    # Obstacle boxes from occupied cells.
    rects = merge_rects(grid == 1)
    for index, (row_min, row_max, col_min, col_max) in enumerate(rects):
        box = UsdGeom.Cube.Define(stage, f"/collision/obstacle_{index:04d}")
        box.CreateSizeAttr(1.0)
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        sx = (col_max - col_min + 1) * resolution
        sy = (row_max - row_min + 1) * resolution
        cx = origin_x + (col_min + col_max + 1) * resolution / 2.0
        cy = origin_y + (row_min + row_max + 1) * resolution / 2.0
        box_xform = UsdGeom.Xformable(box)
        box_xform.AddTranslateOp().Set(
            Gf.Vec3d(cx, cy, OBSTACLE_HEIGHT_M / 2.0)
        )
        box_xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, OBSTACLE_HEIGHT_M))
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    print(f"[cgs-scene] collision boxes={len(rects)} floor=({world_width:.2f}x{world_height:.2f}) -> {COLLISION_USDA}")


def write_wrapper_usda() -> None:
    stage = Usd.Stage.CreateNew(str(WRAPPER_USDA))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/cgs_office").GetPrim())
    assembly = UsdGeom.Xform.Define(stage, "/cgs_office")
    assembly.GetPrim().SetKind("assembly")
    visual = UsdGeom.Xform.Define(stage, "/cgs_office/visual")
    visual.GetPrim().GetReferences().AddReference(str(VISUAL_USD.relative_to(OUT).as_posix()))
    collision = UsdGeom.Xform.Define(stage, "/cgs_office/collision")
    collision.GetPrim().GetReferences().AddReference(
        str(COLLISION_USDA.relative_to(OUT).as_posix())
    )
    light = UsdLux.SphereLight.Define(stage, "/cgs_office/room_light")
    light.CreateIntensityAttr(30000.0)
    light.CreateRadiusAttr(3.0)
    light.CreateColorAttr(Gf.Vec3f(1, 1, 1))
    light.AddTranslateOp().Set(Gf.Vec3d(2.2, 3.2, 2.5))
    stage.GetRootLayer().Save()
    print(f"[cgs-scene] wrapper -> {WRAPPER_USDA}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="maps dir with pointcloud.ply / map.pgm / map.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="where to write the USD bundle (default: scene_asset/cgs_office)")
    args = parser.parse_args(argv)
    configure(args.input_dir, args.output_dir)
    xyz, rgb = read_ply_rgb(PLY)
    print(f"[cgs-scene] ply vertices={len(xyz)} "
          f"x[{xyz[:, 0].min():.3f},{xyz[:, 0].max():.3f}] "
          f"y[{xyz[:, 1].min():.3f},{xyz[:, 1].max():.3f}] "
          f"z[{xyz[:, 2].min():.3f},{xyz[:, 2].max():.3f}]")
    grid, resolution, (origin_x, origin_y) = read_occupancy_grid()
    free = int((grid == 0).sum())
    occupied = int((grid == 1).sum())
    unknown = int((grid == -1).sum())
    print(f"[cgs-scene] occupancy {grid.shape} @{resolution} "
          f"free={free} occupied={occupied} unknown={unknown} origin=({origin_x:.3f},{origin_y:.3f})")
    OUT.mkdir(parents=True, exist_ok=True)
    write_visual_usd(xyz, rgb)
    write_collision_usda(grid, resolution, origin_x, origin_y)
    write_wrapper_usda()
    (OUT / "scene_asset_metadata.json").write_text(json.dumps({
        "scene_name": "cgs_office",
        "source_pointcloud": str(PLY),
        "visual_points": len(xyz),
        "point_cloud_bounds": {
            "min": [round(float(xyz[:, i].min()), 6) for i in range(3)],
            "max": [round(float(xyz[:, i].max()), 6) for i in range(3)],
        },
        "occupancy": {
            "shape": list(grid.shape),
            "resolution_m": resolution,
            "origin": [origin_x, origin_y],
            "free": free,
            "occupied": occupied,
            "unknown": unknown,
        },
        "collision": {
            "obstacle_boxes": len(merge_rects(grid == 1)),
            "obstacle_height_m": OBSTACLE_HEIGHT_M,
            "floor_thickness_m": FLOOR_THICKNESS_M,
        },
        "frame": "map_frame_identity_transform",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[cgs-scene] metadata -> {OUT / 'scene_asset_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
