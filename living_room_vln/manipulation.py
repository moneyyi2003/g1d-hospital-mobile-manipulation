"""Runtime manipulation station for the scanned ``home_lab`` scene.

The reconstruction asset remains untouched.  This module adds a small,
physical table and a cup-sized rigid body only to the live Isaac stage, so
that perception and expert-data collection observe the same object that the
G1-D will later manipulate.
"""

from __future__ import annotations

from typing import Any


TABLE_PRIM_PATH = "/World/LivingRoomManipulation/table_top"
CUP_PRIM_PATH = "/World/LivingRoomManipulation/coffee_cup"
# This patch lies in the surveyed free-space component, ahead/right of the
# calibrated base pose.  It intentionally does not rely on any USD semantic
# label from the reconstruction.
BASE_POSE = (0.76093745, -1.74414063, 0.0)
TABLE_CENTER = (1.38, -2.12, 0.36)
TABLE_SIZE = (0.80, 0.62, 0.72)
CUP_CENTER = (1.18, -2.02, 0.795)


def add_manipulation_station(stage: Any) -> dict[str, object]:
    """Author a collision-enabled table and a G1-D-hand-sized physical cup.

    The cup is a 64 mm diameter, 105 mm tall, 80 g cylinder — small enough
    for the calibrated thumb/middle finger envelope.  The root is the rigid
    body targeted by the MaChuanhao expert; its child owns visible geometry
    and collision.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    if stage.GetPrimAtPath(TABLE_PRIM_PATH).IsValid():
        return {
            "table_prim_path": TABLE_PRIM_PATH,
            "cup_prim_path": CUP_PRIM_PATH,
            "base_pose": BASE_POSE,
            "cup_center_world_m": CUP_CENTER,
        }

    print("[living-room station] author table", flush=True)
    table = UsdGeom.Cube.Define(stage, TABLE_PRIM_PATH)
    table.CreateSizeAttr(1.0)
    table.CreateDisplayColorAttr([Gf.Vec3f(0.34, 0.22, 0.12)])
    table_xform = UsdGeom.Xformable(table)
    table_xform.AddTranslateOp().Set(Gf.Vec3d(*TABLE_CENTER))
    table_xform.AddScaleOp().Set(Gf.Vec3f(*TABLE_SIZE))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())

    print("[living-room station] author cup root", flush=True)
    cup_root = UsdGeom.Xform.Define(stage, CUP_PRIM_PATH)
    cup_xform = UsdGeom.Xformable(cup_root)
    cup_xform.AddTranslateOp().Set(Gf.Vec3d(*CUP_CENTER))
    # The Expert bridge injects xformOp:orient immediately before use.  Do
    # not author it during composition: this NuRec USD has a mixed transform
    # stack and Isaac 6 can terminate while composing an orient op here.
    print("[living-room station] author cup rigid body", flush=True)
    rigid = UsdPhysics.RigidBodyAPI.Apply(cup_root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    print("[living-room station] author cup mass", flush=True)
    UsdPhysics.MassAPI.Apply(cup_root.GetPrim()).CreateMassAttr(0.08)

    print("[living-room station] author cup geometry", flush=True)
    cup = UsdGeom.Cylinder.Define(stage, f"{CUP_PRIM_PATH}/geometry")
    cup.CreateRadiusAttr(0.032)
    cup.CreateHeightAttr(0.105)
    cup.CreateAxisAttr(UsdGeom.Tokens.z)
    cup.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.90, 0.78)])
    UsdPhysics.CollisionAPI.Apply(cup.GetPrim())

    print("[living-room station] complete", flush=True)
    return {
        "table_prim_path": TABLE_PRIM_PATH,
        "cup_prim_path": CUP_PRIM_PATH,
        "base_pose": BASE_POSE,
        "cup_center_world_m": CUP_CENTER,
        "cup_dimensions_m": {"diameter": 0.064, "height": 0.105},
        "dynamic": True,
    }
