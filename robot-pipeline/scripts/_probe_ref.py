"""Diagnose why some USD references fail to compose in the kit pxr build."""

from __future__ import annotations

import sys

from pxr import Sdf, Usd, UsdGeom


def fresh_test() -> None:
    stage = Usd.Stage.CreateNew("/tmp/fresh_x.usda")
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    x = UsdGeom.Xform.Define(stage, "/fresh_x")
    stage.SetDefaultPrim(x.GetPrim())
    stage.GetRootLayer().Save()

    probe = Usd.Stage.CreateNew("/tmp/fresh_probe.usda")
    root = probe.DefinePrim("/probe")
    root.GetReferences().AddReference("/tmp/fresh_x.usda", "/fresh_x")
    probe.GetRootLayer().Save()

    opened = Usd.Stage.Open("/tmp/fresh_probe.usda")
    print("fresh probe prims:", len(list(opened.Traverse())))


def probe_collision_subdir() -> None:
    # reference the collision file (known to work) from a fresh probe
    probe = Usd.Stage.CreateNew("/tmp/fresh_probe2.usda")
    root = probe.DefinePrim("/probe")
    root.GetReferences().AddReference(
        "/workspace/scene_asset/cgs_office/collision_usda/cgs_office.collision.usda",
        "/collision",
    )
    probe.GetRootLayer().Save()
    opened = Usd.Stage.Open("/tmp/fresh_probe2.usda")
    print("collision probe prims:", len(list(opened.Traverse())))


if __name__ == "__main__":
    fresh_test()
    probe_collision_subdir()
