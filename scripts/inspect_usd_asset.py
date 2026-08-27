#!/usr/bin/env python3
"""Inspect a USD asset after bootstrapping Isaac Sim's bundled USD runtime."""

import argparse

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("asset")
args = parser.parse_args()

app = SimulationApp({"headless": True})
try:
    from pxr import Usd

    stage = Usd.Stage.Open(args.asset)
    if stage is None:
        raise RuntimeError(f"Could not open {args.asset}")
    for prim in stage.Traverse():
        properties = [
            prop.GetName()
            for prop in prim.GetProperties()
            if prop.GetName().startswith(("physics:", "physx", "material:"))
        ]
        print(
            str(prim.GetPath()),
            f"type={prim.GetTypeName()}",
            f"apis={list(prim.GetAppliedSchemas())}",
            f"properties={properties}",
        )
finally:
    app.close()
