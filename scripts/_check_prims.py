import sys
sys.path.insert(0, "/workspace")
from isaacsim.core.experimental.utils import stage as stage_utils
stage = stage_utils.get_current_stage()
for i in range(1, 10):
    path = f"/World/FamilyHomeObjects/Item{i:02d}"
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        xform = prim.GetAttribute("xformOp:translate")
        pos = xform.Get() if xform.IsValid() else "no translate"
        print(f"{path}: VALID, pos={pos}")
    else:
        print(f"{path}: INVALID")
