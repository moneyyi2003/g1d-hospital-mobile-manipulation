"""Open the scanned living-room asset and save an overview for calibration."""

from __future__ import annotations

import os
from pathlib import Path
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1920, "height": 1080, "active_gpu": 0, "multi_gpu": False})

import omni.usd
from isaacsim.core.rendering_manager import ViewportManager


ROOT = Path(__file__).resolve().parents[1]
STAGE_URL = f"file://{ROOT / 'scene_asset/living_room/home_lab.usda'}"
OUTPUT = ROOT / "outputs/living_room_vln/calibration_views"


def pump(steps: int) -> None:
    for _ in range(steps):
        app.update()


context = omni.usd.get_context()
if not context.open_stage(STAGE_URL):
    raise RuntimeError(f"cannot load living-room stage: {STAGE_URL}")
pump(360)

# The delivered assembly is authored as a standalone asset and includes a
# 60,000-intensity diagnostic light plus invert-tone-map flags.  Those settings
# blow out the NuRec render when it is composed into a normal Isaac viewport.
# Keep the asset file untouched; make the viewer-only overrides in memory.
stage = context.get_stage()
light = stage.GetPrimAtPath("/home_lab/collision_center_light")
if light.IsValid():
    # The source's diagnostic setting (60,000) clips the splat in the Isaac
    # viewport, but zero makes its baked appearance almost black.  This is a
    # viewer-only exposure calibration, not a source-asset edit.
    light.GetAttribute("inputs:intensity").Set(0.0)
import carb
settings = carb.settings.get_settings()
# NuRec encodes its appearance with the two authored inverse-compositing flags.
# Keep them enabled; the actual exposure issue was the auxiliary light's 60k
# intensity, which is calibrated above.
settings.set("/rtx/post/registeredCompositing/invertToneMap", True)
settings.set("/rtx/post/registeredCompositing/invertColorCorrection", True)
settings.set("/rtx/post/tonemap/op", 2)

from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

OUTPUT.mkdir(parents=True, exist_ok=True)
# NuRec scenes are meaningful from captured, in-room camera positions; an
# external bird's-eye view mostly shows a dense splat silhouette.  These views
# sit just inside the collision-normalized room bounds (about 6.8 x 6.8 m).
views = (
    ("interior_south", [0.0, -2.8, 1.55], [0.0, 0.55, 1.25]),
    ("interior_west", [-2.75, 0.0, 1.55], [0.5, 0.0, 1.25]),
    ("interior_north", [0.0, 2.75, 1.55], [0.0, -0.55, 1.25]),
    ("interior_east", [2.75, 0.0, 1.55], [-0.5, 0.0, 1.25]),
)
for name, eye, target in views:
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=target)
    pump(120)
    destination = OUTPUT / f"{name}.png"
    # Isaac Sim 6 returns a MultiAOVFileCapture here, not an asyncio Future.
    # It has no stable public done()/exception() API.  Pumping the Kit loop
    # until the requested file exists is version-safe and avoids a nested
    # asyncio event loop while the scene renderer writes the PNG.
    capture_viewport_to_file(get_active_viewport(), file_path=str(destination), is_hdr=False)
    deadline = time.monotonic() + 30.0
    while not destination.exists() and time.monotonic() < deadline:
        app.update()
    if not destination.exists():
        raise TimeoutError(f"timed out saving viewport capture: {destination}")
    print(f"Calibration screenshot: {destination}", flush=True)
app.close()
