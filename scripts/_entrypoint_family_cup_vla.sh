#!/bin/bash
set -e

# --- Ensure real Xorg (not Xvfb) is installed so Vulkan can present ---
if ! dpkg -s xserver-xorg-core >/dev/null 2>&1; then
    echo "[entrypoint] Installing xserver-xorg-core..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xserver-xorg-core
fi

# --- NVIDIA Xorg config for headless (no physical display) GPU rendering ---
XORG_CONF="/etc/X11/xorg.conf"
GPU_PCI_ID="$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | head -n 1)"
GPU_BUS_HEX="$(printf '%s' "$GPU_PCI_ID" | cut -d: -f2)"
GPU_BUS_DEC="$((16#$GPU_BUS_HEX))"
cat > "$XORG_CONF" << XORGEOF
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0" 0 0
EndSection

Section "Files"
    ModulePath "/usr/lib/x86_64-linux-gnu/nvidia/xorg"
    ModulePath "/usr/lib/xorg/modules"
EndSection

Section "Device"
    Identifier  "Card0"
    Driver      "nvidia"
    BusID       "PCI:${GPU_BUS_DEC}:0:0"
    Option      "AllowEmptyInitialConfiguration" "True"
    Option      "UseDisplayDevice" "None"
EndSection

Section "Screen"
    Identifier "Screen0"
    Device     "Card0"
    DefaultDepth 24
    SubSection "Display"
        Depth     24
        Virtual   1920 1080
    EndSubSection
EndSection
XORGEOF
echo "[entrypoint] Wrote $XORG_CONF for $GPU_PCI_ID"

# --- Clean stale X lock files from unclean shutdowns ---
rm -f /tmp/.X0-lock /tmp/.X1-lock /tmp/.X11-unix/X0 /tmp/.X11-unix/X1

# --- VNC password ---
x11vnc -storepasswd g1dview /tmp/isaac-vnc.pass

# --- Start real Xorg server (with NVIDIA driver, supports Vulkan WSI) ---
echo "[entrypoint] Starting Xorg :1 with NVIDIA driver..."
Xorg :1 -ac -noreset +extension GLX +extension RENDER +extension RANDR \
    -config "$XORG_CONF" \
    >/tmp/xorg.log 2>&1 &
XORG_PID=$!

# Wait for Xorg to be ready
for i in $(seq 1 20); do
    if [ -S /tmp/.X11-unix/X1 ]; then
        echo "[entrypoint] Xorg :1 ready after ${i}s"
        break
    fi
    sleep 1
done

# --- VNC + noVNC ---
# noVNC otherwise defaults to "off" and shows the 1920x1080 remote desktop as
# a small, unscaled rectangle.  Make "scale" the default while still allowing
# the user to change it from the noVNC settings panel.
sed -i "s/UI.initSetting('resize', 'off')/UI.initSetting('resize', 'scale')/" \
    /usr/share/novnc/app/ui.js
# Prefer lower-latency defaults for an animated OpenGL viewport.  XDamage can
# miss Vulkan-presented frames and the automatic link estimator previously
# classified the SSH/websocket tunnel as only 54 KB/s, yielding ~2 updates/s.
sed -i "s/UI.initSetting('quality', 6)/UI.initSetting('quality', 3)/" \
    /usr/share/novnc/app/ui.js
sed -i "s/UI.initSetting('compression', 2)/UI.initSetting('compression', 1)/" \
    /usr/share/novnc/app/ui.js
# Give the low-latency page a unique module URL so browsers cannot reuse an
# older cached ui.js with quality=6.  The X desktop remains 1920x1080; only
# the VNC transport is encoded at 1280x720 and then scaled to the browser.
cp /usr/share/novnc/vnc.html /usr/share/novnc/fast2.html
sed -i 's#src="app/ui.js"#src="app/ui.js?profile=family-fast-v2-20260810"#' \
    /usr/share/novnc/fast2.html
x11vnc -display :1 -geometry 1920x1080 -scale 2/3 -forever -shared \
    -rfbauth /tmp/isaac-vnc.pass -rfbport 5901 \
    -noxdamage -snapfb -wait 5 -defer 5 -extra_fbur 2 -threads \
    -speeds 100,5000,20 \
    >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 6080 localhost:5901 >/tmp/novnc.log 2>&1 &

# x11vnc is retained for mouse/keyboard control.  This MJPEG path is the
# smooth observation view: it sends a continuous 15 FPS stream instead of
# round-tripping one framebuffer request per VNC update.
if command -v ffmpeg >/dev/null 2>&1; then
    python3 /workspace/scripts/serve_x11_mjpeg.py \
        --host 0.0.0.0 --port 6082 --display :1.0 \
        --capture-size 1920x900 --width 960 --fps 15 --quality 12 \
        >/tmp/isaac-mjpeg.log 2>&1 &
else
    echo "[entrypoint] ffmpeg unavailable; skipping optional 6082 MJPEG relay"
fi

export DISPLAY=:1

# Pin OpenVLA sidecar to a different GPU than Isaac Sim (GPU 0)
# to avoid CUDA OOM on the 24 GB RTX 4090
export G1D_SIDECAR_CUDA_DEVICE=1

# Older reviewed map manifests store their original in-container prefix.
# Preserve that artifact identity with a container-local compatibility link;
# the host project remains /data/MaMingyi/robot-vln mounted at /workspace.
if [ ! -e /root/autodl-tmp ]; then
    ln -s /workspace /root/autodl-tmp
fi

# Materialize the runtime-demo catalogs (formal LingBot artifacts plus
# occupancy-validated extra places and reviewed multi-cup props) before Isaac
# parses its CLI.  The embedded 6013 controller must consume these files too;
# generating them only in the standalone Dashboard leaves it on the formal
# single-cup catalog.
PYTHONPATH=/workspace/lingbot_semantic_nav/src \
    /workspace/.conda/envs/vln/bin/python -c \
    'from argparse import Namespace; from pathlib import Path; from scripts.serve_family_home_dashboard import FamilyHomeDashboardSession; FamilyHomeDashboardSession(Namespace(artifacts=Path("/workspace/outputs/family_home_vln"), output=Path("/workspace/outputs/family_home_vln"), intent_provider="deepseek", no_rule_fallback=False))'

# A read-only browser viewer for inspecting Expert RGB/action trajectories.
python3 /workspace/scripts/serve_expert_trajectory_viewer.py \
    --root /workspace/outputs --host 0.0.0.0 --port 6014 \
    >/tmp/expert-trajectory-viewer.log 2>&1 &

# --- Launch Isaac Sim with expert pick VLA pipeline ---
# Interactive mode: control page on port 6013, live viewport via VNC 6080.
# Dual-agent family task with OpenVLA inference + MaChuanhao DLS-IK expert pick.
exec /isaac-sim/python.sh run_g1d_simple_room_vln.py \
  --scene-profile family-home \
  --output-dir /workspace/outputs/family_home_vln \
  --places /workspace/outputs/family_home_vln/places_web_demo.json \
  --objects /workspace/outputs/family_home_vln/objects_web_demo.json \
  --dual-agent \
  --family-task \
  --openvla \
  --openvla-python /workspace/.conda/envs/openvla-oft/bin/python \
  --openvla-model /workspace/checkpoints/openvla-oft-libero-combined \
  --openvla-adapter /workspace/checkpoints/openvla-oft-g1d-v14/g1d-family-home-oft/lora_adapter \
  --openvla-action-head /workspace/checkpoints/openvla-oft-g1d-v14/g1d-family-home-oft/action_head--latest_checkpoint.pt \
  --openvla-dataset-statistics /workspace/checkpoints/openvla-oft-g1d-v14/g1d-family-home-oft/dataset_statistics.json \
  --openvla-unnorm-key g1d_family_home_pick \
  --openvla-instruction 'pick up the coffee cup' \
  --expert-pick \
  --record-expert-demo \
  --assisted-motion-scale 3.0 \
  --resolution 960x720 \
  --live-search-frames 3 \
  --interactive-port 6013 \
  --interactive-host 0.0.0.0
