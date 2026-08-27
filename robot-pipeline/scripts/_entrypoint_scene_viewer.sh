#!/bin/bash
set -e

# --- Install VNC stack + real Xorg so Vulkan can present ---
if ! dpkg -s xserver-xorg-core x11vnc novnc websockify >/dev/null 2>&1; then
    echo "[entrypoint] Installing VNC stack..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        xserver-xorg-core x11vnc novnc websockify >/dev/null 2>&1
    echo "[entrypoint] VNC stack installed"
fi

# --- NVIDIA Xorg config for headless (no physical display) GPU rendering ---
XORG_CONF="/etc/X11/xorg.conf"
if [ ! -f "$XORG_CONF" ]; then
    cat > "$XORG_CONF" << 'XORGEOF'
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
    BusID       "PCI:1:0:0"
    Option      "AllowEmptyInitialConfiguration" "True"
EndSection

Section "Screen"
    Identifier "Screen0"
    Device     "Card0"
    SubSection "Display"
        Depth     24
        Modes     "1920x1080"
    EndSubSection
EndSection
XORGEOF
    echo "[entrypoint] Wrote $XORG_CONF"
fi

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
if [ ! -S /tmp/.X11-unix/X1 ]; then
    echo "[entrypoint] ERROR: Xorg :1 failed to start, see /tmp/xorg.log"
    cat /tmp/xorg.log
    exit 1
fi

# --- VNC + noVNC (loopback only; access is via SSH tunnel) ---
x11vnc -display :1 -forever -shared -rfbauth /tmp/isaac-vnc.pass -rfbport 5901 -localhost >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 127.0.0.1:6080 localhost:5901 >/tmp/novnc.log 2>&1 &

export DISPLAY=:1
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

echo "[entrypoint] noVNC ready on http://127.0.0.1:6080 (loopback only)"
echo "[entrypoint] Launching living_room scene viewer..."

# Container sees only host GPU 1, which is visible GPU 0 here.
exec /isaac-sim/python.sh /workspace/view_living_room.py \
  --/renderer/activeGpu=0 \
  --/renderer/multiGpu/enabled=false \
  --/renderer/multiGpu/autoEnable=false \
  --/physics/cudaDevice=0
