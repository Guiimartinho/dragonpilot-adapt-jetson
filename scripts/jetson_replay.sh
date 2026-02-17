#!/usr/bin/env bash
#
# Launch replay + UI on Jetson AGX Xavier for testing.
# Usage: ./scripts/jetson_replay.sh [replay_args...]
#
# Examples:
#   ./scripts/jetson_replay.sh --demo              # Use demo route
#   ./scripts/jetson_replay.sh -d /data/routes      # Use local route
#   ./scripts/jetson_replay.sh --demo --dcam --ecam  # All cameras
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$OPENPILOT_DIR"

# Ensure TERM is set for ncurses (replay consoleui)
export TERM="${TERM:-xterm-256color}"

# Display settings for UI
export DISPLAY="${DISPLAY:-:0}"
export BIG="${BIG:-1}"
export SCALE="${SCALE:-0.889}"

# Source openpilot environment
source launch_env.sh

# Cleanup function
cleanup() {
  echo "Stopping processes..."
  [ -n "${REPLAY_PID:-}" ] && kill "$REPLAY_PID" 2>/dev/null || true
  [ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null || true
  [ -n "${VNC_PID:-}" ] && kill "$VNC_PID" 2>/dev/null || true
  wait 2>/dev/null
  echo "Done."
}
trap cleanup EXIT

# Start VNC if not already running
# Optimized flags: -wait 50 -defer 30 caps polling to ~20fps (matches camera rate),
# -noxdamage avoids expensive X damage tracking, -nocursor/-norepeat reduce overhead.
# Result: ~2% CPU vs ~37% with defaults.
if ! pgrep -f "x11vnc.*5900" > /dev/null 2>&1; then
  echo "Starting VNC server on port 5900 (low-CPU mode)..."
  x11vnc -display :0 -clip 1920x960+0+60 -scale 0.5 \
         -rfbport 5900 -forever -shared -nopw \
         -wait 50 -defer 30 \
         -noxdamage -nocursor -norepeat \
         -bg -o /tmp/x11vnc.log 2>/dev/null
  VNC_PID=""  # bg mode, managed by x11vnc
  sleep 1
else
  echo "VNC already running."
  VNC_PID=""
fi

# Clear stale msgq/visionipc state
rm -f /dev/shm/msgq_* /tmp/visionipc_* 2>/dev/null || true

# Start replay
echo "Starting replay with args: ${*:-<none>}"
./tools/replay/replay "$@" > /tmp/replay.log 2>&1 &
REPLAY_PID=$!
echo "Replay PID: $REPLAY_PID"

# Wait for VisionIPC to be ready
echo "Waiting for VisionIPC..."
for i in $(seq 1 30); do
  if [ -S /tmp/visionipc_camerad ]; then
    echo "VisionIPC ready."
    break
  fi
  sleep 0.5
done

if [ ! -S /tmp/visionipc_camerad ]; then
  echo "WARNING: VisionIPC not detected after 15s. Check /tmp/replay.log"
fi

# Start UI
echo "Starting UI..."
.venv/bin/python3 -m selfdrive.ui.ui > /tmp/ui.log 2>&1 &
UI_PID=$!
echo "UI PID: $UI_PID"

echo ""
echo "=== Jetson Replay Running ==="
echo "  Replay: PID $REPLAY_PID (log: /tmp/replay.log)"
echo "  UI:     PID $UI_PID (log: /tmp/ui.log)"
echo "  VNC:    port 5900"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Wait for either process to exit
wait -n "$REPLAY_PID" "$UI_PID" 2>/dev/null || true
echo "A process exited. Cleaning up..."
