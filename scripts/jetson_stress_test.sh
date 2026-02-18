#!/usr/bin/env bash
#
# 12-hour stress test for Jetson AGX Xavier
# Runs replay + UI with watchdog monitoring and auto-restart
#
# Usage: ./scripts/jetson_stress_test.sh [hours]
# Default: 12 hours
#

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$OPENPILOT_DIR"

DURATION_HOURS="${1:-12}"
DURATION_SECS=$((DURATION_HOURS * 3600))
LOG_DIR="/tmp/stress_test"
METRICS_LOG="$LOG_DIR/metrics.csv"
EVENTS_LOG="$LOG_DIR/events.log"
REPLAY_LOG="$LOG_DIR/replay.log"
UI_LOG="$LOG_DIR/ui.log"
CHECK_INTERVAL=30  # seconds between health checks

# Counters
REPLAY_RESTARTS=0
UI_RESTARTS=0
ERRORS=0

mkdir -p "$LOG_DIR"

export TERM="${TERM:-xterm-256color}"
export DISPLAY="${DISPLAY:-:0}"
export BIG="${BIG:-1}"
export SCALE="${SCALE:-0.889}"

source launch_env.sh
# Disable DPMS and screensaver to prevent X11 blanking (causes UI 1fps)
DISPLAY=:0 xset s off 2>/dev/null
DISPLAY=:0 xset -dpms 2>/dev/null
DISPLAY=:0 xset s noblank 2>/dev/null

log_event() {
  local ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] $1" | tee -a "$EVENTS_LOG"
}

cleanup() {
  log_event "SHUTDOWN: Stopping stress test"
  [ -n "${REPLAY_PID:-}" ] && kill "$REPLAY_PID" 2>/dev/null
  [ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null
  sleep 1
  [ -n "${REPLAY_PID:-}" ] && kill -9 "$REPLAY_PID" 2>/dev/null
  [ -n "${UI_PID:-}" ] && kill -9 "$UI_PID" 2>/dev/null
  wait 2>/dev/null

  local elapsed=$(( $(date +%s) - START_TIME ))
  local hours=$((elapsed / 3600))
  local mins=$(( (elapsed % 3600) / 60 ))
  log_event "SUMMARY: Ran ${hours}h${mins}m | Replay restarts: $REPLAY_RESTARTS | UI restarts: $UI_RESTARTS | Errors: $ERRORS"
  echo ""
  echo "=== STRESS TEST SUMMARY ==="
  echo "Duration: ${hours}h${mins}m"
  echo "Replay restarts: $REPLAY_RESTARTS"
  echo "UI restarts: $UI_RESTARTS"
  echo "Total errors: $ERRORS"
  echo "Logs: $LOG_DIR/"
  echo "==========================="
}
trap cleanup EXIT

start_replay() {
  rm -f /dev/shm/msgq_* /tmp/visionipc_* 2>/dev/null
  ./tools/replay/replay --demo > "$REPLAY_LOG" 2>&1 &
  REPLAY_PID=$!
  log_event "REPLAY: Started PID $REPLAY_PID"

  # Wait for VisionIPC
  for i in $(seq 1 30); do
    [ -S /tmp/visionipc_camerad ] && break
    sleep 0.5
  done

  if [ ! -S /tmp/visionipc_camerad ]; then
    log_event "ERROR: VisionIPC not ready after 15s"
    ERRORS=$((ERRORS + 1))
    return 1
  fi
  log_event "REPLAY: VisionIPC ready"
  return 0
}

start_ui() {
  .venv/bin/python3 -m selfdrive.ui.ui > "$UI_LOG" 2>&1 &
  UI_PID=$!
  log_event "UI: Started PID $UI_PID"
  sleep 3
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    log_event "ERROR: UI died immediately"
    ERRORS=$((ERRORS + 1))
    return 1
  fi
  return 0
}

collect_metrics() {
  local ts=$(date '+%H:%M:%S')
  local gpu_load=$(cat /sys/devices/gpu.0/load 2>/dev/null || echo -1)
  local cpu_temp=$(echo "scale=1; $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)/1000" | bc)
  local gpu_temp=$(echo "scale=1; $(cat /sys/class/thermal/thermal_zone2/temp 2>/dev/null || echo 0)/1000" | bc)
  local mem_used=$(free -m | awk '/Mem:/{print $3}')
  local ui_fps=$(tail -1 "$UI_LOG" 2>/dev/null | grep -oP 'FPS dropped below 60: \K\d+' || echo "60+")
  local replay_cpu=$(ps -p "$REPLAY_PID" -o %cpu --no-headers 2>/dev/null | tr -d ' ' || echo 0)
  local ui_cpu=$(ps -p "$UI_PID" -o %cpu --no-headers 2>/dev/null | tr -d ' ' || echo 0)

  echo "$ts,$gpu_load,$cpu_temp,$gpu_temp,$mem_used,$ui_fps,$replay_cpu,$ui_cpu,$REPLAY_RESTARTS,$UI_RESTARTS,$ERRORS" >> "$METRICS_LOG"
}

check_health() {
  local healthy=true

  # Check replay alive
  if ! kill -0 "$REPLAY_PID" 2>/dev/null; then
    log_event "WATCHDOG: Replay died (PID $REPLAY_PID), restarting..."
    REPLAY_RESTARTS=$((REPLAY_RESTARTS + 1))
    ERRORS=$((ERRORS + 1))
    start_replay
    sleep 3
    # UI needs restart too since VisionIPC was reset
    kill "$UI_PID" 2>/dev/null; wait "$UI_PID" 2>/dev/null
    start_ui
    healthy=false
  fi

  # Check UI alive
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    log_event "WATCHDOG: UI died (PID $UI_PID), restarting..."
    UI_RESTARTS=$((UI_RESTARTS + 1))
    ERRORS=$((ERRORS + 1))
    start_ui
    healthy=false
  fi

  # Check UI stuck at 1 FPS for too long (10+ consecutive)
  if [ -f "$UI_LOG" ]; then
    local stuck_count=$(tail -10 "$UI_LOG" | grep -c 'FPS dropped below 60: 1$' || true)
    if [ "$stuck_count" -ge 10 ]; then
      log_event "WATCHDOG: UI stuck at 1 FPS, restarting..."
      kill "$UI_PID" 2>/dev/null; wait "$UI_PID" 2>/dev/null
      UI_RESTARTS=$((UI_RESTARTS + 1))
      ERRORS=$((ERRORS + 1))
      start_ui
      healthy=false
    fi
  fi

  # Check thermal throttling
  local cpu_temp_raw=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
  if [ "$cpu_temp_raw" -gt 85000 ]; then
    log_event "WARNING: CPU temp $(echo "scale=1; $cpu_temp_raw/1000" | bc)°C - thermal throttling risk"
    ERRORS=$((ERRORS + 1))
  fi

  $healthy && return 0 || return 1
}

# === MAIN ===
echo "=== Jetson Stress Test ==="
echo "Duration: ${DURATION_HOURS}h"
echo "Logs: $LOG_DIR/"
echo ""

# CSV header
echo "time,gpu_load,cpu_temp,gpu_temp,mem_mb,ui_fps,replay_cpu,ui_cpu,replay_restarts,ui_restarts,errors" > "$METRICS_LOG"

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_SECS))

log_event "START: ${DURATION_HOURS}h stress test"

# Start processes
start_replay || exit 1
sleep 2
start_ui || exit 1

log_event "RUNNING: All processes started"

# Main monitoring loop
LAST_REPORT=0
while [ "$(date +%s)" -lt "$END_TIME" ]; do
  sleep "$CHECK_INTERVAL"

  check_health
  collect_metrics

  # Hourly status report
  elapsed=$(( $(date +%s) - START_TIME ))
  local hours_elapsed=$((elapsed / 3600))
  if [ "$hours_elapsed" -gt "$LAST_REPORT" ]; then
    LAST_REPORT=$hours_elapsed
    remaining=$(( (END_TIME - $(date +%s)) / 3600 ))
    log_event "STATUS: ${hours_elapsed}h elapsed, ${remaining}h remaining | Restarts: R=$REPLAY_RESTARTS UI=$UI_RESTARTS | Errors: $ERRORS"
  fi
done

log_event "COMPLETE: Stress test finished successfully"
