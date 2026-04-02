#!/bin/bash
set -e

# ============================================================
# Batch Janus Engine Startup Script
# Launches multiple engine instances across specified devices.
#
# Usage:
#   ./start_all_engines.sh                    # default: devices 0-1
#   NUM_DEVICES=8 ./start_all_engines.sh      # devices 0-7
#   DEVICES="0 2 4 6" ./start_all_engines.sh  # specific devices
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Configuration ---
NUM_DEVICES="${NUM_DEVICES:-2}"
DEVICES="${DEVICES:-$(seq 0 $((NUM_DEVICES - 1)))}"
SKIP_DEVICES="${SKIP_DEVICES:-}"
STARTUP_DELAY="${STARTUP_DELAY:-5}"

# Kill residual engine instances from previous runs
if pgrep -f "janus --model" > /dev/null 2>&1; then
  echo "Killing residual engine instances from previous run..."
  pkill -f "janus --model" 2>/dev/null || true
  sleep 2
  pkill -9 -f "janus --model" 2>/dev/null || true
  sleep 1
  echo "Cleanup done."
fi

for device_id in ${DEVICES}; do
  skip=false
  for s in ${SKIP_DEVICES}; do
    if [[ "$device_id" == "$s" ]]; then
      skip=true
      break
    fi
  done

  if $skip; then
    echo "Skipping device ${device_id}..."
    continue
  fi

  echo "Starting engine on device ${device_id}..."
  bash "${SCRIPT_DIR}/start_engine.sh" --device_id "${device_id}"
  sleep "${STARTUP_DELAY}"
done

echo "All engine instances started."
