#!/bin/bash
set -e

# ============================================================
# Janus Engine Startup Script
# Launches a single janus engine instance on a specified device.
#
# Usage:
#   ./start_engine.sh --device_id 0
#   DEVICE_ID=1 MODEL_PATH=/path/to/model ./start_engine.sh
# ============================================================

# --- Paths (adjust to your environment) ---
ENGINE_BIN="${ENGINE_BIN:-./janus_engine/build/janus/core/server/janus}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
LOG_DIR="${LOG_DIR:-./logs}"

# --- Device ---
DEVICE_ID="${DEVICE_ID:-0}"
DEVICE_TYPE="${DEVICE_TYPE:-npu}"

# --- Network ---
ETCD_ADDR="${ETCD_ADDR:-127.0.0.1:2379}"
BASE_API_PORT="${BASE_API_PORT:-9010}"
BASE_LISTEN_PORT="${BASE_LISTEN_PORT:-21000}"
BASE_DISAGG_PORT="${BASE_DISAGG_PORT:-1111}"
XTENSOR_BASE_PORT="${XTENSOR_BASE_PORT:-47000}"

# --- Parse command line args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --device_id) DEVICE_ID="$2"; shift 2;;
    --model) MODEL_PATH="$2"; shift 2;;
    --device_type) DEVICE_TYPE="$2"; shift 2;;
    *) break;;
  esac
done

# --- Derived ports ---
API_PORT=$((BASE_API_PORT + DEVICE_ID))
LISTEN_PORT=$((BASE_LISTEN_PORT + DEVICE_ID))
DISAGG_PORT=$((BASE_DISAGG_PORT + DEVICE_ID))
XTENSOR_PORT=$((XTENSOR_BASE_PORT + DEVICE_ID))

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/device_${DEVICE_ID}.log"

echo "Starting janus engine on ${DEVICE_TYPE}:${DEVICE_ID} (port ${API_PORT})..."

${ENGINE_BIN} \
    --model "${MODEL_PATH}" \
    --devices="${DEVICE_TYPE}:${DEVICE_ID}" \
    --port ${API_PORT} \
    --master_node_addr="127.0.0.1:0" \
    --nnodes=1 \
    --max_memory_utilization=0.80 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm=false \
    --enable_disagg_pd=true \
    --disagg_pd_port=${DISAGG_PORT} \
    --instance_role=MIX \
    --disable_ttft_profiling \
    --enable_xtensor=true \
    --xtensor_master_node_addr="127.0.0.1:${XTENSOR_PORT}" \
    --etcd_addr="${ETCD_ADDR}" \
    --transfer_listen_port=${LISTEN_PORT} \
    --enable_service_routing=true \
    --kv_cache_transfer_type=Mooncake \
    --disable_prefilling_batch=true \
    --node_rank=0 \
    "$@" > "${LOG_FILE}" 2>&1 &

echo "Engine started on device ${DEVICE_ID} (PID: $!, log: ${LOG_FILE})"
