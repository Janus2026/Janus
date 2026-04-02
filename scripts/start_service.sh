#!/bin/bash
set -e

# ============================================================
# Janus Service Startup Script
# Launches the janus_master_serving process (HTTP + RPC).
# ============================================================

# --- Paths (adjust to your environment) ---
SERVICE_BIN="${SERVICE_BIN:-./janus_service/build/janus_service/janus_master_serving}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/model}"

# --- Network ---
ETCD_ADDR="${ETCD_ADDR:-http://127.0.0.1:2379}"
HTTP_PORT="${HTTP_PORT:-8888}"
RPC_PORT="${RPC_PORT:-8889}"

# --- Scheduling ---
LOAD_BALANCE_POLICY="${LOAD_BALANCE_POLICY:-LST_IMH}"

# --- Optional config file paths (leave empty to use defaults) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_CONFIG_PATH="${MODELS_CONFIG_PATH:-${SCRIPT_DIR}/models_config.json}"
GP_STEADY_DATA_PATH="${GP_STEADY_DATA_PATH:-}"
MODEL_ALIAS_MAP_PATH="${MODEL_ALIAS_MAP_PATH:-}"

echo "Starting janus service..."
echo "  HTTP port: ${HTTP_PORT}, RPC port: ${RPC_PORT}"
echo "  etcd: ${ETCD_ADDR}"

GLOG_logtostderr=1 \
GLOG_v=1 \
ENABLE_DECODE_RESPONSE_TO_SERVICE=true \
${SERVICE_BIN} \
    --etcd_addr="${ETCD_ADDR}" \
    --http_server_port=${HTTP_PORT} \
    --rpc_server_port=${RPC_PORT} \
    --tokenizer_path="${TOKENIZER_PATH}" \
    --load_balance_policy="${LOAD_BALANCE_POLICY}" \
    --http_server_max_concurrency=0 \
    --rpc_server_max_concurrency=0 \
    --http_server_num_threads=128 \
    --rpc_server_num_threads=128 \
    --num_threads=128 \
    --gpu_hbm_per_gpu_gb=40.0 \
    --max_models_per_gpu_in_steady_pool=1 \
    ${MODELS_CONFIG_PATH:+--models_config_path="${MODELS_CONFIG_PATH}"} \
    ${GP_STEADY_DATA_PATH:+--gp_steady_data_path="${GP_STEADY_DATA_PATH}"} \
    ${MODEL_ALIAS_MAP_PATH:+--model_alias_map_path="${MODEL_ALIAS_MAP_PATH}"} \
    "$@" &

echo "Service started (PID: $!)"
