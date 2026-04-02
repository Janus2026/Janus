#!/bin/bash
set -e

ETCD_BIN="${ETCD_BIN:-etcd}"
ETCD_PORT=${ETCD_PORT:-2379}

echo "Starting etcd on port ${ETCD_PORT}..."
${ETCD_BIN} \
  --listen-client-urls "http://0.0.0.0:${ETCD_PORT}" \
  --advertise-client-urls "http://0.0.0.0:${ETCD_PORT}"
