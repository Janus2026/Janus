![Janus Logo](pics/janus_logo.jpg)

# Janus

**Multi-LLM Serving at Production Scale**

Janus is a Service–Engine co-designed multi-model LLM serving system built around a **fast-and-slow-thinking** architecture. It simultaneously addresses three challenges of production MaaS (Model-as-a-Service) clusters: bursty unpredictable traffic, power-law application popularity, and heterogeneous yet complementary resource demands.

Deployed on a 768-device production cluster with 62 applications and 45.6M real-traffic requests, Janus maintains **0.97–1.0 SLO attainment** while saving **24% device-hours** over the best baseline.

## Architecture

Janus has two layers forming a closed feedback loop:

### Service Layer (`janus_service/`)

The centralized control plane coordinates all engine instances through three components:

- **Performance Oracle**: Gaussian-process based surrogate models that predict per-model resource consumption (Capacity GP) and SLO attainment (Elasticity GP) from workload features.
- **Model Scheduler**: Dual-timescale scheduling aligned with the fast-and-slow-thinking paradigm:
  - *Slow thinking* (every 30s): Vector bin-packing for Steady Pool model colocation, exploiting multi-dimensional resource complementarity (HBM, compute, bandwidth).
  - *Fast thinking* (every 0.5s): SLO-driven autoscaling for the Elastic Pool, enabling sub-second burst absorption.
- **Request Scheduler (LST-IMH)**: A new algorithm for parallel-machine deadline scheduling that maximizes on-time completions with a **2-approximation guarantee** in O(mn log n) time.

### Engine Layer (`janus_engine/`)

The distributed data plane restructures the inference runtime for elastic multi-model colocation:

- **xTensor**: A virtual HBM manager that decouples physical pages from any particular model, organizing all device memory into a unified page pool with bidirectional allocation (weights growing upward, KV caches/activations growing downward).
- **Three-state Model Lifecycle** (active/warm/cold): Decouples model lifecycle from instance lifecycle, supporting two reactivation paths:
  - *H2D wakeup*: Recovers warm models from host pinned memory.
  - *D2D fork*: Sub-second burst scale-out by pulling weights directly from a peer device (388–783ms for 8B–32B models).
- **Dual-pool Resource Management**:
  - *Steady Pool*: Colocates tail applications with complementary resource profiles via multi-stream parallel execution.
  - *Elastic Pool*: Provides dedicated, rapidly scalable instances for head applications with P/D disaggregation.

## Repository Structure

```
janus/
├── janus_engine/    # Inference engine (Python package with C++ backend)
├── janus_service/   # Distributed serving layer (C++ service)
├── third_party/     # Shared third-party dependencies for both engine and service
├── scripts/         # Deployment scripts
└── pics/            # Project assets (logo, etc.)
```

## Prerequisites

- Linux (x86_64 or aarch64)
- Python >= 3.10
- CMake >= 3.26
- Rust toolchain
- PyTorch
- [vcpkg](https://github.com/microsoft/vcpkg) (set `VCPKG_ROOT` environment variable)
- [etcd](https://github.com/etcd-io/etcd/releases) (for service discovery)
- Device-specific SDK (e.g., CUDA toolkit, Ascend toolkit)

## Build

### Build janus_engine

```bash
cd janus_engine
python setup.py build --device <cuda|npu|mlu|ilu>
pip install -e .
```

### Build janus_service

```bash
cd janus_service
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Start

The system consists of three components that must be started in order:

### 1. Start etcd

```bash
ETCD_PORT=2379 bash scripts/start_etcd.sh
```

### 2. Start janus_service

```bash
SERVICE_BIN=./janus_service/build/janus_service/janus_master_serving \
ETCD_ADDR="http://<host>:2379" \
TOKENIZER_PATH=/path/to/model \
HTTP_PORT=8888 \
RPC_PORT=8889 \
bash scripts/start_service.sh
```

### 3. Start engine instances

Launch a single engine:

```bash
ENGINE_BIN=./janus_engine/build/janus/core/server/janus \
ETCD_ADDR="<host>:2379" \
MODEL_PATH=/path/to/model \
DEVICE_TYPE=npu \
bash scripts/start_engine.sh --device_id 0
```

Or launch multiple engines at once:

```bash
ENGINE_BIN=./janus_engine/build/janus/core/server/janus \
ETCD_ADDR="<host>:2379" \
MODEL_PATH=/path/to/model \
DEVICE_TYPE=npu \
NUM_DEVICES=2 \
bash scripts/start_all_engines.sh
```

### 4. Send a request

```bash
curl http://<host>:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model_name>","prompt":"Hello","max_tokens":32}'
```

### Standalone Engine (single-model, no service layer)

For quick testing without etcd or janus_service:

```bash
./janus_engine/build/janus/core/server/janus \
  --model /path/to/model \
  --devices=npu:0 \
  --port 8080

curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model_dir_name>","prompt":"Hello","max_tokens":32}'
```

The `model` field must match the directory name of `--model` (or set `--model_id` to override).

## Key Results

| Metric | Janus | Best Baseline |
|--------|-------|---------------|
| SLO Attainment | 0.97–1.0 | 0.80–0.92 (Prism) |
| Device-hours/day | 13,440 | 17,856 (ServerlessLLM) |
| D2D Fork (8B, TP=2) | 408 ms | — |
| D2D Fork (32B, TP=2) | 783 ms | — |
| D2D Fork (671B, TP=16) | 742 ms | — |

## License

Apache 2.0
