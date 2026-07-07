![Janus Logo](pics/janus_logo.jpg)

# Janus

**Multi-LLM Serving at Production Scale**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/SOSP-2026-b31b1b.svg)](#citation)
[![Artifact: Zenodo](https://img.shields.io/badge/Artifact-Zenodo_TBD-orange.svg)](#citation)

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

## Configuration

The launch scripts are driven by environment variables. The most common ones:

| Variable | Component | Description |
|----------|-----------|-------------|
| `ETCD_ADDR` | service, engine | Address of the etcd endpoint used for service discovery (e.g. `http://<host>:2379`). |
| `ETCD_PORT` | etcd | Port for the local etcd instance (default `2379`). |
| `SERVICE_BIN` | service | Path to the `janus_master_serving` binary. |
| `TOKENIZER_PATH` | service | Path to the tokenizer/model used by the service layer. |
| `HTTP_PORT` | service | HTTP port exposed for the OpenAI-compatible API (default `8888`). |
| `RPC_PORT` | service | Internal RPC port between service and engines (default `8889`). |
| `ENGINE_BIN` | engine | Path to the `janus` engine binary. |
| `MODEL_PATH` | engine | Path to the model weights served by the engine. |
| `DEVICE_TYPE` | engine | Accelerator backend: `cuda`, `npu`, `mlu`, or `ilu`. |
| `NUM_DEVICES` | engine | Number of devices to launch when using `start_all_engines.sh`. |

Standalone-engine flags (see `--help` for the full list): `--model`, `--model_id`, `--devices` (e.g. `npu:0`), `--port`.

## Hardware & Evaluation Environment

The results reported in the paper were obtained on a **CloudMatrix 384 supernode**:

- **48 compute nodes**, each with **8 Ascend 910C** accelerators.
- Each accelerator exposes **2 dies**, and each die acts as an independent device with **376 TFLOPS** and **64 GB HBM** — **768 devices** in total.
- **3 TB DRAM** per node, unified-addressed with HBM via the Unified Bus.
- Interconnect (measured / spec): D2D **163.6 / 196 GB/s**, H2D **59.0 / 140 GB/s**, per-device HBM bandwidth **1490 / 1600 GB/s**.
- Component-level experiments use **1×8** or **2×8** nodes.

Models used in evaluation span Qwen3-0.6B to Qwen3-32B, Qwen2.5-3B/14B, Qwen2-7B, and DeepSeek-V3.

> **Note (for artifact reviewers):** The full end-to-end evaluation requires a large multi-node Ascend cluster. A single 8-device node (`1×8`) is sufficient to reproduce the component-level experiments. Minimal single-device functional testing can be done via the [Standalone Engine](#standalone-engine-single-model-no-service-layer) flow above. `<Minimal reviewer environment / access instructions: TBD>`

## Key Results

| Metric | Janus | Best Baseline |
|--------|-------|---------------|
| SLO Attainment | 0.97–1.0 | 0.80–0.92 (Prism) |
| Device-hours/day | 13,440 | 17,856 (ServerlessLLM) |
| D2D Fork (8B, TP=2) | 408 ms | — |
| D2D Fork (32B, TP=2) | 783 ms | — |
| D2D Fork (671B, TP=16) | 742 ms | — |

## Acknowledgments

Janus builds on the [xLLM](https://github.com/xLLM-AI/xllm) project:

- The **Janus engine** (`janus_engine/`) is developed on top of [xLLM](https://github.com/xLLM-AI/xllm), a high-performance LLM inference engine optimized for diverse AI accelerators.
- The **Janus service** (`janus_service/`) is developed on top of [xLLM-service](https://github.com/xLLM-AI/xllm-service), a flexible serving framework for efficient and fault-tolerant clustered LLM inference.

We thank the xLLM community for their open-source work.

## Citation

Janus was published at the *ACM Symposium on Operating Systems Principles (SOSP)* 2026.
If you use Janus in your research, please cite our paper:

> *Multi-LLM Serving at Production Scale.* In Proceedings of the 32nd ACM Symposium
> on Operating Systems Principles (SOSP '26), 2026.

BibTeX:

```bibtex
@inproceedings{janus2026,
  title     = {Multi-LLM Serving at Production Scale},
  author    = {Zhou, T. and Wang, Y. and Zhou, Y. and Liu, Z. and Wang, Z. and
               Peng, Y. and Wang, Y. and Zhang, Y. and Yin, J. and Tian, K. and
               Fu, F. and Liu, T. and Peng, T. and Yang, T. and Cui, B. and
               Miao, X. and Zhang, K.},
  booktitle = {Proceedings of the 32nd ACM Symposium on Operating Systems Principles (SOSP '26)},
  year      = {2026},
  note      = {To appear}
}
```

> **Note:** The camera-ready paper is not yet published; the full bibliographic
> details above will be finalized once the proceedings are released. An archival
> snapshot of this artifact will also be deposited on Zenodo.

## License

Janus is released under the [Apache License 2.0](LICENSE). See the [`LICENSE`](LICENSE)
file for the full text.
