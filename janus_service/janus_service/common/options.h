#pragma once

#include <string>

#include "common/macros.h"

namespace janus_service {

class Options {
 public:
  Options() = default;
  ~Options() = default;

  // http server options
  PROPERTY(std::string, server_host);

  PROPERTY(int32_t, http_port) = 9998;

  PROPERTY(int32_t, http_idle_timeout_s) = -1;

  PROPERTY(int32_t, http_num_threads) = 32;

  PROPERTY(int32_t, http_max_concurrency) = 0;

  // rpc server options
  PROPERTY(int32_t, rpc_port) = 9999;

  PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  PROPERTY(int32_t, rpc_num_threads) = 32;

  PROPERTY(int32_t, rpc_max_concurrency) = 0;

  PROPERTY(int32_t, num_threads) = 32;

  PROPERTY(int32_t, max_concurrency) = 32;

  PROPERTY(int32_t, timeout_ms) = 32;

  // instance manager options
  PROPERTY(std::string, etcd_addr);

  PROPERTY(int32_t, detect_disconnected_instance_interval) = 15;

  // scheduler options
  PROPERTY(std::string, load_balance_policy);

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(uint32_t, murmur_hash3_seed) = 1024;

  PROPERTY(std::string, service_name);

  // tokenizer options
  PROPERTY(std::string, tokenizer_path);

  // trace options
  PROPERTY(bool, enable_request_trace) = false;

  // default TTFT SLO in milliseconds (used when request doesn't specify one)
  PROPERTY(int32_t, default_ttft_slo_ms) = 30000;

  // LST-IMH pre-pull threshold in milliseconds
  PROPERTY(int32_t, lst_imh_pre_pull_ms) = 0;

  // MixPD instance tagging
  PROPERTY(bool, enable_mix_pd) = false;

  // SLO penalty factor: multiply SLO by this when deadline expires (re-enqueue)
  PROPERTY(double, slo_penalty_factor) = 2.0;

  // Max number of SLO expansions before truly discarding
  PROPERTY(int32_t, max_slo_expansions) = 3;

  // Disable steady pool: all models go to elastic pool with PD disaggregation
  PROPERTY(bool, disable_steady_pool) = false;

  // Disable elastic pool: all models stay in steady pool, no dynamic scaling
  PROPERTY(bool, disable_elastic_pool) = false;

  // Fixed instance count per model when disable_steady_pool is true.
  // 0 = use all available instances; >= 2 = fixed count.
  PROPERTY(int32_t, elastic_instance_count) = 0;

  // Path to JSON file that maps service names to model paths.
  // When non-empty, MODELS is loaded from this file at startup.
  PROPERTY(std::string, models_config_path);

  // Number of GPUs used for tensor parallelism per instance.
  PROPERTY(int32_t, tensor_parallel_size) = 1;
};

}  // namespace janus_service