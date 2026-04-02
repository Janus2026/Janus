#pragma once

#include <gflags/gflags.h>

DECLARE_string(server_host);

DECLARE_int32(http_server_port);

DECLARE_int32(http_server_idle_timeout_s);

DECLARE_int32(http_server_num_threads);

DECLARE_int32(http_server_max_concurrency);

DECLARE_int32(rpc_server_port);

DECLARE_int32(rpc_server_idle_timeout_s);

DECLARE_int32(rpc_server_num_threads);

DECLARE_int32(rpc_server_max_concurrency);

DECLARE_uint32(murmur_hash3_seed);

DECLARE_int32(timeout_ms);

DECLARE_string(listen_addr);

DECLARE_int32(port);

DECLARE_int32(idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_int32(max_concurrency);

DECLARE_string(etcd_addr);

DECLARE_string(load_balance_policy);

DECLARE_int32(detect_disconnected_instance_interval);

DECLARE_int32(block_size);

DECLARE_string(tokenizer_path);

DECLARE_bool(enable_request_trace);

DECLARE_int32(target_ttft);

DECLARE_int32(target_tpot);

DECLARE_int32(default_ttft_slo_ms);

DECLARE_int32(lst_imh_pre_pull_ms);

DECLARE_double(slo_penalty_factor);

DECLARE_int32(max_slo_expansions);

DECLARE_bool(enable_prefill_only_mode);

DECLARE_double(gpu_hbm_per_gpu_gb);

DECLARE_double(gpu_compute_sm_per_gpu);

DECLARE_bool(enable_mix_pd);

DECLARE_double(gpu_bandwidth_per_gpu);

DECLARE_string(model_alias_map_path);

DECLARE_string(gp_steady_data_path);

DECLARE_string(gp_dynamic_data_path);

DECLARE_bool(disable_steady_pool);

DECLARE_bool(disable_elastic_pool);

DECLARE_int32(elastic_instance_count);

DECLARE_string(models_config_path);

DECLARE_string(pool_memory_log_path);

DECLARE_string(check_memory_log_path);

DECLARE_double(overlap_abundance_gb);

DECLARE_int32(max_models_per_gpu_in_steady_pool);

DECLARE_int32(tensor_parallel_size);

DECLARE_string(initial_model_id);
