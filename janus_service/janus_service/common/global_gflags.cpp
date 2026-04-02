#include "common/global_gflags.h"

#include "brpc/reloadable_flags.h"

DEFINE_string(server_host,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(http_server_port, 8888, "Port for janus http service to listen on");

DEFINE_int32(http_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(http_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(http_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(rpc_server_port, 8889, "Port for janus rpc service to listen on");

DEFINE_int32(rpc_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(rpc_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(rpc_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_string(etcd_addr,
              "0.0.0.0:2379",
              "etcd adderss for save instance meta info");

DEFINE_uint32(murmur_hash3_seed, 1024, "default Murmur Hash seed");

DEFINE_int32(port, 8888, "Port for janus service to listen on");

DEFINE_int32(num_threads, 32, "Number of threads to process requests");

DEFINE_int32(max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(timeout_ms,
             -1,
             "Max duration of bRPC Channel. -1 means wait indefinitely.");

DEFINE_string(listen_addr,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_string(load_balance_policy,
              "RR",
              "Disaggregated prefill-decode policy.");

DEFINE_int32(detect_disconnected_instance_interval,
             15,
             "The interval that server detect the disconnected instance.");

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_string(tokenizer_path, "", "tokenizer config path.");

DEFINE_bool(enable_request_trace, false, "Whether to enable request trace");

DEFINE_int32(default_ttft_slo_ms,
             30000,
             "Default TTFT SLO in milliseconds when request doesn't specify one.");

DEFINE_int32(lst_imh_pre_pull_ms,
             0,
             "LST-IMH pre-pull threshold in ms. Dispatch coordinator pulls "
             "requests when instance estimated remaining time <= this value. "
             "0 means wait until instance is completely idle (default).");

DEFINE_double(slo_penalty_factor,
              3.0,
              "When a request's TTFT SLO expires in LST-IMH, multiply its SLO "
              "by this factor and re-enqueue with lower priority instead of "
              "discarding.");

DEFINE_int32(max_slo_expansions,
             2,
             "Maximum number of times a request's SLO can be expanded by "
             "slo_penalty_factor before it is truly discarded.");

DEFINE_int32(target_ttft,
             1000,
             "Target Time to First Token (TTFT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_ttft, brpc::NonNegativeInteger);

DEFINE_int32(target_tpot,
             50,
             "Target Time Per Output Token (TPOT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_tpot, brpc::NonNegativeInteger);

DEFINE_bool(enable_prefill_only_mode,
            false,
            "When true, all forwarded requests have max_tokens overridden to 1 "
            "(prefill-only elastic pool mode).");

DEFINE_double(gpu_hbm_per_gpu_gb,
              58.0,
              "HBM capacity per GPU in GB, used for auto-scaling resource model.");

DEFINE_double(gpu_compute_sm_per_gpu,
              1.0,
              "Compute SM units per GPU, used for auto-scaling resource model.");

DEFINE_bool(enable_mix_pd,
            false,
            "Enable MixPD instance tagging: each MIX instance gets a role tag "
            "(PREFILL/DECODE/NORMAL) and LST-IMH routes accordingly.");

DEFINE_double(gpu_bandwidth_per_gpu,
              1.0,
              "HBM-to-SRAM bandwidth capacity per GPU (normalized).");

DEFINE_string(model_alias_map_path,
              "",
              "Path to JSON file mapping alias model IDs to real model IDs "
              "for GP resource model lookup. Format: {\"alias\": \"real\", ...}. "
              "Empty means no alias mapping (model IDs used as-is).");

DEFINE_string(gp_steady_data_path,
              "",
              "Path to JSON file containing GP training data for steady pool "
              "resource model. Empty means use linear fallback.");

DEFINE_string(gp_dynamic_data_path,
              "",
              "Path to JSON file containing GP training data for dynamic pool "
              "resource model. Empty means use linear fallback.");

DEFINE_bool(disable_steady_pool,
            false,
            "When true, disable steady pool entirely. All models are assigned "
            "to the elastic pool with PD disaggregation enabled.");

DEFINE_bool(disable_elastic_pool,
            false,
            "When true, disable elastic pool entirely. All models are assigned "
            "to the steady pool only, no PD disaggregation or dynamic scaling.");

DEFINE_int32(elastic_instance_count,
             0,
             "Fixed number of instances per model when disable_steady_pool is "
             "true. 0 means use all available instances. Must be >= 2 if set "
             "to a positive value.");

DEFINE_string(models_config_path,
              "",
              "Path to JSON file containing the list of (service, model_path) "
              "pairs to be served. Each entry must have \"service\" and "
              "\"model_path\" fields. When set, this file is loaded at startup "
              "instead of using the hardcoded MODELS list.");

DEFINE_string(pool_memory_log_path,
              "",
              "Path to a dedicated log file for pool allocation and GPU memory "
              "utilization stats (written every 10s). Empty means use glog INFO.");

DEFINE_string(check_memory_log_path,
              "",
              "Path to a dedicated log file for per-instance XTensor heartbeat "
              "memory details (written every 1s). Used for memory leak diagnosis.");

DEFINE_double(overlap_abundance_gb,
              10.0,
              "Extra memory headroom (in GB) required for overlapped scale-up/"
              "scale-down. A GPU is eligible for overlapped scaling only when "
              "free_bytes >= new_model_size + overlap_abundance_gb.");

DEFINE_int32(max_models_per_gpu_in_steady_pool,
             2,
             "Maximum number of models that can be co-located on a single GPU "
             "instance in the steady pool. 0 means no limit (resource-only).");

DEFINE_int32(tensor_parallel_size,
             1,
             "Number of GPUs used for tensor parallelism per instance. "
             "Each instance occupies this many GPUs.");

DEFINE_string(initial_model_id,
              "",
              "Model ID of the initial model loaded by janus via --model. "
              "When non-empty, janus-service will send a /sleep request to "
              "each janus instance before fork_master_and_sleep to free GPU "
              "memory occupied by the initial model.");
