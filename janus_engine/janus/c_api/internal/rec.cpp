#include "c_api/rec.h"

#include <folly/Unit.h>
#include <folly/experimental/coro/Timeout.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <pthread.h>

#include <atomic>
#include <cstring>
#include <exception>
#include <stdexcept>

#include "helper.h"

JANUS_CAPI_EXPORT JANUS_REC_Handler* janus_rec_create(void) {
  JANUS_REC_Handler* handler = new JANUS_REC_Handler();
  CHECK(nullptr != handler);

  handler->initialized = false;

  return handler;
}

JANUS_CAPI_EXPORT void janus_rec_destroy(JANUS_REC_Handler* handler) {
  if (!handler) return;

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  delete handler;
}

JANUS_CAPI_EXPORT void janus_rec_init_options_default(
    JANUS_InitOptions* init_options) {
  if (nullptr == init_options) return;
  *init_options = JANUS_INIT_REC_OPTIONS_DEFAULT;
}

JANUS_CAPI_EXPORT bool janus_rec_initialize(
    JANUS_REC_Handler* handler,
    const char* model_path,
    const char* devices,
    const JANUS_InitOptions* init_options) {
  if (!handler || !model_path || !devices) return false;

  try {
    JANUS_InitOptions janus_init_options;
    janus::helper::set_init_options(
        janus::helper::BackendType::REC, init_options, &janus_init_options);

    std::string log_dir(janus_init_options.log_dir);
    if (!log_dir.empty()) {
      janus::helper::init_log(janus_init_options.log_dir);
    }

    if (!std::filesystem::exists(model_path)) {
      LOG(ERROR) << "model path[" << model_path << "] does not exist";
      return false;
    }

    janus::Options options;
    options.model_path(model_path)
        .task_type(janus_init_options.task)
        .devices(devices)
        .draft_model_path(janus_init_options.draft_model)
        .draft_devices(janus_init_options.draft_devices)
        .backend("rec")
        .block_size(janus_init_options.block_size)
        .max_cache_size(janus_init_options.max_cache_size)
        .max_memory_utilization(janus_init_options.max_memory_utilization)
        .enable_prefix_cache(janus_init_options.enable_prefix_cache)
        .max_tokens_per_batch(janus_init_options.max_tokens_per_batch)
        .max_seqs_per_batch(janus_init_options.max_seqs_per_batch)
        .max_tokens_per_chunk_for_prefill(
            janus_init_options.max_tokens_per_chunk_for_prefill)
        .num_speculative_tokens(janus_init_options.num_speculative_tokens)
        .num_request_handling_threads(
            janus_init_options.num_request_handling_threads)
        .communication_backend(janus_init_options.communication_backend)
        .expert_parallel_degree(janus_init_options.expert_parallel_degree)
        .enable_mla(janus_init_options.enable_mla)
        .enable_chunked_prefill(janus_init_options.enable_chunked_prefill)
        .master_node_addr(janus_init_options.master_node_addr)
        .device_ip(janus_init_options.device_ip)
        .transfer_listen_port(janus_init_options.transfer_listen_port)
        .nnodes(janus_init_options.nnodes)
        .node_rank(janus_init_options.node_rank)
        .dp_size(janus_init_options.dp_size)
        .ep_size(janus_init_options.ep_size)
        .instance_name(janus_init_options.instance_name)
        .enable_disagg_pd(janus_init_options.enable_disagg_pd)
        .enable_schedule_overlap(janus_init_options.enable_schedule_overlap)
        .enable_pd_ooc(janus_init_options.enable_pd_ooc)
        .kv_cache_transfer_mode(janus_init_options.kv_cache_transfer_mode)
        .enable_shm(janus_init_options.enable_shm)
        .is_local(true)
        .server_idx(janus_init_options.server_idx);

    // @TODO: Currently, gflags are configured through hard coding, which needs
    // to be improved in the future. For example, a separate gflags
    // configuration file can be provided to the so for setting gflags.
    FLAGS_beam_width = janus_init_options.beam_width;
    FLAGS_max_decode_rounds = janus_init_options.max_decode_rounds;
    FLAGS_max_seqs_per_batch = janus_init_options.max_seqs_per_batch;
    FLAGS_max_tokens_per_batch = janus_init_options.max_tokens_per_batch;
    FLAGS_block_size = janus_init_options.block_size;

    FLAGS_enable_graph = true;
    FLAGS_rec_worker_max_concurrency = 2;
    FLAGS_enable_rec_fast_sampler = true;
    FLAGS_enable_prefill_piecewise_graph = true;
    FLAGS_enable_xattention_one_stage = false;
    FLAGS_enable_xattention_two_stage_decode = true;
    FLAGS_enable_graph_mode_decode_no_padding = true;
    // FLAGS_enable_rec_prefill_only = true;
    FLAGS_enable_topk_sorted = false;

    options.enable_graph(FLAGS_enable_graph);

    handler->master = std::make_unique<janus::RecMaster>(options);
    handler->master->run();

    size_t cpu_cores = std::thread::hardware_concurrency();
    size_t thread_num = std::clamp((cpu_cores == 0) ? 8 : cpu_cores / 2,
                                   static_cast<size_t>(4),
                                   static_cast<size_t>(16));
    handler->executor =
        std::make_unique<folly::CPUThreadPoolExecutor>(thread_num);

    std::filesystem::path model_path_fs =
        std::filesystem::path(model_path).lexically_normal();
    std::string model_id;
    if (model_path_fs.has_filename()) {
      model_id = model_path_fs.filename().string();
    } else if (!model_path_fs.empty()) {
      model_id = model_path_fs.string();
    } else {
      model_id = "default";
    }
    handler->model_ids.clear();
    handler->model_ids.emplace_back(model_id);

    handler->initialized = true;

    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "LLM initialization failed: " << e.what();
  }

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  return false;
}

JANUS_CAPI_EXPORT void janus_rec_request_params_default(
    JANUS_RequestParams* request_params) {
  if (nullptr == request_params) return;
  *request_params = JANUS_REC_REQUEST_PARAMS_DEFAULT;
}

JANUS_CAPI_EXPORT JANUS_Response* janus_rec_text_completions(
    JANUS_REC_Handler* handler,
    const char* model_id,
    const char* prompt,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !prompt ||
      *prompt == '\0') {
    return janus::helper::build_error_response(
        "", JANUS_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  return janus::helper::handle_inference_request(
      handler,
      janus::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      prompt,
      nullptr,
      timeout_ms,
      request_params);
}

JANUS_CAPI_EXPORT JANUS_Response* janus_rec_token_completions(
    JANUS_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !token_ids ||
      token_size == 0) {
    return janus::helper::build_error_response(
        "", JANUS_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  std::vector<int> token_ids_vec;
  for (int i = 0; i < token_size; i++) {
    token_ids_vec.push_back(token_ids[i]);
  }

  return janus::helper::handle_inference_request(
      handler,
      janus::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      token_ids_vec,
      nullptr,
      timeout_ms,
      request_params);
}

JANUS_CAPI_EXPORT JANUS_Response* janus_rec_multimodal_completions(
    JANUS_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    const JANUS_MM_Data* mm_data,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !token_ids ||
      token_size == 0) {
    return janus::helper::build_error_response(
        "", JANUS_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  if (!mm_data) {
    return janus_rec_token_completions(
        handler, model_id, token_ids, token_size, timeout_ms, request_params);
  }

  janus::MMData internal_mm_data;
  try {
    bool ret = janus::helper::convert_janus_mm_data_to_internal(mm_data,
                                                              internal_mm_data);
    if (!ret) {
      return janus::helper::build_error_response(
          "", JANUS_StatusCode::kInternalError, "Fail in mm_data conversion");
    }
  } catch (const std::exception& e) {
    return janus::helper::build_error_response(
        "",
        JANUS_StatusCode::kInternalError,
        "Critical error in mm_data conversion: " + std::string(e.what()));
  }

  std::vector<int> token_ids_vec;
  for (int i = 0; i < token_size; i++) {
    token_ids_vec.push_back(token_ids[i]);
  }

  return janus::helper::handle_inference_request(
      handler,
      janus::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      token_ids_vec,
      static_cast<void*>(&internal_mm_data),
      timeout_ms,
      request_params);
}

JANUS_CAPI_EXPORT JANUS_Response* janus_rec_chat_completions(
    JANUS_REC_Handler* handler,
    const char* model_id,
    const JANUS_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !messages ||
      messages_count == 0) {
    return janus::helper::build_error_response(
        "", JANUS_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  std::vector<janus::Message> janus_messages;
  janus_messages.reserve(messages_count);
  for (int i = 0; i < messages_count; i++) {
    janus_messages.emplace_back(messages[i].role, messages[i].content);
  }

  return janus::helper::handle_inference_request(
      handler,
      janus::helper::InferenceType::REC_CHAT_COMPLETIONS,
      model_id,
      janus_messages,
      nullptr,
      timeout_ms,
      request_params);
}

JANUS_CAPI_EXPORT void janus_rec_free_response(JANUS_Response* resp) {
  return janus::helper::janus_free_response(resp);
}
