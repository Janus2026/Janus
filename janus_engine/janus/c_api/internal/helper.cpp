#include "helper.h"

#include <glog/logging.h>
#include <pthread.h>
#include <torch/torch.h>

#include <atomic>
#include <string>

#include "core/common/global_flags.h"
#include "core/util/env_var.h"
#include "core/util/uuid.h"

namespace janus {
namespace helper {
namespace {
thread_local ShortUUID short_uuid;
static std::atomic<bool> g_glog_inited = false;
static pthread_mutex_t g_log_init_mutex = PTHREAD_MUTEX_INITIALIZER;
}  // namespace

std::string generate_request_id() {
  return "janus-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

void init_log(const std::string& log_dir) {
  if (g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (!g_glog_inited.load(std::memory_order_relaxed)) {
    google::InitGoogleLogging("janus");

    std::string log_prefix = log_dir.empty() ? "./" : log_dir + "/";
    google::SetLogDestination(google::INFO,
                              (log_prefix + "janus.log.INFO.").c_str());
    google::SetLogDestination(google::WARNING,
                              (log_prefix + "janus.log.WARNING.").c_str());
    google::SetLogDestination(google::ERROR,
                              (log_prefix + "janus.log.ERROR.").c_str());
    google::SetStderrLogging(google::FATAL);
    g_glog_inited.store(true, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void shutdown_log() {
  if (!g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (g_glog_inited.load(std::memory_order_relaxed)) {
    google::ShutdownGoogleLogging();
    g_glog_inited.store(false, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void set_init_options(BackendType backend_type,
                      const JANUS_InitOptions* init_options,
                      JANUS_InitOptions* janus_init_options) {
  if (init_options == nullptr) {
    if (backend_type == BackendType::LLM) {
      memcpy(janus_init_options,
             &JANUS_INIT_LLM_OPTIONS_DEFAULT,
             sizeof(JANUS_InitOptions));
    } else if (backend_type == BackendType::REC) {
      memcpy(janus_init_options,
             &JANUS_INIT_REC_OPTIONS_DEFAULT,
             sizeof(JANUS_InitOptions));
    }
  } else {
    memcpy(janus_init_options, init_options, sizeof(JANUS_InitOptions));
  }

  return;
}

void transfer_request_params(InferenceType inference_type,
                             const JANUS_RequestParams* request_params,
                             janus::RequestParams* janus_request_params) {
  JANUS_RequestParams final_request_params;
  if (nullptr == request_params) {
    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::LLM_CHAT_COMPLETIONS) {
      memcpy(&final_request_params,
             &JANUS_LLM_REQUEST_PARAMS_DEFAULT,
             sizeof(JANUS_RequestParams));
    } else if (inference_type == InferenceType::REC_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      memcpy(&final_request_params,
             &JANUS_REC_REQUEST_PARAMS_DEFAULT,
             sizeof(JANUS_RequestParams));
    }
  } else {
    memcpy(&final_request_params, request_params, sizeof(JANUS_RequestParams));
  }

  janus_request_params->echo = final_request_params.echo;
  janus_request_params->offline = final_request_params.offline;
  janus_request_params->logprobs = final_request_params.logprobs;
  janus_request_params->ignore_eos = final_request_params.ignore_eos;

  janus_request_params->best_of = final_request_params.best_of;
  janus_request_params->top_k = final_request_params.top_k;
  janus_request_params->top_p = final_request_params.top_p;
  janus_request_params->n = final_request_params.n;
  janus_request_params->max_tokens = final_request_params.max_tokens;
  janus_request_params->frequency_penalty =
      final_request_params.frequency_penalty;
  janus_request_params->presence_penalty = final_request_params.presence_penalty;
  janus_request_params->repetition_penalty =
      final_request_params.repetition_penalty;
  janus_request_params->beam_width = final_request_params.beam_width;
  janus_request_params->top_logprobs = final_request_params.top_logprobs;
  janus_request_params->temperature = final_request_params.temperature;
  janus_request_params->request_id = final_request_params.request_id;
  janus_request_params->ttlt_slo_ms = final_request_params.ttlt_slo_ms;
  janus_request_params->ttft_slo_ms = final_request_params.ttft_slo_ms;
  janus_request_params->tpot_slo_ms = final_request_params.tpot_slo_ms;

  return;
}

JANUS_Response* build_error_response(const std::string& request_id,
                                    JANUS_StatusCode status_code,
                                    const std::string& error_info) {
  JANUS_Response* response = new JANUS_Response();
  CHECK(nullptr != response);

  response->status_code = status_code;
  strncpy(
      response->error_info, error_info.c_str(), JANUS_ERROR_INFO_MAX_LEN - 1);
  response->error_info[JANUS_ERROR_INFO_MAX_LEN - 1] = '\0';

  JANUS_SET_META_STRING_FIELD(response->id, request_id);

  LOG(ERROR) << "Request [" << request_id << "] error: " << error_info
             << " (code: " << static_cast<int>(response->status_code) << ")";

  return response;
}

JANUS_Response* build_success_response(const InferenceType& inference_type,
                                      const RequestOutput& output,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model) {
  JANUS_Response* response = new JANUS_Response();
  CHECK(nullptr != response);

  response->status_code = JANUS_StatusCode::kSuccess;
  response->created = created_time;
  JANUS_SET_META_STRING_FIELD(response->id, request_id);
  JANUS_SET_META_STRING_FIELD(response->model, model);

  if (inference_type == InferenceType::LLM_COMPLETIONS ||
      inference_type == InferenceType::REC_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "text_completion");
  } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
             inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "chat.completion");
  }

  response->choices.entries_size = output.outputs.size();
  response->choices.entries = new JANUS_Choice[response->choices.entries_size]();
  CHECK(nullptr != response->choices.entries);

  for (int i = 0; i < output.outputs.size(); i++) {
    const auto& seq_output = output.outputs[i];
    JANUS_Choice& choice = response->choices.entries[i];
    choice.index = seq_output.index;

    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::REC_COMPLETIONS) {
      size_t text_len = seq_output.text.length();
      choice.text = new char[text_len + 1];
      CHECK(nullptr != choice.text);
      strncpy(choice.text, seq_output.text.c_str(), text_len + 1);
      choice.text[text_len] = '\0';
    } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      choice.message = new JANUS_ChatMessage();
      CHECK(nullptr != choice.message);

      snprintf(choice.message->role, sizeof(choice.message->role), "assistant");
      size_t text_len = seq_output.text.length();
      choice.message->content = new char[text_len + 1];
      CHECK(nullptr != choice.message->content);
      strncpy(choice.message->content, seq_output.text.c_str(), text_len + 1);
      choice.message->content[text_len] = '\0';
    }

    if (seq_output.finish_reason.has_value()) {
      JANUS_SET_META_STRING_FIELD(choice.finish_reason,
                                 seq_output.finish_reason.value());
    }

    if (seq_output.token_ids.size() > 0) {
      choice.token_size = seq_output.token_ids.size();
      choice.token_ids = new int32_t[choice.token_size];
      CHECK(nullptr != choice.token_ids);
      for (int j = 0; j < choice.token_size; j++) {
        choice.token_ids[j] = seq_output.token_ids[j];
      }
    }

    if (seq_output.logprobs.has_value()) {
      choice.logprobs.entries_size = seq_output.logprobs.value().size();
      choice.logprobs.entries =
          new JANUS_LogProb[choice.logprobs.entries_size]();
      CHECK(nullptr != choice.logprobs.entries);
      for (int j = 0; j < seq_output.logprobs.value().size(); j++) {
        const auto& logprob = seq_output.logprobs.value()[j];
        JANUS_LogProb& janus_logprob = choice.logprobs.entries[j];

        janus_logprob.token_id = logprob.token_id;
        janus_logprob.logprob = logprob.logprob;
      }
    }
  }

  if (output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response->usage.prompt_tokens = usage.num_prompt_tokens;
    response->usage.completion_tokens = usage.num_generated_tokens;
    response->usage.total_tokens = usage.num_total_tokens;
  }

  return response;
}

template <typename HandlerType, typename InputType>
JANUS_Response* handle_inference_request(
    HandlerType* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const InputType& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params) {
  CHECK(nullptr != handler);

  std::string request_id;
  if (nullptr != request_params && strlen(request_params->request_id) > 0) {
    request_id = request_params->request_id;
  } else {
    request_id = generate_request_id();
  }

  if (!handler->initialized) {
    return build_error_response(
        request_id, JANUS_StatusCode::kNotInitialized, "LLM is not initialized");
  }

  if (std::find(handler->model_ids.begin(),
                handler->model_ids.end(),
                model_id) == handler->model_ids.end()) {
    return build_error_response(request_id,
                                JANUS_StatusCode::kModelNotFound,
                                "Specified model ID not loaded: " + model_id);
  }

  janus::RequestParams janus_request_params;
  transfer_request_params(inference_type, request_params, &janus_request_params);
  janus_request_params.request_id = request_id;

  const int64_t created_time = absl::ToUnixSeconds(absl::Now());

  try {
    auto promise_ptr = std::make_shared<folly::Promise<JANUS_Response*>>();
    auto future = promise_ptr->getSemiFuture();

    auto on_request_complete = [model_id,
                                request_id,
                                created_time,
                                inference_type,
                                weak_promise = std::weak_ptr(promise_ptr)](
                                   const RequestOutput& req_output) -> bool {
      if (auto locked_promise = weak_promise.lock()) {
        try {
          if (req_output.status.has_value()) {
            if (req_output.status.value().ok()) {
              locked_promise->setValue(build_success_response(inference_type,
                                                              req_output,
                                                              request_id,
                                                              created_time,
                                                              model_id));
            } else {
              locked_promise->setValue(build_error_response(
                  request_id,
                  JANUS_StatusCode::kInternalError,
                  "RequestOutput status is not ok, message: " +
                      req_output.status.value().message()));
            }
          } else {
            locked_promise->setValue(
                build_error_response(request_id,
                                     JANUS_StatusCode::kInternalError,
                                     "RequestOutput status has no value"));
          }
          return true;
        } catch (const std::exception& e) {
          LOG(ERROR) << "Build response failed: " << e.what();
          locked_promise->setValue(build_error_response(
              request_id,
              JANUS_StatusCode::kInternalError,
              "Build response failed: " + std::string(e.what())));
        }
      }
      return false;
    };

    if constexpr (std::is_same_v<HandlerType, JANUS_LLM_Handler>) {
      handler->master->handle_request(input,
                                      std::nullopt,
                                      janus_request_params,
                                      std::nullopt,
                                      on_request_complete);
    } else if constexpr (std::is_same_v<HandlerType, JANUS_REC_Handler>) {
      if constexpr (std::is_same_v<InputType, std::vector<int>>) {
        if (nullptr != extra) {
          janus::MMData* mm_data =
              dynamic_cast<janus::MMData*>(static_cast<janus::MMData*>(extra));
          CHECK(nullptr != mm_data);

          std::optional<janus::MMData> opt_mm_data = std::move(*mm_data);
          handler->master->handle_request(
              input, opt_mm_data, janus_request_params, on_request_complete);

        } else {
          handler->master->handle_request("",
                                          input,
                                          std::nullopt,
                                          janus_request_params,
                                          on_request_complete);
        }
      } else {
        handler->master->handle_request(input,
                                        std::nullopt,
                                        std::nullopt,
                                        janus_request_params,
                                        on_request_complete);
      }
    } else {
      CHECK(false);
    }

    return std::move(future)
        .via(handler->executor.get())
        .within(std::chrono::milliseconds(timeout_ms))
        .thenTry([request_id](
                     folly::Try<JANUS_Response*>&& result) -> JANUS_Response* {
          if (result.hasValue()) return std::move(result).value();

          std::string error_msg;
          JANUS_StatusCode code = JANUS_StatusCode::kInternalError;
          try {
            result.throwUnlessValue();
          } catch (const folly::FutureTimeout& e) {
            error_msg = "Request timed out: " + std::string(e.what());
            code = JANUS_StatusCode::kTimeout;
          } catch (const std::exception& e) {
            error_msg = "Inference failed: " + std::string(e.what());
          } catch (...) {
            error_msg = "Inference failed with unknown exception";
          }
          return build_error_response(request_id, code, error_msg);
        })
        .get();

  } catch (...) {
    return build_error_response(request_id,
                                JANUS_StatusCode::kInternalError,
                                "Critical error in inference pipeline");
  }
}

void janus_free_response(JANUS_Response* resp) {
  if (nullptr == resp) {
    return;
  }

  if (nullptr != resp->choices.entries) {
    for (int i = 0; i < resp->choices.entries_size; ++i) {
      JANUS_Choice& choice = resp->choices.entries[i];

      if (nullptr != choice.text) {
        delete[] choice.text;
        choice.text = nullptr;
      }

      if (nullptr != choice.message) {
        if (nullptr != choice.message->content) {
          delete[] choice.message->content;
          choice.message->content = nullptr;
        }
        delete choice.message;
        choice.message = nullptr;
      }

      if (nullptr != choice.token_ids) {
        delete[] choice.token_ids;
        choice.token_ids = nullptr;
        choice.token_size = 0;
      }

      if (nullptr != choice.logprobs.entries) {
        delete[] choice.logprobs.entries;
        choice.logprobs.entries = nullptr;
      }
      choice.logprobs.entries_size = 0;
    }

    delete[] resp->choices.entries;
    resp->choices.entries = nullptr;
  }

  resp->choices.entries_size = 0;
  delete resp;

  return;
}

torch::ScalarType janus_dtype_to_torch_scalar_type(JANUS_DataType dtype) {
  switch (dtype) {
    case JANUS_DTYPE_UNDEFINED:
      throw std::runtime_error(
          "JANUS_DTYPE_UNDEFINED is not a valid dtype for tensor conversion");
    case JANUS_DTYPE_FLOAT16:
      return torch::kFloat16;
    case JANUS_DTYPE_FLOAT32:
      return torch::kFloat32;
    case JANUS_DTYPE_FLOAT64:
      return torch::kFloat64;
    case JANUS_DTYPE_BFLOAT16:
      return torch::kBFloat16;
    case JANUS_DTYPE_INT8:
      return torch::kInt8;
    case JANUS_DTYPE_INT16:
      return torch::kInt16;
    case JANUS_DTYPE_INT32:
      return torch::kInt32;
    case JANUS_DTYPE_INT64:
      return torch::kInt64;
    case JANUS_DTYPE_BOOL:
      return torch::kBool;
    case JANUS_DTYPE_STRING:
      throw std::runtime_error(
          "String dtype is not supported for torch::Tensor");
    default:
      throw std::runtime_error("Unsupported JANUS_DataType: " +
                               std::to_string(dtype));
  }
}

torch::Tensor convert_janus_tensor_to_torch(const JANUS_Tensor& janus_tensor) {
  if (janus_tensor.data == nullptr) {
    throw std::runtime_error("JANUS_Tensor data pointer is null");
  }

  torch::ScalarType scalar_type =
      janus_dtype_to_torch_scalar_type(janus_tensor.dtype);

  std::vector<int64_t> shape;
  for (int i = 0; i < janus_tensor.dims.rank; ++i) {
    int dim = janus_tensor.dims.dim[i];
    if (dim > 0) {
      shape.push_back(dim);
    }
  }

  if (shape.empty()) {
    throw std::runtime_error("JANUS_Tensor all dimensions are invalid value");
  }

  torch::Tensor tensor =
      torch::from_blob(const_cast<void*>(janus_tensor.data), shape, scalar_type)
          .clone();

  return tensor;
}

janus::MMDataItem convert_janus_mm_item_to_internal(
    const JANUS_MM_Item& janus_item) {
  uint32_t janus_type_val = static_cast<uint32_t>(janus_item.type);
  janus::MMType::Value internal_val = janus::MMType::NONE;

  switch (janus_type_val) {
    case JANUS_MM_TYPE_EMBEDDING:
      internal_val = janus::MMType::EMBEDDING;
      break;
    case JANUS_MM_TYPE_IMAGE:
      internal_val = janus::MMType::IMAGE;
      break;
    case JANUS_MM_TYPE_VIDEO:
      internal_val = janus::MMType::VIDEO;
      break;
    case JANUS_MM_TYPE_AUDIO:
      internal_val = janus::MMType::AUDIO;
      break;
    case JANUS_MM_TYPE_NONE:
      internal_val = janus::MMType::NONE;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported JANUS_MM_Type: ") +
                               std::to_string(janus_type_val));
  }

  janus::MMType item_type(internal_val);
  janus::MMDataItem internal_item(item_type);

  janus::MMItemState& state = internal_item.mutable_state();
  janus::MMItemState::TokenPos& token_pos = state.mutable_token_pos();
  token_pos.offset = janus_item.state.token_pos.offset;
  token_pos.length = janus_item.state.token_pos.length;

  if (janus_item.data.is_single_tensor) {
    torch::Tensor tensor =
        convert_janus_tensor_to_torch(janus_item.data.data.tensor);
    internal_item.add("tensor", tensor);
  } else {
    std::vector<torch::Tensor> tensor_list;
    const JANUS_Tensors& janus_tensors = janus_item.data.data.tensors;
    for (size_t i = 0; i < janus_tensors.entries_size; ++i) {
      tensor_list.push_back(
          convert_janus_tensor_to_torch(janus_tensors.entries[i]));
    }
    internal_item.add("tensor_list", tensor_list);
  }

  return internal_item;
}

bool convert_janus_mm_data_to_internal(const JANUS_MM_Data* mm_data,
                                      janus::MMData& internal_mm_data) {
  if (mm_data == nullptr || mm_data->type_mask == JANUS_MM_TYPE_NONE) {
    return false;
  }

  janus::MMType::Value internal_val =
      static_cast<janus::MMType::Value>(mm_data->type_mask);
  janus::MMType mm_type(internal_val);

  if (mm_data->is_dict) {
    const JANUS_MM_Dict& janus_dict = mm_data->data.dict;
    janus::MMDict internal_dict;

    for (size_t i = 0; i < janus_dict.entries_size; ++i) {
      const JANUS_MM_DictEntry& janus_entry = janus_dict.entries[i];
      janus::MMKey key(janus_entry.key);

      const JANUS_MM_Value& janus_value = janus_entry.value;
      if (janus_value.is_single_tensor) {
        torch::Tensor tensor =
            convert_janus_tensor_to_torch(janus_value.data.tensor);
        internal_dict.insert({key, tensor});
      } else {
        std::vector<torch::Tensor> tensor_list;
        const JANUS_Tensors& janus_tensors = janus_value.data.tensors;
        for (size_t j = 0; j < janus_tensors.entries_size; ++j) {
          tensor_list.push_back(
              convert_janus_tensor_to_torch(janus_tensors.entries[j]));
        }
        internal_dict.insert({key, tensor_list});
      }
    }

    internal_mm_data.set<janus::MMDict>(mm_type, internal_dict);
  } else {
    const JANUS_MM_Items& janus_items = mm_data->data.items;
    janus::MMItemVec internal_item_vec;

    for (size_t i = 0; i < janus_items.entries_size; ++i) {
      const JANUS_MM_Item& janus_item = janus_items.entries[i];

      janus::MMDataItem internal_item =
          convert_janus_mm_item_to_internal(janus_item);
      internal_item_vec.push_back(std::move(internal_item));
    }

    internal_mm_data.set<janus::MMItemVec>(mm_type, internal_item_vec);
  }

  return true;
}

// 1. LLM Handler + const char* (text completions)
template JANUS_Response* handle_inference_request<JANUS_LLM_Handler, const char*>(
    JANUS_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

// 2. LLM Handler + std::vector<janus::Message> (chat completions)
template JANUS_Response*
handle_inference_request<JANUS_LLM_Handler, std::vector<janus::Message>>(
    JANUS_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<janus::Message>& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

// 3. REC Handler + const char* (REC completions)
template JANUS_Response* handle_inference_request<JANUS_REC_Handler, const char*>(
    JANUS_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

// 4. REC Handler + std::vector<janus::Message> (REC chat completions)
template JANUS_Response*
handle_inference_request<JANUS_REC_Handler, std::vector<janus::Message>>(
    JANUS_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<janus::Message>& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

// 5. REC Handler + std::vector<int> (chat completions)
template JANUS_Response*
handle_inference_request<JANUS_REC_Handler, std::vector<int>>(
    JANUS_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<int>& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);
}  // namespace helper
}  // namespace janus
