#pragma once

#include <folly/executors/CPUThreadPoolExecutor.h>

#include "core/common/instance_name.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/util/uuid.h"
#include "types.h"

namespace janus {

struct LLMCore {
  // List of loaded model identifiers
  std::vector<std::string> model_ids;

  // Master controller for LLM runtime management
  std::unique_ptr<LLMMaster> master;

  // Thread pool for asynchronous task execution
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor;
};

namespace detail {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_request_id() {
  return "janus-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}
}  // namespace

enum class InterfaceType { COMPLETIONS, CHAT_COMPLETIONS };

RequestParams transfer_request_params(
    const JANUS_RequestParams& request_params) {
  RequestParams janus_request_params;

  janus_request_params.echo = request_params.echo;
  janus_request_params.offline = request_params.offline;
  janus_request_params.logprobs = request_params.logprobs;
  janus_request_params.best_of = request_params.best_of;
  janus_request_params.slo_ms = request_params.slo_ms;
  janus_request_params.top_k = request_params.top_k;
  janus_request_params.top_p = request_params.top_p;
  janus_request_params.ignore_eos = request_params.ignore_eos;
  janus_request_params.skip_special_tokens = request_params.skip_special_tokens;
  janus_request_params.n = request_params.n;
  janus_request_params.max_tokens = request_params.max_tokens;
  janus_request_params.frequency_penalty = request_params.frequency_penalty;
  janus_request_params.presence_penalty = request_params.presence_penalty;
  janus_request_params.repetition_penalty = request_params.repetition_penalty;
  janus_request_params.stop = request_params.stop;
  janus_request_params.stop_token_ids = request_params.stop_token_ids;
  janus_request_params.beam_width = request_params.beam_width;
  janus_request_params.top_logprobs = request_params.top_logprobs;

  return janus_request_params;
}

JANUS_Response build_success_response(const RequestOutput& output,
                                     const InterfaceType& if_type,
                                     const std::string& request_id,
                                     int64_t created_time,
                                     const std::string& model) {
  JANUS_Response response;

  response.status_code = JANUS_StatusCode::kSuccess;

  response.id = request_id;
  response.created = created_time;
  response.model = model;
  if (if_type == InterfaceType::COMPLETIONS) {
    response.object = "text_completion";
  } else if (if_type == InterfaceType::CHAT_COMPLETIONS) {
    response.object = "chat.completion";
  }

  response.choices.reserve(output.outputs.size());
  for (const auto& output : output.outputs) {
    JANUS_Choice choice;
    choice.index = output.index;

    if (output.logprobs.has_value()) {
      std::vector<JANUS_LogProb> janus_logprobs;
      janus_logprobs.reserve(output.logprobs.value().size());
      for (const auto& logprob : output.logprobs.value()) {
        JANUS_LogProb janus_logprob;
        janus_logprob.token = logprob.token;
        janus_logprob.token_id = logprob.token_id;
        janus_logprob.logprob = logprob.logprob;

        if (logprob.top_logprobs.has_value()) {
          janus_logprob.top_logprobs.reserve(
              logprob.top_logprobs.value().size());
          for (const auto& top_logprob : logprob.top_logprobs.value()) {
            JANUS_LogProbData janus_logprob_data;
            janus_logprob_data.token = top_logprob.token;
            janus_logprob_data.token_id = top_logprob.token_id;
            janus_logprob_data.logprob = top_logprob.logprob;
            janus_logprob.top_logprobs.emplace_back(janus_logprob_data);
          }
        }
        janus_logprobs.emplace_back(janus_logprob);
      }

      choice.logprobs = janus_logprobs;
    }

    if (if_type == InterfaceType::COMPLETIONS) {
      choice.text = output.text;
    } else if (if_type == InterfaceType::CHAT_COMPLETIONS) {
      JANUS_ChatMessage chat_message;
      chat_message.role = "assistant";
      chat_message.content = output.text;
      choice.message = chat_message;
    }

    if (output.finish_reason.has_value()) {
      choice.finish_reason = output.finish_reason.value();
    }

    response.choices.emplace_back(choice);
  }

  if (output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.usage.prompt_tokens = usage.num_prompt_tokens;
    response.usage.completion_tokens = usage.num_generated_tokens;
    response.usage.total_tokens = usage.num_total_tokens;
  }

  return response;
}

JANUS_Response build_error_response(const std::string& request_id,
                                   JANUS_StatusCode status_code,
                                   const std::string& error_info) {
  JANUS_Response response;
  response.status_code = status_code;
  response.error_info = error_info;
  response.id = request_id.empty() ? "unknown_request" : request_id;

  LOG(ERROR) << "Request [" << response.id << "] error: " << error_info
             << " (code: " << static_cast<int>(response.status_code) << ")";

  return response;
}

template <typename InputType>
JANUS_Response handle_inference_request(LLMCore* llm_core,
                                       const std::string& model_id,
                                       const InputType& input,
                                       uint32_t timeout_ms,
                                       const JANUS_RequestParams& request_params,
                                       InterfaceType interface_type) {
  if (!llm_core) {
    return build_error_response(
        "", JANUS_StatusCode::kNotInitialized, "LLM is not initialized");
  }

  auto it = std::find(
      llm_core->model_ids.begin(), llm_core->model_ids.end(), model_id);
  if (it == llm_core->model_ids.end()) {
    return build_error_response("",
                                JANUS_StatusCode::kModelNotFound,
                                "Specified model ID not loaded: " + model_id);
  }

  RequestParams janus_request_params = transfer_request_params(request_params);
  std::string request_id = janus_request_params.request_id.empty()
                               ? generate_request_id()
                               : janus_request_params.request_id;
  janus_request_params.request_id = request_id;
  int64_t created_time = absl::ToUnixSeconds(absl::Now());

  try {
    auto promise_ptr = std::make_shared<folly::Promise<JANUS_Response>>();
    auto weak_promise =
        std::weak_ptr<folly::Promise<JANUS_Response>>(promise_ptr);
    auto future = promise_ptr->getSemiFuture();

    llm_core->master->handle_request(
        input,
        std::nullopt,
        janus_request_params,
        std::nullopt,
        [model_id,
         request_id,
         created_time,
         interface_type,
         weak_promise,
         timeout_ms](const RequestOutput& req_output) -> bool {
          auto promise_ptr = weak_promise.lock();
          if (!promise_ptr) {
            return false;
          }

          try {
            JANUS_Response response = build_success_response(
                req_output, interface_type, request_id, created_time, model_id);
            promise_ptr->setValue(std::move(response));
          } catch (const folly::PromiseAlreadySatisfied& e) {
            return false;
          }

          return true;
        });

    return std::move(future)
        .via(llm_core->executor.get())
        .within(std::chrono::milliseconds(timeout_ms))
        .thenTry([](folly::Try<JANUS_Response>&& result) {
          if (result.hasValue()) {
            return std::move(result).value();
          } else {
            result.throwUnlessValue();
            return JANUS_Response{};
          }
        })
        .get();

  } catch (const folly::FutureTimeout& e) {
    return build_error_response(
        request_id, JANUS_StatusCode::kTimeout, "Request timed out");
  } catch (const std::exception& e) {
    return build_error_response(
        request_id,
        JANUS_StatusCode::kInternalError,
        "Failed to handle request: " + std::string(e.what()));
  }
}

}  // namespace detail
}  // namespace janus
