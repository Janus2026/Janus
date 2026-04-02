#include "async_response_processor.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <glog/logging.h>

#include <memory>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/request/finish_reason.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/blocking_counter.h"
#include "util/env_var.h"

namespace janus {

AsyncResponseProcessor::AsyncResponseProcessor(
    const Tokenizer* tokenizer,
    const std::optional<InstanceRole>& role,
    bool enable_service_routing)
    : response_threadpool_(FLAGS_num_response_handling_threads),
      tokenizer_(tokenizer->clone()),
      role_(role.value_or(InstanceRole::DEFAULT)),
      enable_batch_response_(enable_service_routing) {}

void AsyncResponseProcessor::process_failed_request(
    std::shared_ptr<Request> request,
    Status status) {
  // schedule the response handling
  auto runnable = [request = request, status = status]() {
    request->log_error_statistic(status);
    RequestOutput output;
    output.request_id = request->request_id();
    output.service_request_id = request->service_request_id();
    output.target_xservice_addr = request->source_xservice_addr();
    output.status = status;
    if (request->state().output_func) {
      request->state().output_func(output);
      return;
    }
    if (request->state().outputs_func) {
      request->state().outputs_func({output});
      return;
    }
    LOG(ERROR) << "Both output_func and outputs_func are empty for request: "
               << request->request_id();
  };
  if (request->state().response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(runnable);
  } else {
    response_threadpool_.schedule_with_tid(runnable,
                                           request->state().response_thread_id);
  }
}

void AsyncResponseProcessor::process_completed_request(
    std::shared_ptr<Request> request) {
  // schedule the response handling

  // NOTE: Maybe refactor later.
  // For prefill instance in 'ENABLE_SERVICE_DISPATCH_REQUEST' scenario,
  // Currently, the Janus service uses the BRPC HTTP interface to forward
  // requests. When the prefill process returns the first token to the Janus
  // service, the result is only returned after the OutputFunc callback function
  // is destructed. Therefore, to ensure the TTFT, the Output callback function
  // needs to be recycled once its execution is complete, allowing the first
  // token to be returned immediately. Additionally, the processing here runs in
  // parallel with the subsequent step where the prefill sends the first token
  // to the decode instance.
  //  When this completes, it destructs the request, which in turn causes the
  //  Output object
  // to be destructed. To prevent the scenario where the request is recycled
  // before the response is fully returned to the Janus service, the Output
  // object must be detached to avoid premature destruction.
  auto runnable = [this, request = request]() mutable {
    AUTO_COUNTER(responsing_latency_seconds_non_stream);

    double end_2_end_latency_seconds = request->elapsed_seconds();
    // update the metrics for the request
    HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                      static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
    RequestOutput req_output =
        request->generate_output(*tokenizer_, &generate_output_threadpool_);
    request->log_statistic(end_2_end_latency_seconds);
    if (request->state().output_func) {
      request->state().output_func(req_output);
      return;
    }
    if (request->state().outputs_func) {
      request->state().outputs_func({req_output});
      return;
    }
    LOG(ERROR) << "Both output_func and outputs_func are empty for request: "
               << request->request_id();
  };
  if (request->state().response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(runnable);
  } else {
    response_threadpool_.schedule_with_tid(runnable,
                                           request->state().response_thread_id);
  }
}

void AsyncResponseProcessor::batch_process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }

  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (int i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    auto runnable = [counter,
                     this,
                     request = request,
                     request_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_non_stream);
      double end_2_end_latency_seconds = request->elapsed_seconds();
      // update the metrics for the request
      HISTOGRAM_OBSERVE(
          end_2_end_latency_milliseconds,
          static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
      if (request->finished() || request->cancelled()) {
        request->log_statistic(end_2_end_latency_seconds);
      }

      *request_output = std::move(request->generate_output(*tokenizer_));
      if (request->sequences()[0]->num_generated_tokens() == 1) {
        // currently only support one sequence when enable_service_routing
        request_output->finished_on_prefill_instance = true;
      }
      counter->decrement_count();
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }

  rpc_threadpool_.schedule(
      [counter = std::unique_ptr<BlockingCounter>(counter),
       requests = std::move(requests),
       request_outputs = std::move(request_outputs)]() mutable {
        counter->wait();
        auto& resp_callback = requests[0]->state().outputs_func;
        if (resp_callback) {
          resp_callback(request_outputs);
          return;
        }

        LOG(WARNING) << "Batch callback outputs_func is empty, fallback to "
                        "per-request output_func. request size: "
                     << requests.size();
        for (size_t i = 0; i < requests.size(); ++i) {
          auto& single_callback = requests[i]->state().output_func;
          if (single_callback) {
            single_callback(request_outputs[i]);
            continue;
          }
          LOG(ERROR) << "Both output_func and outputs_func are empty for "
                        "request: "
                     << requests[i]->request_id();
        }
      });
}

// process non-stream requests
void AsyncResponseProcessor::process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }

  if (!enable_batch_response_) {
    for (size_t i = 0; i < requests.size(); ++i) {
      process_completed_request(std::move(requests[i]));
    }
  } else {
    // send request response to janus service in batch through rpc
    batch_process_completed_requests(requests);
  }
}

void AsyncResponseProcessor::process_stream_request(
    std::shared_ptr<Request> request) {
  CHECK(request->state().stream) << "request is not a streaming request";

  std::vector<size_t> indexes;
  std::vector<size_t> num_tokens;
  bool is_all_seqs_closed = true;
  for (size_t i = 0; i < request->sequences().size(); ++i) {
    auto& seq = request->sequences()[i];
    is_all_seqs_closed &= seq->is_closed();
    if (seq->is_closed()) {
      // skip already closed sequences
      continue;
    }

    // check if the sequence has enough tokens to output
    if (seq->has_new_tokens_generated() || seq->finished()) {
      indexes.push_back(i);
      num_tokens.push_back(seq->num_tokens());
    }

    // close the sequence after sending finish reason
    if (seq->finished()) {
      seq->close();
    }
  }

  if (!is_all_seqs_closed) {
    // output the delta text til the end of the sequence to the client

    auto runnable = [request,
                     this,
                     indexes = std::move(indexes),
                     num_tokens = std::move(num_tokens)]() {
      AUTO_COUNTER(responsing_latency_seconds_stream);

      RequestOutput req_output;
      req_output.request_id = request->request_id();
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];
        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output.outputs.push_back(std::move(seq_output.value()));
        }
        if (seq->num_generated_tokens() == 1) {
          req_output.finished_on_prefill_instance = true;
        }
      }
      bool should_continue = true;
      if (request->state().output_func) {
        should_continue = request->state().output_func(req_output);
      } else if (request->state().outputs_func) {
        std::vector<bool> status_set =
            request->state().outputs_func({req_output});
        should_continue = !status_set.empty() && status_set[0];
      } else {
        LOG(ERROR) << "Both output_func and outputs_func are empty for "
                      "request: "
                   << request->request_id();
        should_continue = false;
      }
      if (!should_continue) {
        // cancel the request if callback returns false
        request->set_cancel();
      }
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }
}

void AsyncResponseProcessor::batch_process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }

  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (int i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    CHECK(request->state().stream) << "request is not a streaming request";

    std::vector<size_t> indexes;
    std::vector<size_t> num_tokens;
    for (size_t i = 0; i < request->sequences().size(); ++i) {
      auto& seq = request->sequences()[i];
      if (seq->is_closed()) {
        // skip already closed sequences
        continue;
      }

      // check if the sequence has enough tokens to output
      if (seq->has_new_tokens_generated() || seq->finished()) {
        indexes.push_back(i);
        num_tokens.push_back(seq->num_tokens());
      }

      // close the sequence after sending finish reason
      if (seq->finished()) {
        seq->close();
      }
    }

    // output the delta text til the end of the sequence to the client
    auto runnable = [this,
                     counter,
                     request,
                     indexes = std::move(indexes),
                     num_tokens = std::move(num_tokens),
                     req_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_stream);

      // RequestOutput req_output;
      req_output->request_id = request->request_id();
      req_output->service_request_id = request->service_request_id();
      req_output->target_xservice_addr = request->source_xservice_addr();
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];

        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output->outputs.push_back(std::move(seq_output.value()));
        }
        // Set finished_on_prefill_instance regardless of whether
        // generate_streaming_output produced text, so that the batch
        // callback always decrements the rate limiter for prefill-done
        // requests.
        if (seq->num_generated_tokens() == 1) {
          req_output->finished_on_prefill_instance = true;
        }
      }
      counter->decrement_count();
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }

  rpc_threadpool_.schedule(
      [counter = std::unique_ptr<BlockingCounter>(counter),
       requests = std::move(requests),
       request_outputs = std::move(request_outputs)]() mutable {
        counter->wait();
        std::vector<bool> status_set;
        auto& resp_callback = requests[0]->state().outputs_func;
        if (resp_callback) {
          status_set = resp_callback(request_outputs);
        } else {
          LOG(WARNING) << "Batch callback outputs_func is empty, fallback to "
                          "per-request output_func. request size: "
                       << requests.size();
          status_set.reserve(requests.size());
          for (size_t i = 0; i < requests.size(); ++i) {
            auto& single_callback = requests[i]->state().output_func;
            if (single_callback) {
              status_set.push_back(single_callback(request_outputs[i]));
              continue;
            }
            LOG(ERROR) << "Both output_func and outputs_func are empty for "
                          "request: "
                       << requests[i]->request_id();
            status_set.push_back(false);
          }
        }
        for (size_t i = 0; i < requests.size(); ++i) {
          if (i >= status_set.size() || !status_set[i]) {
            // cancel the request if on_stream returns false
            requests[i]->set_cancel();
          }
        }
      });
}

// process stream requests
void AsyncResponseProcessor::process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }

  if (!enable_batch_response_) {
    for (auto& req : requests) {
      process_stream_request(req);
    }
  } else {
    // send request response to janus service in batch through rpc
    batch_process_stream_requests(requests);
  }
}

// for batch generate, wait all response done.
void AsyncResponseProcessor::wait_completion() {
  // Step 1: drain response_threadpool_ (output serialization tasks).
  {
    size_t thread_num = response_threadpool_.size();
    BlockingCounter counter(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto runnable = [&counter]() mutable { counter.decrement_count(); };
      response_threadpool_.schedule_with_tid(std::move(runnable), i);
    }
    counter.wait();
  }
  // Step 2: drain rpc_threadpool_ so that the batch callback (outputs_func)
  // has finished executing before the caller clears the callbacks.  Without
  // this, prefill_send_first_generation can null-out outputs_func/output_func
  // while the rpc_threadpool_ task still needs them, causing a rate-limiter
  // leak (the decrease_requests call is skipped).
  {
    size_t thread_num = rpc_threadpool_.size();
    BlockingCounter counter(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto runnable = [&counter]() mutable { counter.decrement_count(); };
      rpc_threadpool_.schedule_with_tid(std::move(runnable), i);
    }
    counter.wait();
  }
}

}  // namespace janus
