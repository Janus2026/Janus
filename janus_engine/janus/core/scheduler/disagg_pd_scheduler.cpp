#include "scheduler/disagg_pd_scheduler.h"

#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/server.h>

#include <random>

#include "common/global_flags.h"
#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_scheduler.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"
#include "framework/xtensor/page_allocator.h"
#include "runtime/xservice_client.h"
#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "util/env_var.h"

namespace janus {

DisaggPDScheduler::DisaggPDScheduler(Engine* engine, const Options& options)
    : ContinuousScheduler(engine, options), server_name_("DisaggPDServer") {
  if (!options_.instance_role().has_value()) {
    LOG(FATAL) << "Instance type is not set in disagg pd mode.";
  }

  // Only initialize for non-OOC mode
  // OOC mode (PDOOCScheduler) will handle initialization in its own constructor
  if (!options_.enable_pd_ooc()) {
    // Start dispatch thread for prefill instance
    dispatch_thread_ = std::make_unique<std::thread>(
        &DisaggPDScheduler::dispatch_requests, this);

    // Start RPC server thread
    server_name_.append(std::to_string(options.server_idx()));
    rpc_server_thread_ = std::make_unique<std::thread>(
        &DisaggPDScheduler::start_rpc_server, this);
    initialize_rpc_server(server_name_);
    register_instance_info(server_name_, engine);

    // Profile ttft & tpot and update instance info (for mix instances).
    // NOTE: Disabled for MIX instances in MIXPD mode. Profiling is only
    // consumed by janus-service's SLO_AWARE/LST_IMH load-balance policies,
    // which are not used in the current MIXPD setup (default policy is RR).
    // More critically, profiling allocates a large number of KV blocks at
    // startup, which exhausts the XTensor physical page pool (shared with
    // model weights) and causes a fatal "No free physical pages" crash.
    // if (!options_.disable_ttft_profiling() &&
    //     options_.instance_role().value() == InstanceRole::MIX) {
    //   profile_ttft();
    //   profile_tpot();
    // }
  }
}

DisaggPDScheduler::~DisaggPDScheduler() {
  // Clean up common threads (shared by both OOC and non-OOC modes)
  if (rpc_server_thread_ && rpc_server_thread_->joinable()) {
    rpc_server_thread_->join();
  }

  // Clean up dispatch thread (created in base class for non-OOC mode,
  // or in subclass for OOC mode)
  if (dispatch_thread_ && dispatch_thread_->joinable()) {
    dispatch_thread_->join();
  }

  auto rpc_server = ServerRegistry::get_instance().get_server(server_name_);
  if (rpc_server != nullptr) {
    rpc_server->stop();

    ServerRegistry::get_instance().unregister_server(server_name_);
  }
}

void DisaggPDScheduler::initialize_rpc_server(const std::string& server_name) {
  // wait rpc server initialized
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name);
  while (!rpc_server || !rpc_server->has_initialized()) {
    absl::SleepFor(absl::Milliseconds(100));
    rpc_server = ServerRegistry::get_instance().get_server(server_name);
  }
  // connect to master service
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
  xservice_client_->set_scheduler(this);
  if (FLAGS_enable_xtensor) {
    xservice_client_->set_engine(engine_);
  }
}

void DisaggPDScheduler::register_instance_info(const std::string& server_name,
                                               Engine* engine) {
  // register instance info
  instance_info_.name = xservice_client_->get_instance_name();
  instance_info_.model_id = options_.model_id();
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name);
  instance_info_.rpc_address = rpc_server->listen_address();
  instance_info_.type = options_.instance_role().value().to_string();
  LOG(INFO) << "Instance info: instance name = " << instance_info_.name
            << ", model_id = " << instance_info_.model_id
            << ", instance rpc_address = " << instance_info_.rpc_address
            << ", instance type = " << instance_info_.type;

  engine->get_cache_info(instance_info_.cluster_ids,
                         instance_info_.addrs,
                         instance_info_.k_cache_ids,
                         instance_info_.v_cache_ids);
  instance_info_.dp_size = options_.dp_size();

  // device_ips, ports, p2p_addrs are already populated by the
  // ContinuousScheduler (parent) constructor — no need to call
  // get_device_info / get_p2p_addrs again here.

  // Get total physical pages per worker (for etcd registration)
#if defined(USE_NPU)
  if (FLAGS_enable_xtensor) {
    auto& page_allocator = PageAllocator::get_instance();
    if (page_allocator.is_initialized()) {
      instance_info_.total_phy_pages = page_allocator.get_num_total_phy_pages();
    }
  }
#endif
}

// TODO: maybe we should consider update info case even if info already exists
// in local.
bool DisaggPDScheduler::check_remote_instance_info(
    const std::string& instance_name) {
  if (remote_instances_info_.find(instance_name) !=
      remote_instances_info_.end()) {
    return true;
  }

  InstanceInfo instance_info =
      xservice_client_->get_instance_info(instance_name);
  if (instance_info.name.empty()) {
    LOG(ERROR)
        << "Failed to get instance info from master server, instance name: "
        << instance_name;
    return false;
  }

  remote_instances_info_[instance_name] = instance_info;
  return true;
}

proto::DisaggPDService_Stub* DisaggPDScheduler::create_rpc_channel(
    const std::string& instance_name,
    const std::string& decode_rpc_address) {
  std::lock_guard<std::mutex> lock(instance_channel_map_mutex_);

  // When decode_rpc_address is provided, use it as the cache key to
  // distinguish channels to different models on the same instance.
  std::string channel_key = decode_rpc_address.empty()
                                ? instance_name
                                : decode_rpc_address;
  auto it = instance_channel_map_.find(channel_key);
  if (it == instance_channel_map_.end()) {
    // Always populate remote_instances_info_ for KV cache transfer
    // (cluster_ids, addrs, k_cache_ids, v_cache_ids, dp_size).
    if (remote_instances_info_.find(instance_name) ==
        remote_instances_info_.end()) {
      if (!check_remote_instance_info(instance_name)) {
        LOG(ERROR) << "Check remote instance info failed, instance name: "
                   << instance_name;
        return nullptr;
      }
    }

    std::string target_address;
    if (!decode_rpc_address.empty()) {
      target_address = decode_rpc_address;
      LOG(INFO) << "Create rpc channel using routing-provided address: "
                << target_address;
    } else {
      target_address = remote_instances_info_[instance_name].rpc_address;
      LOG(INFO) << "Create rpc channel to instance: " << instance_name;
    }

    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms = FLAGS_rpc_channel_timeout_ms;
    options.max_retry = 3;
    std::string load_balancer = "";
    if (channel->Init(target_address.c_str(),
                      load_balancer.c_str(),
                      &options) != 0) {
      LOG(ERROR) << "Fail to initialize channel for " << target_address;
      if (decode_rpc_address.empty()) {
        remote_instances_info_.erase(instance_name);
      }
      delete channel;
      return nullptr;
    }

    proto::DisaggPDService_Stub* stub =
        new proto::DisaggPDService_Stub(channel);
    instance_channel_map_[channel_key] = stub;
    return stub;
  }

  return it->second;
}

void DisaggPDScheduler::start_rpc_server() {
  const uint16_t disagg_pd_port = options_.disagg_pd_port().value_or(
      static_cast<uint16_t>(FLAGS_disagg_pd_port));
  std::unique_ptr<DisaggPDService> service =
      std::make_unique<DisaggPDService>(this, engine_);
  auto rpc_server =
      ServerRegistry::get_instance().register_server(server_name_);
  if (!rpc_server->start(std::move(service), disagg_pd_port)) {
    LOG(ERROR) << "Failed to start brpc disagg pd server on port "
               << disagg_pd_port;
    return;
  }
}

void DisaggPDScheduler::step(const absl::Duration& timeout) {
  ContinuousScheduler::step(timeout);
  // send first generation token to decode instance
  if (options_.instance_role() != InstanceRole::DECODE && last_step_prefill_) {
    prefill_send_first_generation();
  }
}

bool DisaggPDScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  kv_cache_manager_->prefetch_from_storage(request);

  if (request->offline()) {
    // offline request, push to offline queue
    prefill_request_queue_offline_.enqueue(request);
    return true;
  }
  // push and wait
  prefill_request_queue_.enqueue(request);

  return true;
}

// prefill send new request to remote instance
void DisaggPDScheduler::dispatch_requests() {
  while (true) {
    const auto timeout = std::chrono::milliseconds(100);
    // Wait for online request until timeout.
    // If timeout, try to get offline request once. If no offline request,
    // continue to wait for online request. This can avoid offline request
    // blocking online request for too long time.
    std::shared_ptr<Request> request;
    if (!prefill_request_queue_.wait_dequeue_timed(request, timeout)) {
      if (!prefill_request_queue_offline_.try_dequeue(request)) {
        continue;
      }
    }

    if (request == nullptr) {
      // nullptr is a signal to exit
      break;
    }

    JANUS_DLOG(INFO) << "[DIAG] dispatch_requests: dequeued request "
              << request->request_id()
              << ", decode_address=" << request->state().decode_address;

    if (request->state().decode_address.empty()) {
      // No decode address provided to the prefill instance, just finish the
      // request.
      response_processor_->process_failed_request(
          request,
          {StatusCode::INVALID_ARGUMENT,
           "No decode address provided to the prefill instance"});
      continue;
    }

    // Local PD: decode on same instance, skip RPC and Mooncake transfer.
    // janus-service expects decode tokens via RPC (handle_generation), not
    // via the HTTP response from the prefill instance.  Close the HTTP
    // connection and redirect all token output through the RPC path.
    if (request->state().decode_address ==
        xservice_client_->get_instance_name()) {
      JANUS_DLOG(INFO) << "[DIAG] Local PD detected for request "
                << request->request_id()
                << ", closing HTTP and switching to RPC path";
      request->state().local_pd = true;

      // Notify the rate limiter that the prefill slot is released, but do
      // NOT set finished=true — that would cause send_delta_to_client_brpc
      // to write "data: [DONE]" into the HTTP stream, which janus-service
      // would forward to the client as a spurious completion marker.
      if (request->state().output_func) {
        RequestOutput done_output;
        done_output.request_id = request->request_id();
        done_output.service_request_id = request->service_request_id();
        done_output.target_xservice_addr = request->source_xservice_addr();
        done_output.finished_on_prefill_instance = true;
        done_output.finished = false;
        request->state().output_func(done_output);
      }

      // The original output_func lambda holds the last shared_ptr to the
      // CompletionCall (brpc stream object).  Overwriting it destroys the
      // CompletionCall, but state().call_ still holds a raw Call* to it.
      // Clear it to prevent update_connection_status() from calling the
      // pure-virtual is_disconnected() on a dangling pointer.
      // The PA (ProgressiveAttachment) is released when CompletionCall is
      // destroyed, which closes the HTTP connection without writing [DONE].
      request->state().call_ = std::nullopt;

      request->state().output_func =
          [this](const RequestOutput& output) -> bool {
        if (xservice_client_ == nullptr) return false;
        auto return_status = xservice_client_->generations({output});
        return !return_status.empty() && return_status[0];
      };
      request->state().outputs_func =
          [this](const std::vector<RequestOutput>& outputs) {
            if (xservice_client_ == nullptr) {
              return std::vector<bool>(outputs.size(), false);
            }
            return xservice_client_->generations(outputs);
          };

      request_queue_.write(request);
      continue;
    }

    std::vector<std::shared_ptr<Request>> requests;
    requests.emplace_back(request);
    std::string selected_instance = request->state().decode_address;
    proto::DisaggPDService_Stub* stub = create_rpc_channel(
        selected_instance, request->state().decode_rpc_address);
    if (stub == nullptr) {
      response_processor_->process_failed_request(
          request, {StatusCode::UNKNOWN, "Fail to create rpc channel"});
      continue;
    }

    // NOTE: TODO: maybe we need to support batch disatch
    // later, this meybe decrease the communication cost.
    // currently we only support one request per dispatch.

    // TODO: try to get a batch request.

    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      for (auto& req : requests) {
        req_to_channel_map_[req->request_id()] = stub;
      }
    }

    // TODO: send the request to the selected D instance
    // Send 'DisaggRequests' and recv 'DisaggResponses'
    janus::proto::DisaggRequests reqs;
    janus::proto::DisaggResponses resps;
    // prefill name (ID)
    reqs.set_prefill_name(xservice_client_->get_instance_name());
    reqs.mutable_reqs()->Reserve(requests.size());
    // currently we only support one request once.
    for (size_t i = 0; i < requests.size(); ++i) {
      // proto::DisaggRequest req;
      auto req = reqs.mutable_reqs()->Add();
      req->set_req_id(requests[i]->request_id());
      req->set_service_req_id(requests[i]->service_request_id());
      req->set_source_xservice_addr(requests[i]->source_xservice_addr());
      req->set_tokens_num(requests[i]->state().prompt_tokens.size());
      req->set_prompt(requests[i]->state().prompt);
      ADD_VECTOR_TO_PROTO(req->mutable_prompt_tokens(),
                          requests[i]->state().prompt_tokens);
      req->set_stream(requests[i]->state().stream);
      req->set_x_request_id(requests[i]->x_request_id());
      req->set_x_request_time(requests[i]->x_request_time());
      req->set_seq_capacity(requests[i]->state().seq_capacity);
      req->set_max_tokens(
          requests[i]->state().stopping_checker.get_max_generated_tokens());
      req->set_max_context_len(
          requests[i]->state().stopping_checker.get_max_context_len());
      req->set_ignore_eos(
          requests[i]->state().stopping_checker.get_ignore_eos());
      req->set_eos_token_id(
          requests[i]->state().stopping_checker.get_eos_token());
      if (requests[i]->state().stopping_checker.get_stop_tokens().size() > 0) {
        ADD_VECTOR_TO_PROTO(
            req->mutable_stop_token_ids(),
            requests[i]->state().stopping_checker.get_stop_tokens());
      }
      if (requests[i]->state().stopping_checker.get_stop_sequences().size() >
          0) {
        for (auto& stop_sequence :
             requests[i]->state().stopping_checker.get_stop_sequences()) {
          // proto::StopSequence proto_seq;
          auto proto_seq = req->mutable_stop_sequences()->Add();
          ADD_VECTOR_TO_PROTO(proto_seq->mutable_seq_tokens(), stop_sequence);
          //*req->mutable_stop_sequences()->Add() = proto_seq;
        }
      }
      req->set_n(requests[i]->state().n);
      req->set_best_of(requests[i]->state().best_of);
      req->set_frequency_penalty(
          requests[i]->state().sampling_param.frequency_penalty);
      req->set_presence_penalty(
          requests[i]->state().sampling_param.presence_penalty);
      req->set_repetition_penalty(
          requests[i]->state().sampling_param.repetition_penalty);
      req->set_temperature(requests[i]->state().sampling_param.temperature);
      req->set_top_p(requests[i]->state().sampling_param.top_p);
      req->set_top_k(requests[i]->state().sampling_param.top_k);
      req->set_logprobs(requests[i]->state().sampling_param.logprobs);
      req->set_top_logprobs(requests[i]->state().sampling_param.top_logprobs);
      req->set_is_embeddings(requests[i]->state().sampling_param.is_embeddings);
      req->set_echo(requests[i]->state().echo);
      req->set_skip_special_tokens(requests[i]->state().skip_special_tokens);
      //*reqs.mutable_reqs()->Add() = req;
    }
    std::vector<std::string> device_ips;
    std::vector<uint16_t> ports;
    engine_->get_device_info(device_ips, ports);
    reqs.mutable_cluster_infos()->mutable_cluster_ids()->Add(
        instance_info_.cluster_ids.begin(), instance_info_.cluster_ids.end());
    reqs.mutable_cluster_infos()->mutable_addrs()->Add(
        instance_info_.addrs.begin(), instance_info_.addrs.end());
    reqs.mutable_cluster_infos()->mutable_device_ips()->Add(device_ips.begin(),
                                                            device_ips.end());
    reqs.mutable_cluster_infos()->mutable_ports()->Add(ports.begin(),
                                                       ports.end());
    reqs.mutable_cluster_infos()->set_dp_size(options_.dp_size());

    // TODO: sync rpc here currently
    brpc::Controller cntl;
    JANUS_DLOG(INFO) << "[DIAG] Sending AddNewRequests RPC to " << selected_instance;
    stub->AddNewRequests(&cntl, &reqs, &resps, nullptr);
    JANUS_DLOG(INFO) << "[DIAG] AddNewRequests RPC returned, failed=" << cntl.Failed();
    if (cntl.Failed()) {
      LOG(ERROR) << "Failed to add new requests to decode instance : "
                 << selected_instance << ", error text : " << cntl.ErrorText();
      for (auto& request : requests) {
        response_processor_->process_failed_request(
            request,
            {StatusCode::UNKNOWN,
             "Failed to add new requests to decode instance"});

        {
          std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
          req_to_channel_map_.erase(request->request_id());
        }
      }
      continue;
    }

    // check reqs which can not dispatch to D instance,
    // and push back to prefill_request_queue_
    JANUS_DLOG(INFO) << "[DIAG] resps.resps().size()=" << resps.resps().size()
              << ", requests.size()=" << requests.size();
    CHECK_EQ(requests.size(), resps.resps().size())
        << "selected_instance : " << selected_instance;
    // insert instance name to linked_instance_
    {
      std::lock_guard<std::mutex> lock(linked_instances_mutex_);
      linked_instance_.emplace(selected_instance);
    }
    JANUS_DLOG(INFO) << "[DIAG] linked_instance_ updated, entering response loop";
    for (size_t i = 0; i < requests.size(); ++i) {
      JANUS_DLOG(INFO) << "[DIAG] response[" << i << "] status_code="
                << resps.resps()[i].status_code();
      if (resps.resps()[i].status_code() != 200) {
        JANUS_DLOG(WARNING) << "[DIAG] status_code != 200, pushing back to prefill queue";
        // push back to prefill_request_queue_
        if (requests[i]->offline()) {
          prefill_request_queue_offline_.enqueue(requests[i]);
        } else {
          prefill_request_queue_.enqueue(requests[i]);
        }

      } else {
        JANUS_DLOG(INFO) << "[DIAG] status_code=200, setting up TransferKVInfo"
                  << ", sequences.size=" << requests[i]->sequences().size();
        for (auto& sequence : requests[i]->sequences()) {
          TransferKVInfo info;
          info.request_id = requests[i]->request_id();
          for (auto& bid : resps.resps()[i].blocks_ids()) {
            info.remote_blocks_ids.emplace_back(bid);
          }
          info.dp_rank = resps.resps()[i].dp_rank();
          // TODO: remote_instances_info_ is not multi-thread safe.
          info.remote_instance_info = remote_instances_info_[selected_instance];

          // XTensor mode: save destination offsets from D-node
          const auto& resp = resps.resps()[i];
          JANUS_DLOG(INFO) << "[DIAG] xtensor_layer_offsets_size="
                    << resp.xtensor_layer_offsets_size()
                    << ", remote_blocks_ids.size="
                    << info.remote_blocks_ids.size();
          if (resp.xtensor_layer_offsets_size() > 0) {
            info.dst_xtensor_layer_offsets.reserve(
                resp.xtensor_layer_offsets_size());
            for (const auto& layer_offsets : resp.xtensor_layer_offsets()) {
              XTensorLayerOffsets layer;
              layer.k_offsets.assign(layer_offsets.k_offsets().begin(),
                                     layer_offsets.k_offsets().end());
              layer.v_offsets.assign(layer_offsets.v_offsets().begin(),
                                     layer_offsets.v_offsets().end());
              layer.index_offsets.assign(layer_offsets.index_offsets().begin(),
                                         layer_offsets.index_offsets().end());
              info.dst_xtensor_layer_offsets.emplace_back(std::move(layer));
            }
            info.dst_xtensor_block_bytes.k_block_bytes =
                resp.xtensor_block_bytes().k_block_bytes();
            info.dst_xtensor_block_bytes.v_block_bytes =
                resp.xtensor_block_bytes().v_block_bytes();
            info.dst_xtensor_block_bytes.index_block_bytes =
                resp.xtensor_block_bytes().index_block_bytes();
            JANUS_DLOG(INFO) << "[DIAG] Received XTensor offsets, num_layers="
                      << info.dst_xtensor_layer_offsets.size();
          } else {
            JANUS_DLOG(WARNING) << "[DIAG] No XTensor offsets from D-node!";
          }

          JANUS_DLOG(INFO) << "[DIAG] Calling set_transfer_kv_info";
          sequence->kv_state().set_transfer_kv_info(std::move(info));
          JANUS_DLOG(INFO) << "[DIAG] set_transfer_kv_info done";
        }

        // push to request_queue_, and will be executed by engine.
        JANUS_DLOG(INFO) << "[DIAG] Request dispatched, writing to request_queue_"
                  << ", remote_instance_info.cluster_ids.size="
                  << remote_instances_info_[selected_instance].cluster_ids.size()
                  << ", dp_size="
                  << remote_instances_info_[selected_instance].dp_size;
        request_queue_.write(requests[i]);
        JANUS_DLOG(INFO) << "[DIAG] request_queue_.write done for request "
                  << requests[i]->request_id();
      }
    }
    JANUS_DLOG(INFO) << "[DIAG] dispatch loop iteration complete";
  }
}

void DisaggPDScheduler::prefill_send_first_generation() {
  if (running_sequences_.size() == 0) {
    return;
  }

  std::vector<std::shared_ptr<Request>> requests;
  std::vector<std::shared_ptr<Request>> non_stream_requests;
  requests.reserve(running_requests_.size());
  non_stream_requests.reserve(running_requests_.size());
  for (size_t i = 0; i < running_requests_.size(); ++i) {
    auto request = running_requests_[i];
    // Check if the request is a recently completed prefill request
    if (request->sequences()[0]->num_generated_tokens() == 1) {
      // Local PD: keep the request in running_requests_ so the normal
      // scheduler flow transitions it to decode via running_queue_.
      // No RPC, no KV deallocation, no Mooncake transfer needed.
      if (request->state().local_pd) {
        JANUS_DLOG(INFO) << "[DIAG] Local PD request " << request->request_id()
                  << " completed prefill, staying in running for decode";
        continue;
      }
      request->log_statistic(request->elapsed_seconds());
      requests.emplace_back(request);
      if (!request->state().stream) {
        non_stream_requests.emplace_back(request);
      }
      running_requests_[i] = nullptr;
    }
  }
  // call non_stream_request's callback in P instance when its prefill ends
  if (!non_stream_requests.empty()) {
    response_processor_->process_completed_requests(non_stream_requests);
  }

  // No prefill request needs to be transferred to decode.
  if (requests.size() == 0) {
    return;
  }

  JANUS_DLOG(INFO) << "[DIAG] prefill_send_first_generation: sending "
            << requests.size() << " requests to decode";

  prefill_threadpool_.schedule([this,
                                requests = std::move(requests)]() mutable {
    // process_stream_request() dispatches the first-token write to the
    // response_threadpool_ asynchronously.  We must wait for that write
    // to complete before releasing the HTTP ProgressiveAttachment,
    // otherwise the first token is lost (data race / empty PA).
    response_processor_->wait_completion();

    // Now safe: release the PA for streaming requests.  This completes
    // the HTTP response so that handle_first_response on janus-service
    // forwards the first token to the client BEFORE the decode instance
    // starts generating tokens (via the FirstGeneration RPC below).
    for (auto& request : requests) {
      if (request->state().stream) {
        request->state().call_ = std::nullopt;
        request->state().output_func = nullptr;
        request->state().outputs_func = nullptr;
      }
    }

    // send request first token to remote instance
    // TODO: here we only support one sequence for now.
    for (auto& request : requests) {
      // TODO: support batch request later
      proto::DisaggGenerationsRequests gens;
      auto gen = gens.mutable_multi_gens()->Add();
      gen->set_req_id(request->request_id());
      gen->set_x_request_id(request->x_request_id());
      gen->set_x_request_time(request->x_request_time());
      if (request->sequences()[0]->first_token().has_value()) {
        auto token = gen->mutable_tokens()->Add();
        token->set_token_id(
            request->sequences()[0]->first_token().value().token_id);
        token->set_time_to_first_token_latency_seconds(
            request->sequences()[0]->time_to_first_token_latency_seconds());
        if (request->sequences()[0]
                ->first_token()
                .value()
                .token_logprob.has_value()) {
          token->set_logprob(request->sequences()[0]
                                 ->first_token()
                                 .value()
                                 .token_logprob.value());
          token->set_has_logprob(true);
        } else {
          token->set_has_logprob(false);
        }
        ADD_VECTOR_TO_PROTO(
            token->mutable_top_tokens(),
            request->sequences()[0]->first_token().value().token_top_tokens);
        ADD_VECTOR_TO_PROTO(
            token->mutable_top_logprobs(),
            request->sequences()[0]->first_token().value().token_top_logprobs);
      }
      gen->set_kv_cache_transfer_mode(options_.kv_cache_transfer_mode());
      if (options_.kv_cache_transfer_mode() == "PULL") {
        ADD_VECTOR_TO_PROTO(gen->mutable_cluster_ids(),
                            instance_info_.cluster_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_addrs(), instance_info_.addrs);
        ADD_VECTOR_TO_PROTO(gen->mutable_k_cache_ids(),
                            instance_info_.k_cache_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_v_cache_ids(),
                            instance_info_.v_cache_ids);

        const auto blocks = request->sequences()[0]->kv_state().kv_blocks();
        std::vector<uint64_t> block_ids;
        block_ids.reserve(blocks.size());
        for (const auto& block : blocks) {
          block_ids.push_back(block.id());
        }
        ADD_VECTOR_TO_PROTO(gen->mutable_block_ids(), block_ids);
        gen->set_dp_size(instance_info_.dp_size);
        gen->set_dp_rank(request->sequences()[0]->dp_rank());
      }

      // send first gens to remote instance
      proto::DisaggPDService_Stub* stub = nullptr;
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        // now we only support one request once.
        stub = req_to_channel_map_[request->request_id()];
      }

      // TODO: Async call later
      proto::Status resp;
      brpc::Controller cntl;
      JANUS_DLOG(INFO) << "[DIAG] Sending FirstGeneration RPC for request "
                << request->request_id();
      stub->FirstGeneration(&cntl, &gens, &resp, nullptr);
      JANUS_DLOG(INFO) << "[DIAG] FirstGeneration RPC returned, failed="
                << cntl.Failed() << ", ok=" << resp.ok();

      if (cntl.Failed() || !resp.ok()) {
        LOG(ERROR) << "Failed to send first generation to decode instance : "
                   << request->state().decode_address
                   << ", error text : " << cntl.ErrorText()
                   << ", response status: " << resp.ok();
      }

      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        req_to_channel_map_.erase(request->request_id());
      }
      kv_cache_manager_->deallocate(request.get());
    }
  });
}

// request is received from prefill
bool DisaggPDScheduler::decode_schedule(
    std::shared_ptr<Request>& request,
    const std::string& prefill_instance_name) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    if (received_request_map_.find(request->request_id()) !=
        received_request_map_.end()) {
      LOG(ERROR) << "Decode receive duplicate request_id from prefill: "
                 << request->request_id();
      kv_cache_manager_->deallocate(request.get());
      return false;
    }
    received_request_map_[request->request_id()] = request;
    instance_to_received_requests_map_[prefill_instance_name].insert(
        request->request_id());
    request_to_instance_map_[request->request_id()] = prefill_instance_name;
  }

  return true;
}

bool DisaggPDScheduler::decode_recv_first_generation(
    const std::string& req_id,
    int64_t token_id,
    bool has_logprob,
    float logprob,
    double time_to_first_token_latency_seconds,
    std::vector<int64_t> top_tokens,
    std::vector<float> top_logprobs,
    const std::string& kv_cache_transfer_mode,
    std::vector<uint64_t> src_cluster_ids,
    std::vector<std::string> src_addrs,
    std::vector<int64_t> src_k_cache_ids,
    std::vector<int64_t> src_v_cache_ids,
    std::vector<uint64_t> src_block_ids,
    int32_t src_dp_size,
    int32_t src_dp_rank) {
  // push to request_queue_, and will be executed by engine.
  std::shared_ptr<Request> request = nullptr;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = received_request_map_.find(req_id);
    if (it == received_request_map_.end()) {
      LOG(ERROR) << "Failed to find request, request id: " << req_id;
      return false;
    }
    request = it->second;
    received_request_map_.erase(it);

    auto inst_it = request_to_instance_map_.find(req_id);
    if (inst_it != request_to_instance_map_.end()) {
      instance_to_received_requests_map_[inst_it->second].erase(req_id);
      request_to_instance_map_.erase(inst_it);
    }
  }

  Token first_token(token_id);
  if (has_logprob) {
    first_token.logprob = logprob;
    if (!top_tokens.empty() && !top_logprobs.empty()) {
      // NOTE: slice vector here, to avoid copy
      // so we need keep the vector `top_tokens` and `top_logprobs` lifetime
      first_token.top_tokens = top_tokens;
      first_token.top_logprobs = top_logprobs;
    }
  }
  // Enable checking whether to skip the prefill token
  if (request->state().stream) {
    request->sequences()[0]->enable_checking_prefill_token();
  }

  // update latency metrics
  request->sequences()[0]->set_time_to_first_token_latency_seconds(
      time_to_first_token_latency_seconds);
  // update latest_generate_time_ for sequence
  request->sequences()[0]->tbt(
      request->created_time() +
      absl::Seconds(time_to_first_token_latency_seconds));

  // TODO: we only support one sequence for currently.
  if (enable_schedule_overlap()) {
    Token fake_token(-1);
    request->sequences()[0]->append_token(fake_token);
    request->sequences()[0]->update_last_step_token(first_token);
  } else {
    request->sequences()[0]->append_token(first_token);
  }

  // pull kv cache
  if (kv_cache_transfer_mode == "PULL") {
    const auto blocks = request->sequences()[0]->kv_state().kv_blocks();
    std::vector<uint64_t> dst_block_ids;
    dst_block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      dst_block_ids.push_back(block.id());
    }

    int32_t dst_dp_rank = request->sequences()[0]->dp_rank();
    engine_->pull_kv_blocks(src_dp_size,
                            src_dp_rank,
                            src_cluster_ids,
                            src_addrs,
                            src_k_cache_ids,
                            src_v_cache_ids,
                            src_block_ids,
                            dst_dp_rank,
                            dst_block_ids);
  }

  request_queue_.write(request);
  return true;
}

bool DisaggPDScheduler::try_allocate(Sequence* sequence) {
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  if (kv_cache_manager_->kv_cache_utilization() <
      FLAGS_prefill_scheduling_memory_usage_threshold) {
    return kv_cache_manager_->try_allocate(sequence);
  } else {
    return false;
  }
}

void DisaggPDScheduler::update_token_latency_metrics(
    std::vector<Sequence*>& sequences) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  const auto now = absl::Now();
  for (Sequence* sequence : sequences) {
    if (sequence->is_chunked_prefill_stage() ||
        sequence->last_token_handled()) {
      continue;
    }
    int64_t tbt_milliseconds = sequence->tbt(now);
    if (sequence->is_first_token()) {
      HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                        tbt_milliseconds);
      sequence->set_time_to_first_token_latency_seconds(
          static_cast<double>(tbt_milliseconds) / 1000);
      recent_ttft_.emplace_back(tbt_milliseconds);
    } else {
      HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
      recent_tbt_.emplace_back(tbt_milliseconds);
    }
  }
}

void DisaggPDScheduler::get_latency_metrics(std::vector<int64_t>& ttft,
                                            std::vector<int64_t>& tbt) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);
  ttft = std::move(recent_ttft_);
  tbt = std::move(recent_tbt_);
}

bool DisaggPDScheduler::link_instance(
    const std::string& instance_name,
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<std::string>& device_ips,
    const std::vector<uint16_t>& ports,
    const int32_t dp_size) {
  std::lock_guard<std::mutex> lock(linked_instances_mutex_);
  if (!engine_->link_cluster(cluster_ids, addrs, device_ips, ports, dp_size)) {
    LOG(ERROR) << "Link instance failed, instance_name: " << instance_name;
    return false;
  }
  LOG(INFO) << "Successfully linked instance, instance_name: " << instance_name;
  linked_instance_.emplace(instance_name);
  return true;
}

bool DisaggPDScheduler::unlink_instance(
    const std::string& instance_name,
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<std::string>& device_ips,
    const std::vector<uint16_t>& ports,
    const int32_t dp_size) {
  // Clear received requests from this instance
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = instance_to_received_requests_map_.find(instance_name);
    if (it != instance_to_received_requests_map_.end()) {
      for (const auto& req_id : it->second) {
        received_request_map_.erase(req_id);
        request_to_instance_map_.erase(req_id);
      }
      instance_to_received_requests_map_.erase(it);
    }
  }

  std::lock_guard<std::mutex> lock(linked_instances_mutex_);
  if (!engine_->unlink_cluster(
          cluster_ids, addrs, device_ips, ports, dp_size)) {
    LOG(ERROR) << "Unlink instance failed, instance_name: " << instance_name;
    return false;
  }
  LOG(INFO) << "Successfully unlinked instance, instance_name: "
            << instance_name;
  linked_instance_.erase(instance_name);
  return true;
}

}  // namespace janus
