#include "rpc_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/closure_guard.h>

#include "common/types.h"
#include "common/utils.h"
#include "common/janus/status.h"
#include "scheduler/scheduler.h"

namespace janus_service {

JanusRpcServiceImpl::JanusRpcServiceImpl(const Options& options,
                                       Scheduler* scheduler)
    : options_(options), scheduler_(scheduler) {
  enable_decode_response_to_service_ =
      utils::get_bool_env("ENABLE_DECODE_RESPONSE_TO_SERVICE", false);
}

JanusRpcServiceImpl::~JanusRpcServiceImpl() { scheduler_->exited(); }

void JanusRpcServiceImpl::heartbeat(const proto::HeartbeatRequest* req) {
  scheduler_->handle_instance_heartbeat(req);
}

InstanceMetaInfo JanusRpcServiceImpl::get_instance_info(
    const std::string& instance_name) {
  return scheduler_->get_instance_info(instance_name);
}

std::vector<std::string> JanusRpcServiceImpl::get_static_decode_list(
    const std::string& instance_name) {
  return scheduler_->get_static_decode_list(instance_name);
}

std::vector<std::string> JanusRpcServiceImpl::get_static_prefill_list(
    const std::string& instance_name) {
  return scheduler_->get_static_prefill_list(instance_name);
}

bool JanusRpcServiceImpl::handle_generation(
    const llm::RequestOutput& request_output) {
  return scheduler_->handle_generation(request_output);
}

ServiceConfig JanusRpcServiceImpl::get_config() {
  return ServiceConfig(enable_decode_response_to_service_);
}

bool JanusRpcServiceImpl::wakeup_model(const std::string& instance_name,
                                      const std::string& model_id,
                                      InstanceTag tag) {
  scheduler_->wakeup_model(instance_name, model_id, tag);
  return true;
}

bool JanusRpcServiceImpl::sleep_model(const std::string& instance_name,
                                     const std::string& model_id) {
  scheduler_->sleep_model(instance_name, model_id);
  return true;
}

JanusRpcService::JanusRpcService(const Options& options, Scheduler* scheduler) {
  janus_rpc_service_impl_ =
      std::make_unique<JanusRpcServiceImpl>(options, scheduler);
}

JanusRpcService::~JanusRpcService() {}

void JanusRpcService::Hello(google::protobuf::RpcController* cntl_base,
                           const proto::Empty* req,
                           proto::Status* resp,
                           google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  resp->set_ok(true);
}

void JanusRpcService::GetInstanceInfo(google::protobuf::RpcController* cntl_base,
                                     const proto::InstanceID* req,
                                     proto::InstanceMetaInfo* resp,
                                     google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  InstanceMetaInfo metainfo =
      janus_rpc_service_impl_->get_instance_info(req->name());
  resp->set_name(metainfo.name);
  resp->set_rpc_address(metainfo.rpc_address);
  if (metainfo.type == InstanceType::PREFILL) {
    resp->set_type(proto::InstanceType::PREFILL);
  } else if (metainfo.type == InstanceType::DECODE) {
    resp->set_type(proto::InstanceType::DECODE);
  } else if (metainfo.type == InstanceType::MIX) {
    resp->set_type(proto::InstanceType::MIX);
  } else {
    resp->set_type(proto::InstanceType::DEFAULT);
  }
  for (auto& cluster_id : metainfo.cluster_ids) {
    *(resp->mutable_cluster_ids()->Add()) = cluster_id;
  }
  for (auto& addr : metainfo.addrs) {
    *(resp->mutable_addrs()->Add()) = addr;
  }
  for (auto& k_cache_id : metainfo.k_cache_ids) {
    *(resp->mutable_k_cache_ids()->Add()) = k_cache_id;
  }
  for (auto& v_cache_id : metainfo.v_cache_ids) {
    *(resp->mutable_v_cache_ids()->Add()) = v_cache_id;
  }
  resp->set_dp_size(metainfo.dp_size);
}

void JanusRpcService::Heartbeat(google::protobuf::RpcController* cntl_base,
                               const proto::HeartbeatRequest* req,
                               proto::Status* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  janus_rpc_service_impl_->heartbeat(req);
  resp->set_ok(true);
}

void JanusRpcService::GetStaticDecodeList(
    google::protobuf::RpcController* cntl_base,
    const proto::InstanceID* req,
    proto::InstanceIDs* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<std::string> decode_list =
      janus_rpc_service_impl_->get_static_decode_list(req->name());
  for (auto& d : decode_list) {
    *(resp->mutable_names()->Add()) = std::move(d);
  }
}

void JanusRpcService::GetStaticPrefillList(
    google::protobuf::RpcController* cntl_base,
    const proto::InstanceID* req,
    proto::InstanceIDs* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<std::string> prefill_list =
      janus_rpc_service_impl_->get_static_prefill_list(req->name());
  for (auto& p : prefill_list) {
    *(resp->mutable_names()->Add()) = std::move(p);
  }
}

void JanusRpcService::Generations(google::protobuf::RpcController* cntl_base,
                                 const proto::DisaggStreamGenerations* req,
                                 proto::StatusSet* resp,
                                 google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  // TODO: use threadpool here
  //
  for (auto& request : req->gens()) {
    // convert proto request to `RequestOutput`
    llm::RequestOutput request_output;
    request_output.request_id = request.req_id();
    request_output.service_request_id = request.service_req_id();
    if (request.has_gen_status()) {
      request_output.status = llm::Status(
          static_cast<llm::StatusCode>(request.gen_status().status_code()),
          request.gen_status().status_msg());
    }
    if (request.has_usage()) {
      llm::Usage u;
      u.num_prompt_tokens = request.usage().num_prompt_tokens();
      u.num_generated_tokens = request.usage().num_generated_tokens();
      u.num_total_tokens = request.usage().num_total_tokens();
      request_output.usage = std::move(u);
    }
    request_output.finished = request.finished();
    for (auto& output : request.outputs()) {
      llm::SequenceOutput sequence_output;
      sequence_output.index = output.index();
      sequence_output.text = output.text();
      sequence_output.token_ids = std::vector<int32_t>(
          output.token_ids().begin(), output.token_ids().end());
      if (!output.finish_reason().empty()) {
        sequence_output.finish_reason = output.finish_reason();
      }
      if (output.logprobs().size() > 0) {
        std::vector<llm::LogProb> logprobs;
        for (auto& logprob : output.logprobs()) {
          llm::LogProb lp;
          lp.token = logprob.log_prob_data().token();
          lp.token_id = logprob.log_prob_data().token_id();
          lp.logprob = logprob.log_prob_data().logprob();
          lp.finished_token = logprob.log_prob_data().finished_token();
          if (logprob.top_logprobs().size() > 0) {
            std::vector<llm::LogProbData> top_logprobs;
            for (auto& top_logprob : logprob.top_logprobs()) {
              llm::LogProbData lpd;
              lpd.token = top_logprob.token();
              lpd.token_id = top_logprob.token_id();
              lpd.logprob = top_logprob.logprob();
              lpd.finished_token = top_logprob.finished_token();
              top_logprobs.emplace_back(std::move(lpd));
            }
            lp.top_logprobs = std::move(top_logprobs);
          }
          logprobs.emplace_back(std::move(lp));
        }
        sequence_output.logprobs = std::move(logprobs);
      }
      request_output.outputs.emplace_back(std::move(sequence_output));
    }
    resp->mutable_all_status()->Add()->set_ok(
        janus_rpc_service_impl_->handle_generation(request_output));
  }
}

void JanusRpcService::GetConfig(google::protobuf::RpcController* cntl_base,
                               const proto::Empty* req,
                               proto::ServiceConfig* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto config = janus_rpc_service_impl_->get_config();
  resp->set_enable_decode_response_to_service(
      config.enable_decode_response_to_service);
}

void JanusRpcService::AssignTag(google::protobuf::RpcController* cntl_base,
                               const proto::TagAssignRequest* req,
                               proto::TagAssignResponse* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  InstanceTag tag;
  switch (req->tag()) {
    case proto::TAG_NONE:
      tag = InstanceTag::NONE;
      break;
    case proto::TAG_NORMAL:
      tag = InstanceTag::NORMAL;
      break;
    case proto::TAG_PREFILL:
      tag = InstanceTag::PREFILL;
      break;
    case proto::TAG_DECODE:
      tag = InstanceTag::DECODE;
      break;
    default:
      resp->set_ok(false);
      resp->set_error_message("Invalid tag type");
      return;
  }

  LOG(INFO) << "AssignTag: instance=" << req->instance_name()
            << " model=" << req->model_id()
            << " tag=" << instance_tag_name(tag);

  bool ok = janus_rpc_service_impl_->wakeup_model(
      req->instance_name(), req->model_id(), tag);
  resp->set_ok(ok);
  if (!ok) {
    resp->set_error_message("Failed to wakeup model with tag");
  }
}

void JanusRpcService::RemoveTag(google::protobuf::RpcController* cntl_base,
                               const proto::TagAssignRequest* req,
                               proto::TagAssignResponse* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  LOG(INFO) << "RemoveTag: instance=" << req->instance_name()
            << " model=" << req->model_id();

  bool ok = janus_rpc_service_impl_->sleep_model(
      req->instance_name(), req->model_id());
  resp->set_ok(ok);
  if (!ok) {
    resp->set_error_message("Failed to sleep model");
  }
}

}  // namespace janus_service
