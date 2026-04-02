#include "vlm_executor_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/request/mm_data_visitor.h"
#include "platform/device.h"

namespace janus {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      args_(args),
      device_(device),
      options_(options) {
  if (FLAGS_enable_graph) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options, Device::type_str());
  }
}

ForwardInput VlmExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

MMDict VlmExecutorImpl::encode(const ModelInputParams& params) {
  return dynamic_cast<CausalVLM*>(model_)->encode(params);
}

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  auto& mm_data = params.mm_data;
  EncoderInputGatherVisitor input_gather;
  mm_data.foreach (input_gather);
  CHECK(input_gather.finish(mm_data));
  mm_data.to(device_);

  auto embedding = encode(params);
  EncoderOutputScatterVisitor scatter(embedding);
  mm_data.foreach (scatter);
  CHECK(scatter.finish());

  EncoderEmbeddingGatherVisitor gather(device_);
  mm_data.foreach (gather);
  CHECK(gather.finish(mm_data));

  params.input_embedding = model_->get_input_embeddings(tokens, params);

  if (llm_executor_) {
    return llm_executor_->run(tokens, positions, kv_caches, params);
  }

  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace janus