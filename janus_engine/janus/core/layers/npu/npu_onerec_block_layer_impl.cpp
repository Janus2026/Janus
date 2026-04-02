#include "npu_onerec_block_layer_impl.h"

#include <glog/logging.h>

namespace janus {
namespace layer {

NpuOneRecBlockLayerImpl::NpuOneRecBlockLayerImpl(const ModelContext& context,
                                                 bool is_decoder,
                                                 int32_t layer_id)
    : device_(context.get_tensor_options().device()),
      is_decoder_(is_decoder),
      layer_id_(layer_id) {}

torch::Tensor NpuOneRecBlockLayerImpl::forward(torch::Tensor& hidden_states,
                                               torch::Tensor& attn_mask,
                                               KVCache& kv_cache,
                                               ModelInputParams& input_params,
                                               torch::Tensor* encoder_output,
                                               int32_t node_id,
                                               aclrtEvent* event,
                                               std::atomic<bool>* event_flag) {
  (void)attn_mask;
  (void)kv_cache;
  (void)node_id;
  (void)event;
  (void)event_flag;

  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec block requires onerec_params().";

  const bool is_prefill =
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;
  (void)is_prefill;

  if (encoder_output != nullptr && encoder_output->defined() &&
      encoder_output->device() != device_) {
    *encoder_output = encoder_output->to(device_);
  }

  if (hidden_states.device() != device_) {
    hidden_states = hidden_states.to(device_);
  }

  if (!is_decoder_ && hidden_states.dim() > 1) {
    return hidden_states.contiguous();
  }

  return hidden_states;
}

}  // namespace layer
}  // namespace janus
