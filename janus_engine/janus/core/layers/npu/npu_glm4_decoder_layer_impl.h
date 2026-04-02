#pragma once
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "loader/glm4_decoder_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "janus_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "janus_atb_layers/core/include/atb_speed/base/model.h"
#include "janus_atb_layers/core/include/atb_speed/log.h"
#include "janus_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "janus_atb_layers/models/glm/layer/decoder_layer.h"

namespace janus {
namespace layer {

class NpuGlm4DecoderLayerImpl : public BaseLayer {
 public:
  explicit NpuGlm4DecoderLayerImpl(const ModelContext& context);

  ~NpuGlm4DecoderLayerImpl() override = default;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

 private:
  void param_from_args(atb_speed::chatglm::ChatglmLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill);

  void initialize_quantization_parameters(
      atb_speed::chatglm::ChatglmLayerParam& param);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::chatglm::ChatglmLayerParam& param);

  int64_t init_attn_mask();

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  std::string model_name_;
  atb_speed::chatglm::ChatglmLayerParam prefill_param_;
  atb_speed::chatglm::ChatglmLayerParam decode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;

  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;
};
TORCH_MODULE(NpuGlm4DecoderLayer);

}  // namespace layer
}  // namespace janus
