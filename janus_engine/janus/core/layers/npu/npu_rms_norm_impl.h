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
#include "loader/rms_norm_loader.h"
#include "loader/rms_norm_manual_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "janus_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "janus_atb_layers/core/include/atb_speed/base/model.h"
#include "janus_atb_layers/core/include/atb_speed/log.h"
#include "janus_atb_layers/core/include/atb_speed/utils/model_factory.h"

namespace janus {
namespace layer {

class NpuRMSNormImpl : public BaseLayer {
 public:
  explicit NpuRMSNormImpl(const ModelContext& context);

  ~NpuRMSNormImpl() override = default;

  torch::Tensor forward(torch::Tensor& x, int nodeId);

 private:
  int64_t init_layer() override;

  int64_t init_node(atb_speed::Model::Node& node,
                    atb::infer::RmsNormParam& param);

  void build_node_variant_pack(atb_speed::Model::Node& node, torch::Tensor& x);

  void param_from_args(atb::infer::RmsNormParam& param, const ModelArgs& args);

  atb_speed::Model::Node norm_node_;
  std::string model_name_;
  atb::infer::RmsNormParam norm_param_;
  atb::Tensor internal_tensors_;
};
TORCH_MODULE(NpuRMSNorm);

}  // namespace layer
}  // namespace janus
