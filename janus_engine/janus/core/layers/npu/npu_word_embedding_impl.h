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
#include "framework/model/model_input_params.h"
#include "loader/word_embedding_loader.h"
#include "loader/word_embedding_manual_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "janus_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "janus_atb_layers/core/include/atb_speed/base/model.h"
#include "janus_atb_layers/core/include/atb_speed/log.h"
#include "janus_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "janus_atb_layers/operations/fusion/embedding/word_embedding.h"

namespace janus {
namespace layer {

class NpuWordEmbeddingImpl : public BaseLayer {
 public:
  explicit NpuWordEmbeddingImpl(const ModelContext& context);

  ~NpuWordEmbeddingImpl() override = default;

  void param_from_args(atb_speed::common::WordEmbeddingParam& param,
                       const janus::ModelArgs& args,
                       const janus::ParallelArgs& parallel_args);

  int64_t init_layer();

  torch::Tensor forward(const torch::Tensor& x, int nodeId);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& x);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::WordEmbeddingParam& param);

  atb_speed::Model::Node embedding_node_;
  std::string modelName_;
  std::vector<at::Tensor> atOutTensors_;
  // std::string name_;
  atb_speed::common::WordEmbeddingParam embedding_param_;
  atb::Tensor internalTensors;
};
TORCH_MODULE(NpuWordEmbedding);

}  // namespace layer
}  // namespace janus
