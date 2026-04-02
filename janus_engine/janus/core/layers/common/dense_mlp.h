#pragma once

#include <torch/torch.h>

#include "activation.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "linear.h"

namespace janus {
namespace layer {

class DenseMLPImpl : public torch::nn::Module {
 public:
  DenseMLPImpl() = default;
  DenseMLPImpl(int64_t hidden_size,
               int64_t intermediate_size,
               bool is_gated,
               bool has_bias,
               const std::string& hidden_act,
               bool enable_result_reduction,
               const QuantArgs& quant_args,
               ProcessGroup* process_group,
               const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& hidden_states);

  void load_state_dict(const StateDict& state_dict);
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& gate_up_name,
                       const std::string& down_name);

  // Get FP8 input scale from gate_up_proj for fused RMSNorm+FP8 quantization
  std::optional<torch::Tensor> get_fp8_input_scale() const;

 private:
  bool is_gated_;
  int64_t intermediate_size_;
  ProcessGroup* process_group_;
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};
  Activation act_{nullptr};
  bool is_smoothquant_;
  std::string hidden_act_;
};
TORCH_MODULE(DenseMLP);

}  // namespace layer
}  // namespace janus