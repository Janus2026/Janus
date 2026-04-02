#include "ilu_ops_api.h"

using namespace ixformer;

namespace janus::kernel::ilu {

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode) {
  if (act_mode == "silu") {
    infer::silu_and_mul(input, out);
  } else {
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only support silu, gelu, gelu_tanh";
  }
}
}  // namespace janus::kernel::ilu
