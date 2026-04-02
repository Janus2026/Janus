#pragma once

#include <map>
#include <vector>

#include "base_manual_loader.h"
#include "core/layers/npu/npu_base_layer.h"

namespace janus {
namespace layer {

class Qwen3DecoderManualLoader : public BaseManualLoader {
 public:
  Qwen3DecoderManualLoader(uint64_t weight_count,
                           const ModelContext& context,
                           bool enableAddNorm);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  bool is_nz_format_tensor(int weight_index) override;

 protected:
  void merge_host_at_weights();
  at::Tensor at_placeholder_;
  bool enableAddNorm_;
  int rank_id_;
};

}  // namespace layer
}  // namespace janus