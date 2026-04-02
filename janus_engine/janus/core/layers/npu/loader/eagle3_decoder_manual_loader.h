#pragma once

#include <map>
#include <vector>

#include "base_manual_loader.h"
#include "core/layers/npu/npu_base_layer.h"

namespace janus {
namespace layer {

class Eagle3DecoderManualLoader : public BaseManualLoader {
 public:
  Eagle3DecoderManualLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;
  void merge_and_move_pinned_host();

 protected:
  int device_id_;
  TransposeType check_transpose(at::Tensor& tensor);
  void merge_host_at_weights();
};

}  // namespace layer
}  // namespace janus