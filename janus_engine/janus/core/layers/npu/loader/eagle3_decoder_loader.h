#pragma once

#include <map>
#include <vector>

#include "core/layers/npu/npu_base_layer.h"

namespace janus {
namespace layer {

class Eagle3DecoderLoader : public BaseLoader {
 public:
  Eagle3DecoderLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 protected:
  TransposeType check_transpose(at::Tensor& tensor);
  int device_id_;
};

}  // namespace layer
}  // namespace janus
