#pragma once

#include <map>
#include <vector>

#include "base_manual_loader.h"
#include "core/layers/npu/npu_base_layer.h"
namespace janus {
namespace layer {

class Qwen2DecoderManualLoader : public BaseManualLoader {
 public:
  Qwen2DecoderManualLoader(uint64_t weight_count, const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  bool is_nz_format_tensor(int weight_index) override;

 protected:
  int device_id_;
  void merge_host_at_weights();
};

}  // namespace layer
}  // namespace janus