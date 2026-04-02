#pragma once

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace janus {
namespace layer {

// RollingWeightBuffer manages N contiguous device memory slots for rolling
// decoder layer weight loading. Each slot holds one decoder layer's weights.
// Slot index = layer_index % num_slots.
//
// Memory allocation strategy:
//   - enable_xtensor=true:  memory is allocated via XTensorAllocator
//                           (counted in weight pages budget)
//   - enable_xtensor=false: memory is allocated via aclrtMalloc
class RollingWeightBuffer {
 public:
  // num_slots: number of cached decoder layer slots(N =
  // rolling_load_num_cached_layers) storage_size: byte size of one decoder
  // layer's weights (aligned) model_id: XTensor model id
  //           (used when enable_xtensor=true; ignored otherwise)
  RollingWeightBuffer(int32_t num_slots,
                      size_t storage_size,
                      const std::string& model_id = "");
  ~RollingWeightBuffer();

  // Non-copyable
  RollingWeightBuffer(const RollingWeightBuffer&) = delete;
  RollingWeightBuffer& operator=(const RollingWeightBuffer&) = delete;

  // Returns the device pointer for slot (layer_index % num_slots).
  void* get_slot_ptr(int32_t layer_index) const;

  // Refresh rolling buffer base pointer from current weight allocation.
  // In xtensor mode this reserves/updates a new address in the current
  // weight region after wakeup.
  void refresh_address();

  size_t slot_size() const { return storage_size_; }
  int32_t num_slots() const { return num_slots_; }
  bool is_allocated() const { return base_ptr_ != nullptr; }

 private:
  int32_t num_slots_;
  size_t storage_size_;
  std::string model_id_;
  void* base_ptr_ = nullptr;
  bool use_xtensor_ =
      false;  // whether memory came from XTensor (not aclrtMalloc)
};

}  // namespace layer
}  // namespace janus
