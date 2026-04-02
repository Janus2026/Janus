#pragma once

#include <acl/acl.h>

#include <atomic>
#include <vector>

namespace janus {

class NPULayerSynchronizerImpl {
 public:
  NPULayerSynchronizerImpl(const int64_t num_layers,
                           const int32_t timeout = -1);
  virtual ~NPULayerSynchronizerImpl();

  aclrtEvent* get_event(const int64_t layer_index);
  std::atomic<bool>* get_event_flag(const int64_t layer_index);
  bool synchronize_layer(const int64_t layer_index);
  uint32_t get_event_size() { return events_.size(); };

 private:
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
  const int32_t timeout_;
};

}  // namespace janus
