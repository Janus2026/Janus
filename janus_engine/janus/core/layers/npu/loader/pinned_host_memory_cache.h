#pragma once

#include <cstddef>
#include <cstdint>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace janus {
namespace layer {

struct PinnedHostMemoryWeightSlice {
  uint64_t offset = 0;
  uint64_t bytes = 0;
  std::vector<int64_t> sizes;
  torch::ScalarType dtype = torch::kFloat16;
};

struct PinnedHostMemorySegment {
  uint64_t offset = 0;
  uint64_t bytes = 0;
  void* storage = nullptr;
};

struct PinnedHostMemoryEntry {
  std::string cache_key;
  void* host_pinned_storage = nullptr;
  uint64_t storage_size = 0;
  std::vector<PinnedHostMemorySegment> host_pinned_segments;
  std::vector<PinnedHostMemoryWeightSlice> weight_slices;
  bool loading = false;
  bool ready = false;
  size_t ref_count = 0;
  std::mutex mutex;
  std::condition_variable cv;
};

class PinnedHostMemoryCache {
 public:
  static PinnedHostMemoryCache& get_instance() {
    static PinnedHostMemoryCache instance;
    return instance;
  }

  std::shared_ptr<PinnedHostMemoryEntry> acquire_or_create(
      const std::string& cache_key,
      bool* cache_hit);

  std::vector<PinnedHostMemorySegment> allocate_host_storage(
      const std::shared_ptr<PinnedHostMemoryEntry>& entry,
      const std::vector<PinnedHostMemorySegment>& segments);

  void publish(const std::shared_ptr<PinnedHostMemoryEntry>& entry,
               uint64_t storage_size,
               const std::vector<PinnedHostMemorySegment>& host_pinned_segments,
               const std::vector<PinnedHostMemoryWeightSlice>& weight_slices);

  void release(const std::shared_ptr<PinnedHostMemoryEntry>& entry);

 private:
  PinnedHostMemoryCache() = default;

  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<PinnedHostMemoryEntry>>
      entries_;
};

}  // namespace layer
}  // namespace janus
