#include "pinned_host_memory_cache.h"

#include <acl/acl_rt.h>
#include <glog/logging.h>

namespace janus {
namespace layer {

std::shared_ptr<PinnedHostMemoryEntry> PinnedHostMemoryCache::acquire_or_create(
    const std::string& cache_key,
    bool* cache_hit) {
  CHECK(cache_hit != nullptr) << "cache_hit must not be null";

  std::shared_ptr<PinnedHostMemoryEntry> entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(cache_key);
    if (it == entries_.end()) {
      entry = std::make_shared<PinnedHostMemoryEntry>();
      entry->cache_key = cache_key;
      entry->loading = true;
      entry->ref_count = 1;
      entries_.emplace(cache_key, entry);
      *cache_hit = false;
      return entry;
    }

    entry = it->second;
    ++entry->ref_count;
  }

  std::unique_lock<std::mutex> entry_lock(entry->mutex);
  entry->cv.wait(entry_lock, [&entry]() { return !entry->loading; });
  CHECK(entry->ready) << "Pinned host cache entry is not ready for key="
                      << cache_key;
  *cache_hit = true;
  return entry;
}

std::vector<PinnedHostMemorySegment> PinnedHostMemoryCache::allocate_host_storage(
    const std::shared_ptr<PinnedHostMemoryEntry>& entry,
    const std::vector<PinnedHostMemorySegment>& segments) {
  CHECK(entry != nullptr) << "Pinned host cache entry is null";
  CHECK(!segments.empty())
      << "Pinned host cache must have at least one host segment";

  std::lock_guard<std::mutex> entry_lock(entry->mutex);
  if (entry->host_pinned_segments.empty()) {
    entry->host_pinned_segments = segments;
    for (auto& segment : entry->host_pinned_segments) {
      CHECK_GT(segment.bytes, 0)
          << "Pinned host cache segment bytes must be > 0";
      auto ret = aclrtMallocHost(&segment.storage, segment.bytes);
      CHECK_EQ(ret, ACL_SUCCESS)
          << "Failed to allocate cached pinned host segment offset="
          << segment.offset << ", size=" << segment.bytes;
    }
    entry->host_pinned_storage = entry->host_pinned_segments.front().storage;
  } else {
    CHECK_EQ(entry->host_pinned_segments.size(), segments.size())
        << "Pinned host cache segment count mismatch, key=" << entry->cache_key
        << ", cached=" << entry->host_pinned_segments.size()
        << ", requested=" << segments.size();
    for (size_t i = 0; i < segments.size(); ++i) {
      CHECK_EQ(entry->host_pinned_segments[i].offset, segments[i].offset)
          << "Pinned host cache segment offset mismatch, key="
          << entry->cache_key << ", index=" << i;
      CHECK_EQ(entry->host_pinned_segments[i].bytes, segments[i].bytes)
          << "Pinned host cache segment size mismatch, key="
          << entry->cache_key << ", index=" << i;
      CHECK(entry->host_pinned_segments[i].storage != nullptr)
          << "Pinned host cache segment storage is null, key="
          << entry->cache_key << ", index=" << i;
    }
  }
  return entry->host_pinned_segments;
}

void PinnedHostMemoryCache::publish(
    const std::shared_ptr<PinnedHostMemoryEntry>& entry,
    uint64_t storage_size,
    const std::vector<PinnedHostMemorySegment>& host_pinned_segments,
    const std::vector<PinnedHostMemoryWeightSlice>& weight_slices) {
  CHECK(entry != nullptr) << "Pinned host cache entry is null";

  std::lock_guard<std::mutex> entry_lock(entry->mutex);
  CHECK(!host_pinned_segments.empty())
      << "Pinned host cache publish needs host segments for key="
      << entry->cache_key;
  CHECK_EQ(entry->host_pinned_segments.size(), host_pinned_segments.size())
      << "Pinned host cache publish segment count mismatch for key="
      << entry->cache_key;
  entry->storage_size = storage_size;
  entry->host_pinned_storage = host_pinned_segments.front().storage;
  entry->host_pinned_segments = host_pinned_segments;
  entry->weight_slices = weight_slices;
  entry->ready = true;
  entry->loading = false;
  entry->cv.notify_all();
}

void PinnedHostMemoryCache::release(
    const std::shared_ptr<PinnedHostMemoryEntry>& entry) {
  if (entry == nullptr) {
    return;
  }

  std::vector<void*> host_pinned_segments;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(entry->cache_key);
    if (it == entries_.end()) {
      return;
    }

    auto cached_entry = it->second;
    CHECK_EQ(cached_entry.get(), entry.get())
        << "Pinned host cache entry mismatch for key=" << entry->cache_key;
    CHECK_GT(cached_entry->ref_count, 0)
        << "Pinned host cache ref_count is already zero for key="
        << entry->cache_key;
    --cached_entry->ref_count;
    if (cached_entry->ref_count == 0 && !cached_entry->loading) {
      {
        std::lock_guard<std::mutex> entry_lock(cached_entry->mutex);
        for (const auto& segment : cached_entry->host_pinned_segments) {
          if (segment.storage != nullptr) {
            host_pinned_segments.push_back(segment.storage);
          }
        }
        cached_entry->host_pinned_storage = nullptr;
        cached_entry->storage_size = 0;
        cached_entry->host_pinned_segments.clear();
        cached_entry->weight_slices.clear();
        cached_entry->ready = false;
      }
      entries_.erase(it);
    }
  }

  for (void* storage : host_pinned_segments) {
    auto ret = aclrtFreeHost(storage);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Failed to free cached pinned host segment, ret=" << ret;
    }
  }
}

}  // namespace layer
}  // namespace janus
