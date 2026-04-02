#pragma once

#include <memory>
#include <string>

#include "base_loader.h"
#include "pinned_host_memory_cache.h"
#include "rolling_weight_buffer.h"

namespace janus {
namespace layer {

class BaseManualLoader : public BaseLoader {
 public:
  using WeightSlice = PinnedHostMemoryWeightSlice;
  using HostPinnedSegment = PinnedHostMemorySegment;

  BaseManualLoader(uint64_t weight_count, const ModelContext& context);

  virtual ~BaseManualLoader() override;

  bool prepare_pinned_host_cache();
  void set_pinned_host_cache_component_key(std::string component_key);

  virtual void copy_weights_to_pinned_host();

  virtual void copy_weights_to_device();

  virtual void copy_weights_to_device_async();

  // Async H2D using the specified ACL stream (used by RollingLoadManager).
  virtual void copy_weights_to_device_async(aclrtStream stream);

  virtual void init_device_at_weights();

  virtual void init_weight_slices();

  virtual void merge_and_move_pinned_host() override;

  virtual void merge_loaded_weights() override;

  virtual void free_weights() override;

  virtual void reload_weights() override;

  virtual void reload_weights_from_device() override;

  // Rolling load path: refresh device slot pointer from rolling buffer and
  // rebuild AT tensor views from latest device base.
  virtual void refresh_rolling_weights() override;

  // Rolling load support: set the shared rolling buffer and this layer's index.
  // Device slot pointer / AT tensor views are refreshed via
  // refresh_rolling_weights().
  void set_rolling_buffer(std::shared_ptr<RollingWeightBuffer> buf,
                          int32_t layer_index);
  void* get_host_pinned_storage() const { return host_pinned_storage_; }
  uint64_t get_storage_size() const { return storage_size_; }

  // Allocate device storage (public for rolling load manager usage).
  void allocate_device_storage();

 protected:
  virtual void merge_host_at_weights() = 0;
  bool is_pinned_host_cache_hit() const { return pinned_host_cache_hit_; }
  std::string build_pinned_host_cache_key() const;
  bool can_copy_pinned_host_from_device() const;
  void copy_device_storage_to_pinned_host();
  virtual std::vector<HostPinnedSegment> build_host_pinned_segments() const;
  std::vector<HostPinnedSegment> build_balanced_host_pinned_segments(
      size_t segment_count) const;

  std::string model_id_;
  std::string model_path_;
  void* host_pinned_storage_ = nullptr;
  std::vector<HostPinnedSegment> host_pinned_segments_;
  void* device_storage_ = nullptr;
  uint64_t storage_size_ = 0;
  std::vector<WeightSlice> weight_slices_;
  static constexpr size_t kDeviceAlignment = 64;
  static constexpr size_t kHostAlignment = 64;

  virtual bool is_nz_format_tensor(int weight_index) { return false; };

  void release_device_storage();
  void release_host_storage();
  void refresh_host_pinned_storage_alias();
  const HostPinnedSegment& find_host_pinned_segment(uint64_t offset,
                                                    uint64_t bytes) const;

  std::shared_ptr<RollingWeightBuffer> rolling_buffer_ = nullptr;
  int32_t layer_index_ = -1;
  std::string pinned_host_cache_component_key_;
  std::string pinned_host_cache_key_;
  std::shared_ptr<PinnedHostMemoryEntry> pinned_host_cache_entry_ = nullptr;
  bool pinned_host_cache_hit_ = false;
  int copy_host_nd_to_nz(torch::Tensor host_tensor,
                         void* dst_ptr,
                         uint64_t len,
                         aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE);
  torch::Tensor convert_to_torch_tensor(const std::vector<int64_t>& dims,
                                        const torch::ScalarType dtype,
                                        const uintptr_t& dev_addr,
                                        int acl_format = ACL_FORMAT_ND);
};

}  // namespace layer
}  // namespace janus
