#include "base_manual_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>

#include "core/common/global_flags.h"
#include "framework/xtensor/xtensor_allocator.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace janus {
namespace layer {

namespace {
static inline size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}
}  // namespace

BaseManualLoader::~BaseManualLoader() {
  release_host_storage();
  release_device_storage();
}

bool BaseManualLoader::prepare_pinned_host_cache() {
  if (!FLAGS_enable_manual_loader || pinned_host_cache_component_key_.empty() ||
      model_path_.empty()) {
    return false;
  }
  if (pinned_host_cache_entry_ != nullptr) {
    return pinned_host_cache_hit_;
  }

  pinned_host_cache_key_ = build_pinned_host_cache_key();
  bool cache_hit = false;
  pinned_host_cache_entry_ = PinnedHostMemoryCache::get_instance()
                                 .acquire_or_create(pinned_host_cache_key_,
                                                    &cache_hit);
  pinned_host_cache_hit_ = cache_hit;
  if (pinned_host_cache_hit_) {
    CHECK(pinned_host_cache_entry_ != nullptr)
        << "Pinned host cache hit without cache entry";
    host_pinned_storage_ = pinned_host_cache_entry_->host_pinned_storage;
    host_pinned_segments_ = pinned_host_cache_entry_->host_pinned_segments;
    storage_size_ = pinned_host_cache_entry_->storage_size;
    weight_slices_ = pinned_host_cache_entry_->weight_slices;
    refresh_host_pinned_storage_alias();
    LOG(INFO) << "Pinned host cache hit for " << pinned_host_cache_key_;
  } else {
    LOG(INFO) << "Pinned host cache miss for " << pinned_host_cache_key_;
  }
  if (can_copy_pinned_host_from_device()) {
    copy_weights_to_pinned_host();
    pinned_host_cache_hit_ = true;
    LOG(INFO) << "Pinned host cache populated from device for "
              << pinned_host_cache_key_;
    return true;
  }
  return pinned_host_cache_hit_;
}

void BaseManualLoader::set_pinned_host_cache_component_key(
    std::string component_key) {
  pinned_host_cache_component_key_ = std::move(component_key);
}

void BaseManualLoader::free_weights() { release_host_storage(); }

void BaseManualLoader::set_rolling_buffer(
    std::shared_ptr<RollingWeightBuffer> buf,
    int32_t layer_index) {
  rolling_buffer_ = std::move(buf);
  layer_index_ = layer_index;
}

void BaseManualLoader::refresh_rolling_weights() {
  if (rolling_buffer_ == nullptr) {
    return;
  }
  allocate_device_storage();
  init_device_at_weights();
}

void BaseManualLoader::allocate_device_storage() {
  if (rolling_buffer_ != nullptr) {
    // Rolling load path: use the pre-allocated slot instead of
    // XTensorAllocator. Decoder layer weights bypass XTensor regardless of
    // enable_xtensor flag.
    CHECK_GE(layer_index_, 0) << "layer_index_ not set for rolling buffer";
    device_storage_ = rolling_buffer_->get_slot_ptr(layer_index_);
    CHECK(device_storage_ != nullptr)
        << "RollingWeightBuffer slot is null for layer " << layer_index_;
    return;
  }
  if (FLAGS_enable_xtensor) {
    auto& allocator = XTensorAllocator::get_instance();
    bool ok =
        allocator.allocate_weight(model_id_, device_storage_, storage_size_);
    CHECK(ok) << "Failed to allocate contiguous device storage size="
              << storage_size_;
    return;
  }
  auto ret = aclrtMallocAlign32(
      &device_storage_, storage_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclrtMallocAlign32 failed for BaseManualLoader, size="
      << storage_size_ << ", ret=" << ret;
}

void BaseManualLoader::reload_weights() {
  allocate_device_storage();
  copy_weights_to_device_async();
  init_device_at_weights();
}

void BaseManualLoader::reload_weights_from_device() {
  // D2D path: weights already transferred to GlobalXTensor weight region.
  // Call allocate_weight to get the pointer into the pre-allocated region.
  allocate_device_storage();
  init_device_at_weights();
}

void BaseManualLoader::init_weight_slices() {
  weight_slices_.resize(weight_count_);
  size_t offset = 0;
  for (size_t i = 0; i < weight_count_; ++i) {
    weight_slices_[i] = {};
    const auto& tensor = at_host_weight_tensors_[i];
    if (!tensor.defined() || tensor.numel() < 1) {
      continue;
    }
    offset = AlignUp(offset, kHostAlignment);
    weight_slices_[i].offset = offset;
    weight_slices_[i].bytes = tensor.nbytes();
    weight_slices_[i].sizes = tensor.sizes().vec();
    weight_slices_[i].dtype = tensor.scalar_type();
    offset += weight_slices_[i].bytes;
  }
  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(offset, max_alignment);
}

void BaseManualLoader::copy_weights_to_pinned_host() {
  if (pinned_host_cache_hit_) {
    return;
  }
  CHECK_GT(storage_size_, 0) << "model size must be greater than 0.";

  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(storage_size_, max_alignment);
  host_pinned_segments_ = build_host_pinned_segments();
  CHECK(!host_pinned_segments_.empty())
      << "Host pinned segments must not be empty";

  if (pinned_host_cache_entry_ != nullptr) {
    host_pinned_segments_ = PinnedHostMemoryCache::get_instance()
                                .allocate_host_storage(pinned_host_cache_entry_,
                                                       host_pinned_segments_);
  } else {
    for (auto& segment : host_pinned_segments_) {
      auto ret = aclrtMallocHost(&segment.storage, segment.bytes);
      CHECK_EQ(ret, ACL_SUCCESS)
          << "Failed to allocate pinned host segment offset=" << segment.offset
          << ", size=" << segment.bytes;
    }
  }
  refresh_host_pinned_storage_alias();

  if (can_copy_pinned_host_from_device()) {
    copy_device_storage_to_pinned_host();
  } else {
    CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
        << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";

    for (size_t i = 0; i < weight_slices_.size(); ++i) {
      const auto& slice = weight_slices_[i];
      if (!slice.bytes) {
        continue;
      }
      const auto& segment = find_host_pinned_segment(slice.offset, slice.bytes);
      void* dst = static_cast<char*>(segment.storage) +
                  static_cast<ptrdiff_t>(slice.offset - segment.offset);
      auto host_tensor = at_host_weight_tensors_[i].to(torch::kCPU).contiguous();

      if (is_nz_format_tensor(i)) {
        int err = copy_host_nd_to_nz(
            host_tensor, dst, slice.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_EQ(err, ACL_SUCCESS)
            << "copy_host_nd_to_nz failed for tensor index " << i;
      } else {
        std::memcpy(dst, host_tensor.data_ptr(), slice.bytes);
      }
      at_host_weight_tensors_[i] = torch::zeros({1});
    }
  }

  if (pinned_host_cache_entry_ != nullptr) {
    PinnedHostMemoryCache::get_instance().publish(
        pinned_host_cache_entry_,
        storage_size_,
        host_pinned_segments_,
        weight_slices_);
  }
}

void BaseManualLoader::copy_weights_to_device_async() {
  CHECK_EQ(weight_slices_.size(), at_weight_tensors_.size())
      << "weight_slices_ size and at_weight_tensors_ size mismatch.";
  copy_weights_to_device_async(c10_npu::getCurrentNPUStream().stream());
}

void BaseManualLoader::copy_weights_to_device_async(aclrtStream stream) {
  CHECK(!host_pinned_segments_.empty())
      << "Host pinned segments are not initialized";
  for (const auto& segment : host_pinned_segments_) {
    auto ret = aclrtMemcpyAsync(static_cast<char*>(device_storage_) +
                                    static_cast<ptrdiff_t>(segment.offset),
                                segment.bytes,
                                segment.storage,
                                segment.bytes,
                                ACL_MEMCPY_HOST_TO_DEVICE,
                                stream);
    CHECK_EQ(ret, ACL_SUCCESS)
        << "aclrtMemcpyAsync failed for host pinned segment offset="
        << segment.offset << ", size=" << segment.bytes;
  }
}

void BaseManualLoader::copy_weights_to_device() {
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";

  allocate_device_storage();

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* dst = static_cast<char*>(device_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    auto host_tensor = at_host_weight_tensors_[i].contiguous();
    int err;
    if (is_nz_format_tensor(i)) {
      err = copy_host_nd_to_nz(host_tensor, dst, slice.bytes);
    } else {
      err = aclrtMemcpy(dst,
                        slice.bytes,
                        host_tensor.data_ptr(),
                        slice.bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE);
    }
    CHECK_EQ(err, ACL_SUCCESS) << "aclrtMemcpy failed for tensor index " << i;
    at_host_weight_tensors_[i] = torch::zeros({1});
  }
}

int BaseManualLoader::copy_host_nd_to_nz(torch::Tensor host_tensor,
                                         void* dst_ptr,
                                         uint64_t len,
                                         aclrtMemcpyKind kind) {
  auto tmp_tensor = at_npu::native::npu_format_cast(host_tensor.to(device_),
                                                    ACL_FORMAT_FRACTAL_NZ);
  const void* src_ptr = tmp_tensor.data_ptr();
  auto stream = c10_npu::getCurrentNPUStream();
  auto err = aclrtMemcpyAsync(dst_ptr, len, src_ptr, len, kind, stream);
  stream.synchronize();
  tmp_tensor = torch::Tensor();

  return err;
}

void BaseManualLoader::init_device_at_weights() {
  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* base = static_cast<char*>(device_storage_) +
                 static_cast<ptrdiff_t>(slice.offset);
    if (is_nz_format_tensor(i)) {
      at_weight_tensors_[i] =
          convert_to_torch_tensor(slice.sizes,
                                  slice.dtype,
                                  reinterpret_cast<uintptr_t>(base),
                                  ACL_FORMAT_FRACTAL_NZ);
    } else {
      at_weight_tensors_[i] = convert_to_torch_tensor(
          slice.sizes, slice.dtype, reinterpret_cast<uintptr_t>(base));
    }
  }
}

void BaseManualLoader::release_device_storage() {
  if (device_storage_ == nullptr) {
    return;
  }
  if (!FLAGS_enable_xtensor && !rolling_buffer_) {
    auto ret = aclrtFree(device_storage_);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "aclrtFree failed for BaseManualLoader, ret=" << ret;
    }
  }
  device_storage_ = nullptr;
}

void BaseManualLoader::release_host_storage() {
  if (host_pinned_storage_ == nullptr && pinned_host_cache_entry_ == nullptr) {
    return;
  }
  if (pinned_host_cache_entry_ != nullptr) {
    PinnedHostMemoryCache::get_instance().release(pinned_host_cache_entry_);
    pinned_host_cache_entry_.reset();
    pinned_host_cache_hit_ = false;
    host_pinned_segments_.clear();
    host_pinned_storage_ = nullptr;
    return;
  }
  for (const auto& segment : host_pinned_segments_) {
    auto ret = aclrtFreeHost(segment.storage);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Failed to free pinned host segment offset="
                 << segment.offset << ", ret=" << ret;
    }
  }
  host_pinned_segments_.clear();
  host_pinned_storage_ = nullptr;
}

BaseManualLoader::BaseManualLoader(uint64_t weight_count,
                                   const ModelContext& context)
    : BaseLoader(weight_count, context),
      model_id_(context.get_model_id()),
      model_path_(context.get_model_path()) {
  at_host_weight_tensors_.resize(weight_count_);
}

void BaseManualLoader::merge_loaded_weights() {
  merge_host_at_weights();
  init_weight_slices();
  copy_weights_to_device();
  init_device_at_weights();
}

void BaseManualLoader::merge_and_move_pinned_host() {
  if (pinned_host_cache_hit_) {
    return;
  }
  merge_host_at_weights();
  init_weight_slices();
  copy_weights_to_pinned_host();
}

std::string BaseManualLoader::build_pinned_host_cache_key() const {
  CHECK(!pinned_host_cache_component_key_.empty())
      << "Pinned host cache component key is empty";
  CHECK(!model_path_.empty()) << "Pinned host cache model_path is empty";

  const std::string normalized_model_path =
      std::filesystem::path(model_path_).lexically_normal().string();
  return normalized_model_path + "|component=" +
         pinned_host_cache_component_key_ + "|rank=" +
         std::to_string(parallel_args_.rank()) + "|world_size=" +
         std::to_string(parallel_args_.world_size()) + "|dtype=" +
         torch_dtype_ + "|quant=" + quantize_type_;
}

bool BaseManualLoader::can_copy_pinned_host_from_device() const {
  if (rolling_buffer_ != nullptr) {
    return false;
  }
  if (device_storage_ == nullptr || storage_size_ == 0) {
    return false;
  }
  return weight_slices_.size() == weight_count_;
}

void BaseManualLoader::copy_device_storage_to_pinned_host() {
  CHECK(can_copy_pinned_host_from_device())
      << "Device storage is not ready for pinned-host D2H copy";
  CHECK(!host_pinned_segments_.empty())
      << "Host pinned segments are not initialized";
  for (const auto& segment : host_pinned_segments_) {
    auto ret =
        aclrtMemcpy(segment.storage,
                    segment.bytes,
                    static_cast<char*>(device_storage_) +
                        static_cast<ptrdiff_t>(segment.offset),
                    segment.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_EQ(ret, ACL_SUCCESS)
        << "aclrtMemcpy device->pinned_host failed for segment offset="
        << segment.offset << ", size=" << segment.bytes;
  }
}

std::vector<BaseManualLoader::HostPinnedSegment>
BaseManualLoader::build_host_pinned_segments() const {
  return {{0, storage_size_, nullptr}};
}

std::vector<BaseManualLoader::HostPinnedSegment>
BaseManualLoader::build_balanced_host_pinned_segments(
    size_t segment_count) const {
  if (segment_count <= 1 || storage_size_ == 0 || weight_slices_.empty()) {
    return {{0, storage_size_, nullptr}};
  }
  if (segment_count != 2) {
    LOG(WARNING) << "Balanced host pinned segmentation currently supports only "
                 << "2 segments, requested=" << segment_count
                 << ". Falling back to single segment.";
    return {{0, storage_size_, nullptr}};
  }

  uint64_t best_boundary = 0;
  uint64_t best_max_segment_size = std::numeric_limits<uint64_t>::max();
  for (const auto& slice : weight_slices_) {
    if (slice.bytes == 0 || slice.offset == 0 || slice.offset >= storage_size_) {
      continue;
    }
    const uint64_t left_size = slice.offset;
    const uint64_t right_size = storage_size_ - slice.offset;
    if (left_size == 0 || right_size == 0) {
      continue;
    }
    const uint64_t max_segment_size = std::max(left_size, right_size);
    if (max_segment_size < best_max_segment_size) {
      best_max_segment_size = max_segment_size;
      best_boundary = slice.offset;
    }
  }

  if (best_boundary == 0 || best_boundary >= storage_size_) {
    return {{0, storage_size_, nullptr}};
  }

  return {{0, best_boundary, nullptr},
          {best_boundary, storage_size_ - best_boundary, nullptr}};
}

void BaseManualLoader::refresh_host_pinned_storage_alias() {
  host_pinned_storage_ =
      host_pinned_segments_.empty() ? nullptr : host_pinned_segments_.front().storage;
}

const BaseManualLoader::HostPinnedSegment&
BaseManualLoader::find_host_pinned_segment(uint64_t offset,
                                           uint64_t bytes) const {
  for (const auto& segment : host_pinned_segments_) {
    const uint64_t segment_end = segment.offset + segment.bytes;
    const uint64_t slice_end = offset + bytes;
    if (offset >= segment.offset && slice_end <= segment_end) {
      CHECK(segment.storage != nullptr)
          << "Host pinned segment storage is null for offset=" << segment.offset;
      return segment;
    }
  }
  LOG(FATAL) << "No host pinned segment covers range [offset=" << offset
             << ", bytes=" << bytes << "]";
  return host_pinned_segments_.front();
}

torch::Tensor BaseManualLoader::convert_to_torch_tensor(
    const std::vector<int64_t>& dims,
    const torch::ScalarType dtype,
    const uintptr_t& dev_addr,
    int acl_format) {
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  auto tensor = torch::empty({0}, option);
  auto address = reinterpret_cast<void*>(dev_addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);

  // PyTorch 2.7+: StorageImpl now takes DataPtr instead of raw allocator
  storage = fptr(c10::StorageImpl::use_byte_size_t(),
                 c10::SymInt(tensor_nbytes),
                 std::move(c10_data_ptr),
                 allocator,
                 true);

  tensor.set_(storage, 0, dims);
  // Notice: convert to NZ format forcefully, with the underlying data format
  // guaranteed by the developer.
  if (acl_format == ACL_FORMAT_FRACTAL_NZ) {
    auto* tensor_storage = static_cast<torch_npu::NPUStorageImpl*>(
        tensor.storage().unsafeGetStorageImpl());
    tensor_storage->npu_desc_.npu_format_ = ACL_FORMAT_FRACTAL_NZ;
  }

  return tensor;
}
}  // namespace layer
}  // namespace janus
