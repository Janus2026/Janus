#include "mooncake_kv_cache_transfer.h"

#include <glog/logging.h>

#include <limits>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "common/global_flags.h"
#include "framework/xtensor/global_xtensor.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "util/net.h"

namespace janus {

// ============================================================================
// MooncakeKVCacheTransferBase
// ============================================================================

MooncakeKVCacheTransferBase::MooncakeKVCacheTransferBase(
    const int32_t device_id,
    const int16_t listen_port,
    const torch::Device& device,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : device_id_(device_id),
      listen_port_(listen_port),
      mooncake_te_(std::move(engine)) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
}

void MooncakeKVCacheTransferBase::initialize(int32_t device_id) {
  (void)device_id;
  addr_ = mooncake_te_->initialize();
}

void MooncakeKVCacheTransferBase::get_cache_info(uint64_t& cluster_id,
                                                 std::string& addr,
                                                 int64_t& key_cache_id,
                                                 int64_t& value_cache_id) {
  cluster_id = cluster_id_;
  addr = addr_;
  key_cache_id = 0;
  value_cache_id = 0;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVCacheTransferBase::link_cluster(const uint64_t cluster_id,
                                               const std::string& remote_addr,
                                               const std::string& device_ip,
                                               const uint16_t port) {
  LOG(INFO) << "link_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  if (remote_addr == addr_) {
    LOG(INFO) << "Skipping self-connection in link_cluster, local addr="
              << addr_;
    return true;
  }

  return mooncake_te_->open_session(cluster_id, remote_addr);
}

bool MooncakeKVCacheTransferBase::unlink_cluster(const uint64_t& cluster_id,
                                                 const std::string& remote_addr,
                                                 const std::string& device_ip,
                                                 const uint16_t port,
                                                 bool force_flag) {
  LOG(INFO) << "unlink_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->close_session(cluster_id, remote_addr);
}

// ============================================================================
// MooncakeKVCacheTransferNative
// ============================================================================

MooncakeKVCacheTransferNative::MooncakeKVCacheTransferNative(
    const int32_t device_id,
    const int16_t listen_port,
    const torch::Device& device,
    const std::string& model_type)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)),
      model_type_(model_type) {}

void MooncakeKVCacheTransferNative::allocate_kv_cache(
    std::vector<janus::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;
  allocate_kv_cache_native(kv_caches, num_layers, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferNative::register_kv_cache(
    std::vector<janus::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = kv_caches.size();

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < kv_cache_shape[0].size(); ++i) {
    count_per_block *= kv_cache_shape[0][i];
  }
  size_per_block_ = count_per_block * data_size;

  register_per_layer_kv_cache(kv_caches, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferNative::allocate_kv_cache_native(
    std::vector<janus::KVCache>& kv_caches,
    int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  // Original mode: allocate device memory using aclrtMalloc
  // calculate the size of kv cache for each layer
  auto data_size = torch::elementSize(dtype);
  int64_t k_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < kv_cache_shape[0].size(); ++i) {
    k_cache_size_per_layer *= kv_cache_shape[0][i];
  }
  int64_t v_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < kv_cache_shape[1].size(); ++i) {
    v_cache_size_per_layer *= kv_cache_shape[1][i];
  }

  // allocate device memory for kv cache
  std::vector<uint64_t> k_cache_addrs;
  std::vector<uint64_t> v_cache_addrs;
  k_cache_addrs.reserve(num_layers);
  v_cache_addrs.reserve(num_layers);

  std::vector<uintptr_t> k_tensor_addrs;
  std::vector<uintptr_t> v_tensor_addrs;
  k_tensor_addrs.reserve(num_layers);
  v_tensor_addrs.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    void* k_cache_buffer = nullptr;
    void* v_cache_buffer = nullptr;
    auto acl_ret = aclrtMalloc(
        &k_cache_buffer, k_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc k cache failed.";
    acl_ret = aclrtMalloc(
        &v_cache_buffer, v_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc v cache failed.";

    k_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(k_cache_buffer));
    v_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(v_cache_buffer));

    k_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(k_cache_buffer));
    v_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(v_cache_buffer));
  }

  // convert memory addrs to torch tensors
  aclFormat npu_format_type =
      model_type_ == "deepseek_v3" && FLAGS_enable_prefix_cache
          ? ACL_FORMAT_FRACTAL_NZ
          : ACL_FORMAT_ND;
  auto k_torch_tensors = convert_to_torch_tensor(
      kv_cache_shape[0], dtype, k_tensor_addrs, npu_format_type);
  auto v_torch_tensors = convert_to_torch_tensor(
      kv_cache_shape[1], dtype, v_tensor_addrs, npu_format_type);

  torch::Tensor key_cache, value_cache;
  for (int64_t i = 0; i < num_layers; ++i) {
    key_cache = k_torch_tensors[i];
    value_cache = v_torch_tensors[i];
    kv_caches.emplace_back(key_cache, value_cache);
  }
}

void MooncakeKVCacheTransferNative::register_per_layer_kv_cache(
    std::vector<janus::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  int64_t num_cache = num_layers_ * 2;

  std::vector<void*> cache_addrs;
  std::vector<size_t> cache_lens;
  cache_addrs.reserve(num_cache);
  cache_lens.reserve(num_cache);

  for (int32_t i = 0; i < num_layers_; ++i) {
    cache_addrs.emplace_back(kv_caches[i].get_k_cache().data_ptr());
    cache_lens.emplace_back(kv_caches[i].get_k_cache().nbytes());
  }

  for (int32_t i = 0; i < num_layers_; ++i) {
    cache_addrs.emplace_back(kv_caches[i].get_v_cache().data_ptr());
    cache_lens.emplace_back(kv_caches[i].get_v_cache().nbytes());
  }

  if (!mooncake_te_->register_memory(
          cache_addrs, cache_lens, size_per_block_)) {
    LOG(ERROR) << "register_per_layer_kv_cache failed";
    return;
  }

  LOG(INFO) << "register_per_layer_kv_cache success, num_layers=" << num_layers_
            << ", size_per_block=" << size_per_block_;
}

bool MooncakeKVCacheTransferNative::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  (void)src_cluster_id;
  (void)src_k_cache_id;
  (void)src_v_cache_id;
  std::vector<int64_t> layer_ids;
  auto ret = mooncake_te_->pull_memory_blocks(
      src_addr, src_blocks, dst_blocks, layer_ids);
  if (!ret) {
    LOG(ERROR) << "Pull kv cache blocks failed, ret = " << ret;
    return false;
  }
  return true;
}

bool MooncakeKVCacheTransferNative::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  (void)is_spec_draft;
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    layer_synchronizer->synchronize_layer(layer_index);
    for (const auto& pair : merged_kv_infos) {
      std::vector<int64_t> layer_ids = {layer_index};
      const KVCacheInfo& kv_info = pair.second;
      auto ret = mooncake_te_->push_memory_blocks(
          kv_info.dst_addr, kv_info.src_blocks, kv_info.dst_blocks, layer_ids);
      if (!ret) {
        LOG(ERROR) << "Push kv blocks failed, layer = " << layer_index
                   << ", ret = " << ret;
        return false;
      }
    }
  }
  return true;
}

// ============================================================================
// MooncakeKVCacheTransferXTensor
// ============================================================================

MooncakeKVCacheTransferXTensor::MooncakeKVCacheTransferXTensor(
    const int32_t device_id,
    const int16_t listen_port,
    const torch::Device& device)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)) {}

void MooncakeKVCacheTransferXTensor::allocate_kv_cache(
    std::vector<janus::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;
  allocate_kv_cache_xtensor(kv_caches, num_layers, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferXTensor::register_kv_cache(
    std::vector<janus::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = kv_caches.size();

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < kv_cache_shape[0].size(); ++i) {
    count_per_block *= kv_cache_shape[0][i];
  }
  size_per_block_ = count_per_block * data_size;

  register_global_xtensor(kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferXTensor::allocate_kv_cache_xtensor(
    std::vector<janus::KVCache>& kv_caches,
    int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  auto& allocator = XTensorAllocator::get_instance();
  CHECK(!model_id_.empty()) << "model_id must be set for XTensor mode";

  auto k_tensors = allocator.create_k_tensors(
      model_id_, kv_cache_shape[0], dtype, num_layers);
  auto v_tensors = allocator.create_v_tensors(
      model_id_, kv_cache_shape[1], dtype, num_layers);
  std::vector<torch::Tensor> index_tensors;
  const bool enable_index_tensor = kv_cache_shape.size() > 2;
  if (enable_index_tensor) {
    index_tensors = allocator.create_index_tensors(
        model_id_, kv_cache_shape[2], dtype, num_layers);
  }

  for (int64_t i = 0; i < num_layers; ++i) {
#if defined(USE_NPU)
    auto k_tensor =
        at_npu::native::npu_format_cast(k_tensors[i], ACL_FORMAT_ND);
    auto v_tensor =
        at_npu::native::npu_format_cast(v_tensors[i], ACL_FORMAT_ND);
    if (enable_index_tensor) {
      auto index_tensor =
          at_npu::native::npu_format_cast(index_tensors[i], ACL_FORMAT_ND);
      kv_caches.emplace_back(k_tensor, v_tensor, index_tensor);
    } else {
      kv_caches.emplace_back(k_tensor, v_tensor);
    }
#else
    if (enable_index_tensor) {
      kv_caches.emplace_back(k_tensors[i], v_tensors[i], index_tensors[i]);
    } else {
      kv_caches.emplace_back(k_tensors[i], v_tensors[i]);
    }
#endif
  }

  LOG(INFO) << "MooncakeKVCacheTransferXTensor: KV cache allocated"
            << ", model_id=" << model_id_ << ", num_layers=" << num_layers;
}

void MooncakeKVCacheTransferXTensor::register_global_xtensor(
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  auto& global_xtensor = GlobalXTensor::get_instance();
  if (!global_xtensor.is_initialized()) {
    LOG(ERROR) << "GlobalXTensor not initialized in xtensor mode";
    return;
  }

  if (global_xtensor.is_mooncake_registered()) {
    LOG(INFO) << "GlobalXTensor already registered to mooncake, skip";
    return;
  }

  std::vector<void*> addrs = {global_xtensor.base_vaddr()};
  std::vector<size_t> lens = {global_xtensor.total_size()};

  if (!mooncake_te_->register_memory(addrs, lens, size_per_block_)) {
    LOG(ERROR) << "register GlobalXTensor failed";
    return;
  }

  global_xtensor.set_mooncake_registered(true);
  LOG(INFO) << "register_global_xtensor success, total_size="
            << global_xtensor.total_size()
            << ", num_pages=" << global_xtensor.num_total_pages()
            << ", size_per_block=" << size_per_block_;
}

bool MooncakeKVCacheTransferXTensor::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  (void)src_cluster_id;
  (void)src_k_cache_id;
  (void)src_v_cache_id;
  return pull_kv_blocks_xtensor_mode(src_addr, src_blocks, dst_blocks);
}

bool MooncakeKVCacheTransferXTensor::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  (void)is_spec_draft;
  return push_kv_blocks_xtensor_mode(merged_kv_infos, layer_synchronizer);
}

bool MooncakeKVCacheTransferXTensor::pull_kv_blocks_xtensor_mode(
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for XTensor mode pull";
    return false;
  }

  std::vector<XTensorLayerOffsets> src_layer_offsets;
  std::vector<XTensorLayerOffsets> dst_layer_offsets;
  XTensorBlockBytes src_block_bytes;
  XTensorBlockBytes dst_block_bytes;
  if (!get_local_xtensor_offsets(
          src_blocks, src_layer_offsets, &src_block_bytes) ||
      !get_local_xtensor_offsets(
          dst_blocks, dst_layer_offsets, &dst_block_bytes)) {
    return false;
  }

  if (src_block_bytes.k_block_bytes != dst_block_bytes.k_block_bytes ||
      src_block_bytes.v_block_bytes != dst_block_bytes.v_block_bytes ||
      src_block_bytes.index_block_bytes != dst_block_bytes.index_block_bytes) {
    LOG(ERROR) << "Inconsistent xtensor block bytes for pull: src=("
               << src_block_bytes.k_block_bytes << ","
               << src_block_bytes.v_block_bytes << ","
               << src_block_bytes.index_block_bytes << "), dst=("
               << dst_block_bytes.k_block_bytes << ","
               << dst_block_bytes.v_block_bytes << ","
               << dst_block_bytes.index_block_bytes << ")";
    return false;
  }

  if (src_layer_offsets.size() != dst_layer_offsets.size()) {
    LOG(ERROR) << "XTensor pull layer count mismatch: src="
               << src_layer_offsets.size()
               << ", dst=" << dst_layer_offsets.size();
    return false;
  }

  auto* te = static_cast<MooncakeTransferEngine*>(mooncake_te_.get());
  for (size_t layer_id = 0; layer_id < src_layer_offsets.size(); ++layer_id) {
    if (!move_xtensor_tensor_offsets(te,
                                     src_addr,
                                     src_layer_offsets[layer_id].k_offsets,
                                     dst_layer_offsets[layer_id].k_offsets,
                                     src_block_bytes.k_block_bytes,
                                     MooncakeTransferEngine::MoveOpcode::READ,
                                     "K",
                                     static_cast<int64_t>(layer_id)) ||
        !move_xtensor_tensor_offsets(te,
                                     src_addr,
                                     src_layer_offsets[layer_id].v_offsets,
                                     dst_layer_offsets[layer_id].v_offsets,
                                     src_block_bytes.v_block_bytes,
                                     MooncakeTransferEngine::MoveOpcode::READ,
                                     "V",
                                     static_cast<int64_t>(layer_id))) {
      return false;
    }

    if (src_block_bytes.has_index()) {
      if (!move_xtensor_tensor_offsets(
              te,
              src_addr,
              src_layer_offsets[layer_id].index_offsets,
              dst_layer_offsets[layer_id].index_offsets,
              src_block_bytes.index_block_bytes,
              MooncakeTransferEngine::MoveOpcode::READ,
              "index",
              static_cast<int64_t>(layer_id))) {
        return false;
      }
    }
  }

  VLOG(1) << "pull_kv_blocks_xtensor_mode success, num_blocks="
          << src_blocks.size() << ", num_layers=" << num_layers_;
  return true;
}

bool MooncakeKVCacheTransferXTensor::push_kv_blocks_xtensor_mode(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for XTensor mode push";
    return false;
  }

  JANUS_DLOG(INFO) << "[DIAG] push_kv_blocks_xtensor_mode starting"
            << ", model_id=" << model_id_
            << ", num_layers=" << num_layers_
            << ", merged_kv_infos.size=" << merged_kv_infos.size();
  for (const auto& pair : merged_kv_infos) {
    JANUS_DLOG(INFO) << "[DIAG] KV push target: dst_addr=" << pair.second.dst_addr
              << ", src_blocks.size=" << pair.second.src_blocks.size()
              << ", dst_blocks.size=" << pair.second.dst_blocks.size()
              << ", dst_xtensor_layer_offsets.size="
              << pair.second.dst_xtensor_layer_offsets.size();
  }

  std::unordered_map<std::string, XTensorOffsetsResponse> src_xtensor_infos;
  src_xtensor_infos.reserve(merged_kv_infos.size());
  for (const auto& pair : merged_kv_infos) {
    XTensorOffsetsResponse response;
    if (!get_local_xtensor_offsets(pair.second.src_blocks,
                                   response.layer_offsets,
                                   &response.block_bytes)) {
      return false;
    }
    src_xtensor_infos.emplace(pair.first, std::move(response));
  }

  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);

    // Push the KV Cache computed at this layer for all requests
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;
      if (kv_info.dst_xtensor_layer_offsets.empty() ||
          static_cast<size_t>(layer_index) >=
              kv_info.dst_xtensor_layer_offsets.size()) {
        LOG(ERROR) << "No XTensor destination offsets from D-node for layer "
                   << layer_index;
        return false;
      }

      auto src_it = src_xtensor_infos.find(pair.first);
      if (src_it == src_xtensor_infos.end()) {
        LOG(ERROR) << "Missing cached xtensor source offsets for "
                   << pair.first;
        return false;
      }
      const auto& src_xtensor_info = src_it->second;
      if (static_cast<size_t>(layer_index) >=
          src_xtensor_info.layer_offsets.size()) {
        LOG(ERROR) << "Local XTensor source offsets missing for layer "
                   << layer_index;
        return false;
      }

      const XTensorBlockBytes& src_block_bytes = src_xtensor_info.block_bytes;
      XTensorBlockBytes dst_block_bytes = kv_info.dst_xtensor_block_bytes;
      if (!dst_block_bytes.valid()) {
        dst_block_bytes = src_block_bytes;
      }
      if (src_block_bytes.k_block_bytes != dst_block_bytes.k_block_bytes ||
          src_block_bytes.v_block_bytes != dst_block_bytes.v_block_bytes ||
          src_block_bytes.index_block_bytes !=
              dst_block_bytes.index_block_bytes) {
        LOG(ERROR) << "Inconsistent xtensor block bytes for push: src=("
                   << src_block_bytes.k_block_bytes << ","
                   << src_block_bytes.v_block_bytes << ","
                   << src_block_bytes.index_block_bytes << "), dst=("
                   << dst_block_bytes.k_block_bytes << ","
                   << dst_block_bytes.v_block_bytes << ","
                   << dst_block_bytes.index_block_bytes << ")";
        return false;
      }

      const auto& src_offsets = src_xtensor_info.layer_offsets[layer_index];
      const auto& dst_offsets = kv_info.dst_xtensor_layer_offsets[layer_index];
      auto* xtensor_te =
          static_cast<MooncakeTransferEngine*>(mooncake_te_.get());
      if (!move_xtensor_tensor_offsets(
              xtensor_te,
              kv_info.dst_addr,
              src_offsets.k_offsets,
              dst_offsets.k_offsets,
              src_block_bytes.k_block_bytes,
              MooncakeTransferEngine::MoveOpcode::WRITE,
              "K",
              layer_index) ||
          !move_xtensor_tensor_offsets(
              xtensor_te,
              kv_info.dst_addr,
              src_offsets.v_offsets,
              dst_offsets.v_offsets,
              src_block_bytes.v_block_bytes,
              MooncakeTransferEngine::MoveOpcode::WRITE,
              "V",
              layer_index)) {
        return false;
      }

      if (src_block_bytes.has_index()) {
        if (!move_xtensor_tensor_offsets(
                xtensor_te,
                kv_info.dst_addr,
                src_offsets.index_offsets,
                dst_offsets.index_offsets,
                src_block_bytes.index_block_bytes,
                MooncakeTransferEngine::MoveOpcode::WRITE,
                "index",
                layer_index)) {
          return false;
        }
      }
    }
  }

  JANUS_DLOG(INFO) << "[DIAG] push_kv_blocks_xtensor_mode completed successfully"
            << ", num_layers=" << num_layers_;
  return true;
}

bool MooncakeKVCacheTransferXTensor::get_local_xtensor_offsets(
    const std::vector<uint64_t>& block_ids,
    std::vector<XTensorLayerOffsets>& layer_offsets,
    XTensorBlockBytes* block_bytes) const {
  std::vector<int32_t> block_ids_i32;
  block_ids_i32.reserve(block_ids.size());
  for (uint64_t block_id : block_ids) {
    if (block_id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      LOG(ERROR) << "Block id exceeds int32 range in xtensor transfer: "
                 << block_id;
      return false;
    }
    block_ids_i32.push_back(static_cast<int32_t>(block_id));
  }

  auto& allocator = XTensorAllocator::get_instance();
  if (!allocator.get_xtensor_offsets(
          /*dp_rank=*/0,
          model_id_,
          block_ids_i32,
          /*block_size_bytes=*/0,
          layer_offsets,
          block_bytes)) {
    LOG(ERROR) << "Failed to get local xtensor offsets for model " << model_id_
               << ", num_blocks=" << block_ids.size();
    return false;
  }
  return true;
}

bool MooncakeKVCacheTransferXTensor::move_xtensor_tensor_offsets(
    MooncakeTransferEngine* transfer_engine,
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_offsets,
    const std::vector<uint64_t>& dst_offsets,
    uint64_t transfer_size,
    MooncakeTransferEngine::MoveOpcode move_opcode,
    const char* tensor_name,
    int64_t layer_id) const {
  if (src_offsets.empty() && dst_offsets.empty()) {
    return true;
  }
  if (src_offsets.size() != dst_offsets.size()) {
    LOG(ERROR) << "XTensor " << tensor_name
               << " offsets size mismatch at layer " << layer_id
               << ": src=" << src_offsets.size()
               << ", dst=" << dst_offsets.size();
    return false;
  }
  if (transfer_size == 0) {
    LOG(ERROR) << "XTensor " << tensor_name
               << " transfer_size is zero at layer " << layer_id;
    return false;
  }
  if (!transfer_engine->move_memory_by_global_offsets(
          remote_addr, src_offsets, dst_offsets, transfer_size, move_opcode)) {
    LOG(ERROR) << "XTensor " << tensor_name
               << " move_memory_by_global_offsets failed at layer " << layer_id;
    return false;
  }
  return true;
}

}  // namespace janus
