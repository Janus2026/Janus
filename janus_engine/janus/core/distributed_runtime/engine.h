#pragma once

#include <folly/futures/Future.h>

#include <algorithm>
#include <unordered_map>

#include "common/types.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "framework/xtensor/xtensor_kv_layout.h"
#include "runtime/options.h"

namespace janus {
class Engine {
 public:
  virtual ~Engine() = default;

  virtual bool init() { return true; };

  virtual bool init(MasterStatus master_status) { return true; };

  // execute model with batch input
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;

  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;

  // return the tokenizer
  virtual const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  // return the block manager
  virtual BlockManagerPool* block_manager_pool() const {
    auto p = reinterpret_cast<BlockManagerPool*>(kv_cache_manager_.get());
    if (!p) {
      LOG(FATAL) << "kv_cache_manager_ is not BlockManagerPool type!";
    }
    return p;
  }

  // return the model args
  virtual const ModelArgs& model_args() const { return args_; }

  // return the tokenizer args
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }

  // return the active activation memory
  virtual std::vector<int64_t> get_active_activation_memory() const = 0;

  // P/D
  virtual bool pull_kv_blocks(const int32_t src_dp_size,
                              const int32_t src_dp_rank,
                              const std::vector<uint64_t>& src_cluster_ids,
                              const std::vector<std::string>& src_addrs,
                              const std::vector<int64_t>& src_k_cache_ids,
                              const std::vector<int64_t>& src_v_cache_ids,
                              const std::vector<uint64_t>& src_blocks,
                              const int32_t dst_dp_rank,
                              const std::vector<uint64_t>& dst_blocks) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    NOT_IMPLEMENTED();
    return {};
  };

  virtual void transfer_kv_blocks(
      const uint32_t dp_rank,
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    NOT_IMPLEMENTED();
  };

  virtual void prefetch_from_storage(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info,
      std::shared_ptr<std::atomic<int32_t>> flag,
      std::vector<std::shared_ptr<std::atomic<uint32_t>>>* prefetch_results) {
    NOT_IMPLEMENTED();
  };

  virtual void get_device_info(std::vector<std::string>& device_ips,
                               std::vector<uint16_t>& ports) {
    NOT_IMPLEMENTED();
  };

  virtual void get_p2p_addrs(std::vector<std::string>& p2p_addrs) {
    // Default: empty (no P2P support)
  };

  virtual void get_cache_info(std::vector<uint64_t>& cluster_ids,
                              std::vector<std::string>& addrs,
                              std::vector<int64_t>& k_cache_ids,
                              std::vector<int64_t>& v_cache_ids) {
    NOT_IMPLEMENTED();
  };

  // Get XTensor info for etcd registration (from dp group 0)
  // worker_free_phy_pages: free pages per worker
  // model_weight_segments: weight segments in GlobalXTensor per model
  virtual void get_xtensor_info(
      std::vector<size_t>& worker_free_phy_pages,
      std::unordered_map<std::string, std::vector<WeightSegment>>&
          model_weight_segments) {
    NOT_IMPLEMENTED();
  };

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports,
                            const int32_t src_dp_size) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports,
                              const int32_t dp_size) {
    NOT_IMPLEMENTED();
    return false;
  };

  // D2D link for weight transfer - each worker links to one remote addr
  // device_ips: one ip per worker, in worker order
  virtual bool link_d2d(const std::vector<std::string>& device_ips) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool unlink_d2d(const std::vector<std::string>& device_ips) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool sleep(MasterStatus master_status) {
    LOG(FATAL) << " sleep is not implemented!";
    return false;
  };

  virtual bool wakeup(const WakeupOptions& options) {
    LOG(FATAL) << " wakeup is not implemented!";
    return false;
  };

  virtual bool resize(uint64_t new_kv_cache_pages) {
    LOG(WARNING) << "resize is not implemented for this engine";
    return false;
  };

  // XTensor mode: get GlobalXTensor offsets for allocated blocks
  // Returns per-layer offsets for each block.
  // `index_offsets` is optional and populated when MLA index tensors exist.
  // dp_rank: Target DP rank to query (offsets come from workers in that DP
  // group)
  virtual bool get_xtensor_offsets_for_blocks(
      int32_t dp_rank,
      const std::vector<int32_t>& block_ids,
      std::vector<XTensorLayerOffsets>& layer_offsets,
      XTensorBlockBytes* block_bytes = nullptr) {
    return false;
  };

  struct KVCacheCapacity {
    int64_t n_blocks = 0;
    int64_t n_pages = 0;  // for continuous kvcache
    int64_t cache_size_in_bytes = 0;
    int64_t slot_size = 0;
    int64_t index_slot_size = 0;
    int64_t extra_slot_size = 0;
    int64_t linear_slot_size = 0;
    int64_t n_layers = 0;
    XTensorKvPageLayout xtensor_kv_layout;
  };

 protected:
  // model args
  ModelArgs args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // kv cache manager
  // support `block manager` and `xtensor manager` currently.
  std::unique_ptr<KVCacheManager> kv_cache_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
};

inline uint64_t get_kv_cache_required_bytes(
    const Engine::KVCacheCapacity& kv_cache_cap,
    int32_t block_size,
    int64_t n_blocks,
    uint64_t page_size = 0) {
  if (n_blocks <= 0) {
    return 0;
  }

  if (kv_cache_cap.xtensor_kv_layout.valid()) {
    if (page_size == 0) {
      return 0;
    }
    const auto& layout = kv_cache_cap.xtensor_kv_layout;
    const uint64_t virt_pages =
        (static_cast<uint64_t>(n_blocks) + layout.blocks_per_virt_page - 1) /
        layout.blocks_per_virt_page;
    return virt_pages * static_cast<uint64_t>(kv_cache_cap.n_layers) *
           layout.total_tensor_pages_per_virt_page() * page_size;
  }

  const uint64_t bytes_per_block =
      static_cast<uint64_t>(block_size) *
      static_cast<uint64_t>(kv_cache_cap.n_layers) *
      static_cast<uint64_t>(kv_cache_cap.slot_size +
                            kv_cache_cap.index_slot_size +
                            kv_cache_cap.extra_slot_size);
  return static_cast<uint64_t>(n_blocks) * bytes_per_block;
}

inline int64_t calculate_shared_kv_cache_blocks(
    int64_t cache_size_in_bytes,
    const Engine::KVCacheCapacity& target_kv_cache_cap,
    const Engine::KVCacheCapacity& draft_kv_cache_cap,
    int32_t block_size,
    uint64_t page_size = 0) {
  if (cache_size_in_bytes <= 0) {
    return 0;
  }

  int64_t low = 0;
  int64_t high =
      std::min(target_kv_cache_cap.n_blocks, draft_kv_cache_cap.n_blocks);
  const unsigned __int128 cache_budget =
      static_cast<unsigned __int128>(cache_size_in_bytes);

  while (low < high) {
    const int64_t mid = low + (high - low + 1) / 2;
    const unsigned __int128 required =
        static_cast<unsigned __int128>(get_kv_cache_required_bytes(
            target_kv_cache_cap, block_size, mid, page_size)) +
        static_cast<unsigned __int128>(get_kv_cache_required_bytes(
            draft_kv_cache_cap, block_size, mid, page_size));
    if (required <= cache_budget) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }

  return low;
}
}  // namespace janus
