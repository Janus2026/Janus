#pragma once

#include "block_manager_pool.h"
#include "distributed_runtime/engine.h"
#include "util/blockingconcurrentqueue.h"

namespace janus {

class Engine;

struct OffloadBlockPair {
  OffloadBlockPair(Block& s, Block& d) : src(s), dst(d) {}

  OffloadBlockPair(Block&& s, Block&& d)
      : src(std::move(s)), dst(std::move(d)) {}

  OffloadBlockPair(Block& s) : src(s) {}

  OffloadBlockPair(Block&& s) : src(std::move(s)) {}

  Block src;
  Block dst;
};

class HierarchyBlockManagerPool : public BlockManagerPool {
 public:
  using OffloadBlockPairQueue =
      moodycamel::BlockingConcurrentQueue<std::shared_ptr<OffloadBlockPair>>;

  explicit HierarchyBlockManagerPool(const BlockManagerPool::Options& options,
                                     Engine* engine,
                                     int32_t dp_size = 1);
  ~HierarchyBlockManagerPool() = default;

  bool allocate(Sequence* sequence, size_t num_tokens) override;

  // control the copy in blocks num
  bool allocate(Sequence* sequence,
                size_t num_tokens,
                size_t max_copy_in_blocks_num) override;

  void allocate_shared(Sequence* sequence) override;

  void deallocate(Sequence* sequence) override;

  void transfer_blocks(std::vector<Batch>& batches) override;
  void transfer_blocks() override;

  void prefetch_from_storage(std::shared_ptr<Request>& request) override;

  bool update_prefetch_result(std::shared_ptr<Request>& request,
                              const uint32_t timeout) override;

  void get_merged_kvcache_event(KvCacheEvent* event) const override;

 private:
  void allocate_host_shared(Sequence* sequence);

 private:
  Engine* engine_;
  std::vector<std::unique_ptr<BlockManager>> host_block_managers_;

  // BlockTransferInfo per step
  std::vector<std::vector<BlockTransferInfo>> load_block_transfer_infos_;
  std::vector<OffloadBlockPairQueue> offload_block_pair_queues_;
};

}  // namespace janus
