#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/request/mm_data.h"
#include "framework/request/rec_type.h"
#include "framework/request/sequences_group.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace janus {

class RecBatchInputBuilder {
 public:
  virtual ~RecBatchInputBuilder() = default;

  virtual ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) = 0;

  static std::unique_ptr<RecBatchInputBuilder> create(
      RecType rec_type,
      const std::vector<SequencesGroup*>& sequence_groups,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      ThreadPool* thread_pool = nullptr);
};

}  // namespace janus
