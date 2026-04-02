#include "rec_batch_input_builder.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "core/common/rec_model_utils.h"
#include "onerec_batch_input_builder.h"
#include "rec_multi_round_batch_input_builder.h"

namespace janus {

std::unique_ptr<RecBatchInputBuilder> RecBatchInputBuilder::create(
    RecType rec_type,
    const std::vector<SequencesGroup*>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    ThreadPool* thread_pool) {
  switch (rec_type) {
    case RecType::kOneRec:
      return std::make_unique<OneRecBatchInputBuilder>(
          sequence_groups,
          allowed_max_tokens,
          input_embeddings_vec,
          mm_data_vec,
          swap_block_transfer_infos,
          batch_id,
          args,
          batch_forward_type,
          thread_pool);
    case RecType::kLlmRec:
      // Check if Rec multi-round mode is enabled
      if (is_rec_multi_round_mode()) {
        return std::make_unique<RecMultiRoundBatchInputBuilder>(
            sequence_groups,
            allowed_max_tokens,
            input_embeddings_vec,
            mm_data_vec,
            swap_block_transfer_infos,
            batch_id,
            args,
            batch_forward_type,
            thread_pool);
      }
      // Fall through for non-multi-round LlmRec (not yet implemented)
      break;
    case RecType::kNone:
      break;
  }

  LOG(FATAL) << "Unsupported RecType for RecBatchInputBuilder: "
             << static_cast<int32_t>(rec_type);
  return nullptr;
}

}  // namespace janus
