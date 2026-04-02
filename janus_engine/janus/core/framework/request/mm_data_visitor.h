#pragma once

#include <torch/torch.h>

#include <unordered_set>

#include "mm_batch_data.h"
#include "mm_data.h"

namespace janus {

class CollectItemTensorVisitor : public MMData::IItemVisitor {
 public:
  CollectItemTensorVisitor(
      std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas,
      const std::unordered_set<MMKey>& black_keys = {},
      const std::unordered_set<MMKey>& white_keys = {})
      : datas_(datas), black_keys_(black_keys), white_keys_(white_keys) {};

  CollectItemTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                           const std::unordered_set<MMKey>& white_keys = {})
      : datas_(stub_), black_keys_(black_keys), white_keys_(white_keys) {};

  bool visit(MMDataItem& item) override;
  bool visit(const MMKey& key, MMValue& value) override;

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> stub_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>>& datas_;

  std::unordered_set<MMKey> black_keys_;
  std::unordered_set<MMKey> white_keys_;
};

class CollectMMDataTensorVisitor : public MMData::IVisitor {
 public:
  CollectMMDataTensorVisitor(const std::unordered_set<MMKey>& black_keys = {},
                             const std::unordered_set<MMKey>& white_keys = {})
      : item_visitor_(datas_, black_keys, white_keys) {};

  bool visit(MMData& data) override;

 public:
  uint32_t type_ = MMType::NONE;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;

  CollectItemTensorVisitor item_visitor_;
};

class EncoderInputGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderInputGatherVisitor() = default;

  bool visit(MMDataItem& item) override;
  bool finish(MMBatchData& mm_data);

 public:
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
  std::string filter_prefix_ = "embedding";
};

class EncoderOutputScatterVisitor : public MMDataItem::IVisitor {
 public:
  EncoderOutputScatterVisitor(const MMDict& data) : data_(data) {}

  bool visit(MMDataItem& data) override;
  bool finish() const;

 public:
  const MMDict& data_;
  int32_t image_idx = 0;
  int32_t video_idx = 0;
  int32_t audio_idx = 0;
};

class EncoderEmbeddingGatherVisitor : public MMDataItem::IVisitor {
 public:
  EncoderEmbeddingGatherVisitor(const torch::Device& device)
      : device_(device) {}
  bool visit(MMDataItem& data) override;
  bool finish(MMBatchData& mm_data);

 public:
  torch::Device device_;
  std::string gather_prefix_ = "embedding";
  std::unordered_map<MMKey, std::vector<torch::Tensor>> datas_;
};

}  // namespace janus
