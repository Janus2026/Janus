#pragma once

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework//request/request_output.h"
#include "core/util/utils.h"
#include "embedding.pb.h"
#include "tensor.pb.h"

namespace janus {
class TensorProtoBuilder {
 public:
  TensorProtoBuilder(bool use_binary_encoding);
  ~TensorProtoBuilder() = default;
  bool build_repeated_tensor(
      const std::vector<torch::Tensor>& in_tensors,
      google::protobuf::RepeatedPtrField<janus::proto::Tensor>& out_tensors,
      std::string& binary_payload);
  bool build_tensor(const torch::Tensor& in_tensor,
                    janus::proto::Tensor& out_tensor,
                    std::string& binary_payload);
  bool build_tensor(const janus::proto::Tensor& in_tensor,
                    const std::string& binary_payload,
                    torch::Tensor& out_tensor);

 private:
  bool use_binary_encoding_;
};

class EmbeddingOutputBuilder {
 public:
  EmbeddingOutputBuilder(bool embedding_use_binary_encoding,
                         bool metadata_use_binary_encoding);
  ~EmbeddingOutputBuilder();
  bool build_repeated_embedding_output(
      const std::vector<EmbeddingOutput>& in_embeddings,
      google::protobuf::RepeatedPtrField<janus::proto::Embedding>&
          out_embeddings,
      std::string& binary_payload);
  bool build_embedding_output(const EmbeddingOutput& in_embedding,
                              janus::proto::Embedding& out_embedding,
                              std::string& binary_payload);
  bool build_embedding_output(const janus::proto::Embedding& in_embedding,
                              std::string& binary_payload,
                              EmbeddingOutput& out_embedding);

 private:
  bool embedding_use_binary_encoding_;
  bool metadata_use_binary_encoding_;
};

}  // namespace janus