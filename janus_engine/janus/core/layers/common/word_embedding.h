#pragma once

#include "word_embedding_impl.h"

namespace janus {
namespace layer {

class WordEmbedding : public torch::nn::ModuleHolder<WordEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<WordEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = WordEmbeddingImpl;
  WordEmbedding(const ModelContext& context)
      : ModuleHolder(std::make_shared<WordEmbeddingImpl>(context)) {}
};

}  // namespace layer
}  // namespace janus
