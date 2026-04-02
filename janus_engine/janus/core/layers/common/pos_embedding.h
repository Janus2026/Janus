#pragma once

#include "rotary_embedding.h"

namespace janus {
namespace layer {

class PosEmbedding : public torch::nn::ModuleHolder<RotaryEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<RotaryEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RotaryEmbeddingImpl;

  PosEmbedding(const ModelContext& context)
      : ModuleHolder(std::make_shared<RotaryEmbeddingImpl>(context)) {}
};

}  // namespace layer
}  // namespace janus
