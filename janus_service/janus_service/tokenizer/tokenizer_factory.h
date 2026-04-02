#pragma once

#include "fast_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"
#include "tokenizer_args.h"

namespace janus_service {

class TokenizerFactory {
 public:
  static std::unique_ptr<Tokenizer> create_tokenizer(
      const std::string& model_weights_path,
      TokenizerArgs* tokenizer_args);
};

}  // namespace janus_service
