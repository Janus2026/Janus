#pragma once

#include "fast_tokenizer.h"
#include "rec_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"
#include "tokenizer_args.h"
#include "tokenizer_proxy.h"

namespace janus {

class TokenizerFactory {
 public:
  static std::unique_ptr<Tokenizer> create_tokenizer(
      const std::string& model_weights_path,
      TokenizerArgs tokenizer_args,
      bool proxy = true);
};

}  // namespace janus
