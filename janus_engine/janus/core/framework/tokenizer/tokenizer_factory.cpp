#include "tokenizer_factory.h"

#include <filesystem>

namespace janus {

std::unique_ptr<Tokenizer> TokenizerFactory::create_tokenizer(
    const std::string& model_weights_path,
    TokenizerArgs tokenizer_args,
    bool proxy) {
  std::unique_ptr<Tokenizer> tokenizer;
  if (tokenizer_args.tokenizer_type() == "fast") {
    // 1. fast tokenizer
    LOG(INFO) << "Create fast tokenizer.";
    tokenizer = std::make_unique<FastTokenizer>(tokenizer_args);
  } else if (tokenizer_args.tokenizer_type() == "tiktoken" ||
             tokenizer_args.tokenizer_class() == "TikTokenTokenizer") {
    // 2. create tiktoken tokenizer
    LOG(INFO) << "Create Tiktoken tokenizer.";
    tokenizer =
        std::make_unique<TiktokenTokenizer>(model_weights_path, tokenizer_args);
  } else if (tokenizer_args.tokenizer_type() == "rec") {
    // 3. create rec tokenizer
    LOG(INFO) << "Create rec tokenizer.";
    tokenizer =
        std::make_unique<RecTokenizer>(model_weights_path, tokenizer_args);
  } else {
    // 4. create sentencepiece tokenizer
    LOG(INFO) << "Create SentencePiece tokenizer.";
    tokenizer = std::make_unique<SentencePieceTokenizer>(model_weights_path,
                                                         tokenizer_args);
  }

  if (proxy) {
    return std::make_unique<TokenizerProxy>(std::move(tokenizer));
  }
  return tokenizer;
}

}  // namespace janus
