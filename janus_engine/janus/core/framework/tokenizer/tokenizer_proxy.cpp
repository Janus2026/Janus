#include "tokenizer_proxy.h"

#include <filesystem>

#include "fast_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"

namespace janus {

TokenizerProxy::TokenizerProxy(std::unique_ptr<Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

std::unique_ptr<Tokenizer> TokenizerProxy::clone() const {
  return std::make_unique<TokenizerProxy>(tokenizer_->clone());
}

bool TokenizerProxy::encode(const std::string_view& text,
                            std::vector<int32_t>* ids,
                            bool add_special_tokens) const {
  return get_tls_tokenizer()->encode(text, ids, add_special_tokens);
}

std::string TokenizerProxy::decode(const Slice<int32_t>& ids,
                                   bool skip_special_tokens) const {
  return get_tls_tokenizer()->decode(ids, skip_special_tokens);
}

std::optional<int32_t> TokenizerProxy::token_to_id(
    const std::string_view& token) const {
  return get_tls_tokenizer()->token_to_id(token);
}

std::string TokenizerProxy::id_to_token(int32_t id) const {
  return get_tls_tokenizer()->id_to_token(id);
}

size_t TokenizerProxy::vocab_size() const {
  return get_tls_tokenizer()->vocab_size();
}

Tokenizer* TokenizerProxy::get_tls_tokenizer() const {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}
}  // namespace janus