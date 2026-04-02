#pragma once

#include "tokenizer.h"
#include "tokenizer_args.h"
#include "tokenizers/tokenizers.h"

namespace janus {

class FastTokenizer : public Tokenizer {
 public:
  FastTokenizer(const TokenizerArgs& tokenizer_args);

  ~FastTokenizer() override;

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  TokenizerArgs tokenizer_args_;
  TokenizerHandle handle_ = nullptr;
};

}  // namespace janus
