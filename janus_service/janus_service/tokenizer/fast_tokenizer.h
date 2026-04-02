#pragma once

#include "tokenizer.h"
#include "tokenizers/tokenizers.h"

namespace janus_service {

class FastTokenizer : public Tokenizer {
 public:
  FastTokenizer(const std::string& tokenizer_json_path);

  ~FastTokenizer() override;

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  std::string tokenizer_json_path_;

  TokenizerHandle handle_ = nullptr;
};

}  // namespace janus_service
