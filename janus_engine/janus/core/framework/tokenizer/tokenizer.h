#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/util/slice.h"

namespace janus {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual bool encode(const std::string_view& text,
                      std::vector<int32_t>* ids,
                      bool add_special_tokens = true) const {
    return false;
  }

  virtual bool batch_encode(const std::vector<std::string>& texts,
                            std::vector<std::vector<int32_t>>* ids) const {
    for (const auto& text : texts) {
      std::vector<int32_t> single_ids;
      if (!encode(text, &single_ids)) {
        return false;
      }
      ids->push_back(single_ids);
    }
    return true;
  }

  virtual std::string decode(const Slice<int32_t>& ids,
                             bool skip_special_tokens) const {
    return "";
  }

  // Only for generative recommendation
  virtual bool encode(int64_t item_id, std::vector<int32_t>* token_ids) const {
    return false;
  }
  // Only for generative recommendation
  virtual bool decode(const Slice<int32_t>& token_ids,
                      bool skip_special_tokens,
                      std::vector<int64_t>* item_ids) const {
    return false;
  }

  virtual std::optional<int32_t> token_to_id(
      const std::string_view& token) const {
    return std::nullopt;
  }

  virtual std::string id_to_token(int32_t id) const { return ""; }

  virtual size_t vocab_size() const { return 0; }

  virtual std::unique_ptr<Tokenizer> clone() const { return nullptr; }
};

}  // namespace janus
