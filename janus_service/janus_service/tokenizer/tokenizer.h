#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/slice.h"

namespace janus_service {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual bool encode(const std::string_view& text,
                      std::vector<int32_t>* ids) const = 0;

  virtual std::string decode(const Slice<int32_t>& ids,
                             bool skip_special_tokens) const = 0;

  virtual std::optional<int32_t> token_to_id(
      const std::string_view& token) const = 0;

  virtual std::string id_to_token(int32_t id) const = 0;

  virtual size_t vocab_size() const = 0;

  virtual std::unique_ptr<Tokenizer> clone() const = 0;
};

}  // namespace janus_service
