#pragma once

#include <string>
#include <vector>

#include "mm_handler.h"
#include "mm_type.h"

namespace janus {

class MMEmbeddingHandler : public MMHandlerBase {
 public:
  MMEmbeddingHandler(MMType::Value mm_type);
  ~MMEmbeddingHandler() = default;

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) override;
  virtual MMErrCode decode(MMInputItem& input) override;

 private:
  MMType::Value mm_type_;
};

}  // namespace janus