#pragma once

#include "api_service/rerank_service_impl.h"

namespace janus {

class Qwen3RerankServiceImpl final : public RerankServiceImpl {
 public:
  Qwen3RerankServiceImpl(LLMMaster* master,
                         const std::vector<std::string>& models);

  void process_async_impl(std::shared_ptr<RerankCall> call) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(Qwen3RerankServiceImpl);
};

}  // namespace janus
