#pragma once

#include <type_traits>

#include "anthropic.pb.h"
#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"

namespace janus {

// Specialize is_stream_call for AnthropicCall to recognize it as a stream call
template <>
struct is_stream_call<AnthropicCall> : std::true_type {};

class AnthropicServiceImpl final : public APIServiceImpl<AnthropicCall> {
 public:
  AnthropicServiceImpl(LLMMaster* master,
                       const std::vector<std::string>& models);

  void process_async_impl(std::shared_ptr<AnthropicCall> call) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(AnthropicServiceImpl);

  LLMMaster* master_ = nullptr;
  const std::string tool_call_parser_format_;
  const std::string reasoning_parser_format_;
};

}  // namespace janus
