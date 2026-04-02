#pragma once

#include <optional>
#include <string>

namespace janus {

struct ReasoningResult {
  std::optional<std::string> normal_text = std::nullopt;
  std::optional<std::string> reasoning_text = std::nullopt;

  ReasoningResult() = default;

  ReasoningResult(std::optional<std::string> normal,
                  std::optional<std::string> reasoning)
      : normal_text(normal), reasoning_text(reasoning) {}
};

class ReasoningDetector {
 public:
  ReasoningDetector(const std::string& think_start_token,
                    const std::string& think_end_token,
                    bool force_reasoning = false,
                    bool stream_reasoning = true);

  ~ReasoningDetector() = default;

  // Detects and parses reasoning sections in the provided text. Returns both
  // reasoning content and normal text separately.
  ReasoningResult detect_and_parse(std::string& text);

  // Streaming incremental parsing for reasoning content.
  // Handles partial reasoning tags and content.
  //
  // If stream_reasoning is False:
  //     Accumulates reasoning content until the end tag is found
  // If stream_reasoning is True:
  //     Streams reasoning content as it arrives
  ReasoningResult parse_streaming_increment(std::string& new_text);

 protected:
  std::string think_start_token_;
  std::string think_end_token_;
  bool in_reasoning_;
  bool stream_reasoning_;
  std::string buffer_ = "";
  bool stripped_think_start_ = false;
};
}  // namespace janus