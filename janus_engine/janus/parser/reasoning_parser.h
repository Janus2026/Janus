#pragma once

#include <memory>
#include <string>

#include "parser/detector_registry.h"

namespace janus {

class ReasoningParser {
 public:
  ReasoningParser(const std::string& model_type,
                  bool stream_reasoning = true,
                  bool force_reasoning = false);

  // Non-streaming call: one-time parsing
  ReasoningResult parse_non_stream(const std::string& text);
  // Streaming call: incremental parsing
  ReasoningResult parse_stream_chunk(const std::string& chunk_text);

  static std::string get_parser_auto(const std::string& parser,
                                     const std::string& model_type);

 private:
  std::unique_ptr<ReasoningDetector> detector_;
};
}  // namespace janus