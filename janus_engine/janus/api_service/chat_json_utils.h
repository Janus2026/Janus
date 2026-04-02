#pragma once

#include <string>
#include <utility>

#include "core/common/types.h"

namespace janus {

// Preprocess chat JSON to normalize array content to string.
// For text-only backends (is_multimodal=false), combines text array items into
// a single string. Returns an error if non-text content is encountered.
// For multimodal backends (is_multimodal=true), leaves non-text content
// unchanged for downstream processing.
// Returns Status with processed JSON on success, or error status on failure.
std::pair<Status, std::string> preprocess_chat_json(std::string json_str,
                                                    bool is_multimodal);

}  // namespace janus
