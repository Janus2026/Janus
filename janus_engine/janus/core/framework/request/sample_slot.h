#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"

namespace janus {

struct SampleSlot {
  std::string request_id;
  size_t sample_id = 0;
  size_t token_position = 0;
};

bool build_sample_slots(const std::string& request_id,
                        const std::string& prompt,
                        const std::string& literal,
                        const Tokenizer& tokenizer,
                        std::vector<SampleSlot>* sample_slots);

}  // namespace janus
