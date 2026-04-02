#pragma once

#include "llama.h"

namespace janus {
// register the causal model
REGISTER_CAUSAL_MODEL(llama3, LlamaForCausalLM);
}  // namespace janus