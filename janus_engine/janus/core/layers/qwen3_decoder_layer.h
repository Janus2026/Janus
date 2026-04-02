#pragma once

#include "qwen2_decoder_layer.h"

namespace janus {
namespace layer {

using Qwen3DecoderLayerImpl = Qwen2DecoderLayerImpl;
TORCH_MODULE(Qwen3DecoderLayer);

}  // namespace layer
}  // namespace janus