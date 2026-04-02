#ifndef ATB_SPEED_MODELS_QWEN3_VL_ENCODER_H
#define ATB_SPEED_MODELS_QWEN3_VL_ENCODER_H

#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "nlohmann/json.hpp"
#include "models/qwen2_5/vision_encoder/encoder_layer.h"

namespace atb_speed {
namespace qwen {

atb::Status Qwen3VL_EncoderLayer(const VisionEncoderLayerParam& param,
                         atb::Operation** operation);

}  // namespace qwen
}  // namespace atb_speed
#endif