#ifndef ATB_SPEED_MODELS_QWEN3_AUDIO_ENCODER_H
#define ATB_SPEED_MODELS_QWEN3_AUDIO_ENCODER_H

#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "nlohmann/json.hpp"
#include "models/qwen2_5/vision_encoder/encoder_layer.h"

namespace atb_speed {
namespace qwen {


struct AudioEncoderLayerParam : VisionEncoderLayerParam {
   int64_t paddingHeadNums = 0;
};

atb::Status Qwen3_Audio_EncoderLayer(const AudioEncoderLayerParam& param,
                         atb::Operation** operation);


}  // namespace qwen
}  // namespace atb_speed
#endif
