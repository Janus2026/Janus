#pragma once
#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "models/glm4v/glm4v_encoder.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace glm {
using ActivationType = atb::infer::ActivationType;
atb::Status
CreateNormOperation(const VisionEncoderLayerParam &vision_encoder_param,
                    atb::Operation **op) ;

atb::Status
CreateLayerNormOperation(const VisionEncoderLayerParam &vision_encoder_param,
                    atb::Operation **op) ;

atb::Status CreateLinearOperation(atb::Operation **op, bool hasBias = true,
                                  bool istransB = true) ;

atb::Status
CreateAllReduceOperation(const VisionEncoderLayerParam &vision_encoder_param,
                         atb::Operation **op) ;

atb::Status CreateLinearParallelOperation(
    const VisionEncoderLayerParam &vision_encoder_param, atb::Operation **op,
    bool trans_weight = true) ;

atb::Status CreateActivateOperation(atb::Operation** op,ActivationType type) ;

atb::Status CreateMulOperation(atb::Operation **op) ;

atb::Status CreateSplitOperation(atb::Operation **op) ;

atb::Status CreateRopeOperation(atb::Operation **op) ;

atb::Status CreateSelfAttentionOperation(const VisionEncoderLayerParam &param,
                                         atb::Operation **op) ;

atb::Status CreateAddOperation(atb::Operation **op) ;
} // namespace qwen
} // namespace atb_speed
