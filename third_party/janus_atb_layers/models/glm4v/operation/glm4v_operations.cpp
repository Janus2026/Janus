#include <gflags/gflags.h>
#include "glm4v_operations.h"

DECLARE_bool(enable_atb_comm_multiprocess);

namespace atb_speed {
namespace glm {
atb::Status
CreateNormOperation(const VisionEncoderLayerParam &vision_encoder_param,
                    atb::Operation **op) {
  atb::infer::RmsNormParam param;
  param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  param.normParam.epsilon = vision_encoder_param.rmsNormEps;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status
CreateLayerNormOperation(const VisionEncoderLayerParam &vision_encoder_param,
                    atb::Operation **op) {
  atb::infer::LayerNormParam param;
  param.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
  param.normParam.epsilon = vision_encoder_param.rmsNormEps;
  param.normParam.beginNormAxis = 1;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}


atb::Status CreateLinearOperation(atb::Operation **op, bool hasBias,
                                  bool istransB) {
  atb::infer::LinearParam param;
  param.transposeA = false;
  param.transposeB = istransB;
  param.hasBias = hasBias;
  param.outDataType = aclDataType::ACL_DT_UNDEFINED;
  param.enAccum = false;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status
CreateAllReduceOperation(const VisionEncoderLayerParam &vision_encoder_param,
                         atb::Operation **op) {
  atb::infer::AllReduceParam param;
  param.rank = vision_encoder_param.rank;
  param.rankSize = vision_encoder_param.worldSize;
  param.backend = vision_encoder_param.backend;
  if (!FLAGS_enable_atb_comm_multiprocess) {
      param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
  }
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status CreateLinearParallelOperation(
    const VisionEncoderLayerParam &vision_encoder_param, atb::Operation **op,
    bool trans_weight) {
  atb::infer::LinearParallelParam param;
  param.rank = vision_encoder_param.rank;
  param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;
  param.rankSize = vision_encoder_param.worldSize;
  param.backend = vision_encoder_param.backend;
  if (!FLAGS_enable_atb_comm_multiprocess) {
      param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
  }
  param.transWeight = trans_weight;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status CreateActivateOperation(atb::Operation** op,ActivationType type) {
  atb::infer::ActivationParam opParam;
  opParam.activationType = type;
  opParam.geluMode = atb::infer::ActivationParam::GeLUMode::TANH_MODE;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateMulOperation(atb::Operation **op) {
  atb::infer::ElewiseParam mulParam;
  mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mulParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateSplitOperation(atb::Operation **op) {
  atb::infer::SplitParam param;
  param.splitDim = -1;
  param.splitNum = 3;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status CreateRopeOperation(atb::Operation **op) {
  atb::infer::RopeParam ropeParam;
  ropeParam.rotaryCoeff = 2;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ropeParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateSelfAttentionOperation(const VisionEncoderLayerParam &param,
                                         atb::Operation **op) {
  atb::infer::SelfAttentionParam attentionParam;
  attentionParam.headNum = param.numAttentionHeadsPerRank;
  attentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
  attentionParam.calcType =
      atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
  attentionParam.maskType =
      atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED;
  attentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(attentionParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateAddOperation(atb::Operation **op) {
  atb::infer::ElewiseParam addParam;
  addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, op));
  return atb::NO_ERROR;
}
} // namespace glm
} // namespace atb_speed
