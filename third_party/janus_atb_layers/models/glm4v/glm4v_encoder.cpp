#include "models/glm4v/glm4v_encoder.h"
#include "models/glm4v/operation/glm4v_operations.h"
#include <glog/logging.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace glm {
enum Glm4VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_WATTENTION_OUT_WEIGHT,
  IN_LINEAR_GATE_UP_WEIGHT,
  IN_LINEAR_DOWN_WEIGHT
};

std::map<std::string, std::vector<std::string>> GetGlm4v_LayerTensorCandidates() {
  std::map<std::string, std::vector<std::string>> glm4LayerTensorCandidates = {
    {"default_weight",
     {"in_input_norm_weight",
      "in_post_norm_weight",
      "in_qkv_weight",
      "in_attn_proj_weight",
      "in_linear_gate_up_weight",
      "in_linear_down_weight",
      }},
    {"default_input",
     {
        "in_hidden_states",
        "in_cos_embedding",
        "in_sin_embedding",
        "in_seq_len",
     }},
    {"out", {"out"}},
    {"intermediate_out",
     {"input_norm_out",
      "qkv_linear_out",
      "intermediate_q",
      "intermediate_k",
      "intermediate_v",
      "intermediate_rope_q",
      "intermediate_rope_k",
      "intermediate_atten_out",
      "intermediate_proj_out",
      "intermediate_add_out",
      "intermediate_post_atten_norm_out",
      "intermediate_up_gate_out",
      "intermediate_silu_out",
      "intermediate_down_out"
    }}};
  return glm4LayerTensorCandidates;
}

static std::map<std::string, uint32_t> ConstructTensorMap(
    const VisionEncoderLayerParam& param,
    uint32_t& inTensorNum,
    uint32_t& outTensorNum,
    uint32_t& internalTensorNum)  {
  auto glm4LayerTensorCandidates = GetGlm4v_LayerTensorCandidates();
  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {};
  // weight
  atb_speed::common::AddTensorToList(
      glm4LayerTensorCandidates, "default_weight", inTensorList);
  // input
  atb_speed::common::AddTensorToList(
      glm4LayerTensorCandidates, "default_input", inTensorList);
  // out
  atb_speed::common::AddTensorToList(
      glm4LayerTensorCandidates, "out", outTensorList);
  // intermediate
  atb_speed::common::AddTensorToList(
      glm4LayerTensorCandidates, "intermediate_out", intermediateTensorList);
  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return atb_speed::common::GetTensorMap(
      inTensorList, outTensorList, intermediateTensorList);
}

atb::Status BuildGlm4vAttentionBlock(
    const VisionEncoderLayerParam& param,
    const std::map<std::string, uint32_t>& tensorMap,
    atb::GraphParam& opGraph,
    bool isTp) {
  atb::Node inputNormNode;
  atb::Node qkvlinearNode;
  atb::Node splitNode;
  atb::Node ropeNode;
  atb::Node selfAttentionNode;
  atb::Node outProjNode;
  atb::Node selfResidualAddNode;

  // Input Norm
  atb::infer::RmsNormParam inputNormParam;
  inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  inputNormParam.normParam.epsilon = param.rmsNormEps;
  CreateOperation(inputNormParam, &inputNormNode.operation);
  inputNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "in_input_norm_weight"});
  inputNormNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"input_norm_out"});
  opGraph.nodes.push_back(inputNormNode);
  ATB_SPEED_LOG_INFO("GLM4v inputNorm node build success.");

  // QKV Linear
  CreateLinearOperation(&qkvlinearNode.operation, false);
  qkvlinearNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"input_norm_out", "in_qkv_weight"});
  qkvlinearNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  opGraph.nodes.push_back(qkvlinearNode);
  ATB_SPEED_LOG_INFO("GLM4v qkvlinearNode node build success.");
  // Split
  CreateSplitOperation(&splitNode.operation);
  splitNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  splitNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "intermediate_k", "intermediate_v"});
  opGraph.nodes.push_back(splitNode);
  ATB_SPEED_LOG_INFO("GLM4v splitNode node build success.");
  // Rope
  CreateRopeOperation(&ropeNode.operation);
  ropeNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_q",
                                           "intermediate_k",
                                           "in_cos_embedding",
                                           "in_sin_embedding",
                                           "in_seq_len"});
  ropeNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_rope_q", "intermediate_rope_k"});
  opGraph.nodes.push_back(ropeNode);
  ATB_SPEED_LOG_INFO("GLM4v ropeNode node build success.");

  // Self Attention
  CreateSelfAttentionOperation(param, &selfAttentionNode.operation);
  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_rope_q",
                                           "intermediate_rope_k",
                                           "intermediate_v",
                                           "in_seq_len"});
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_atten_out"});
  opGraph.nodes.push_back(selfAttentionNode);
  ATB_SPEED_LOG_INFO("GLM4v selfAttentionNode node build success.");

  // Output Projection
  if (isTp) {
    CreateLinearParallelOperation(param, &outProjNode.operation);
  } else {
    CreateLinearOperation(&outProjNode.operation, false);

  }
  outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_atten_out", "in_attn_proj_weight"});
  outProjNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_proj_out"});
  opGraph.nodes.push_back(outProjNode);
  ATB_SPEED_LOG_INFO("GLM4v  outProjNode node build success.");
  // Residual Add
  CreateAddOperation(&selfResidualAddNode.operation);
  std::string residualInput = "intermediate_proj_out";
  selfResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", residualInput});
  selfResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_add_out"});
  opGraph.nodes.push_back(selfResidualAddNode);
  ATB_SPEED_LOG_INFO("GLM4v selfResidualAddNode node build success.");
  return atb::NO_ERROR;
}

atb::Status BuildGlm4vMLPBlock(const VisionEncoderLayerParam& param,
                          const std::map<std::string, uint32_t>& tensorMap,
                          atb::GraphParam& opGraph,
                          bool isTp) {
  atb::Node postNormNode;
  atb::Node swigluNode;
  atb::Node gateNode;
  atb::Node downNode;
  atb::Node mlpResidualAddNode;

  // Post Norm
  atb::infer::RmsNormParam postNormParam;
  postNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  postNormParam.normParam.epsilon = param.rmsNormEps;
  CreateOperation(postNormParam, &postNormNode.operation);
  postNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_add_out", "in_post_norm_weight"});
  postNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_post_atten_norm_out"});
  opGraph.nodes.push_back(postNormNode);
  ATB_SPEED_LOG_INFO("GLM4v postNormParam node build success.");
  // gate up Linear
  CreateLinearOperation(&gateNode.operation, false);
  gateNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_post_atten_norm_out", "in_linear_gate_up_weight"});
  gateNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_up_gate_out"});
  opGraph.nodes.push_back(gateNode);
  ATB_SPEED_LOG_INFO("GLM4v gateNode node build success.");

  // swiglu Activation
  atb::infer::ActivationType act_type = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
  CreateActivateOperation(&swigluNode.operation, act_type);
  swigluNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_up_gate_out"});
  swigluNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_silu_out"});
  opGraph.nodes.push_back(swigluNode);
  ATB_SPEED_LOG_INFO("GLM4v swigluNode node build success.");
  // down Linear
  if (isTp) {
    CreateLinearParallelOperation(param, &downNode.operation);
    // CreateLinearOperation(&downNode.operation, false, true);
  } else {
    CreateLinearOperation(&downNode.operation, false);
  }
  downNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
                                    {"intermediate_silu_out",
                                    "in_linear_down_weight"});
  downNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_down_out"});
  opGraph.nodes.push_back(downNode);
  ATB_SPEED_LOG_INFO("GLM4v  downNode node build success.");


  // MLP Residual Add
  std::string downInput = "intermediate_down_out";
  CreateAddOperation(&mlpResidualAddNode.operation);
  mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {downInput, "intermediate_add_out"});
  mlpResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);
  ATB_SPEED_LOG_INFO("GLM4v mlpResidualAddNode node build success.");

  return atb::NO_ERROR;
}

atb::Status Glm4v_EncoderLayer(const VisionEncoderLayerParam& param,
                         atb::Operation** operation) {
  atb::GraphParam opGraph;
  opGraph.name = "Glm4_Vision_Encoder_layer";
  bool isTp = (param.worldSize > 1);

  // tensor names
  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param,
                         opGraph.inTensorNum,
                         opGraph.outTensorNum,
                         opGraph.internalTensorNum);
  // Attention
  BuildGlm4vAttentionBlock(param, tensorMap, opGraph, isTp);

  // MLP
  BuildGlm4vMLPBlock(param, tensorMap, opGraph, isTp);
  opGraph.inferShapeFunc =
    [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
        atb::SVector<atb::TensorDesc>& outTensorDescs) {
    if (!inTensorDescs.empty() && !outTensorDescs.empty()) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
    }
    return atb::NO_ERROR;
    };
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}
}  // namespace glm
}  // namespace atb_speed