#include "models/qwen2_5/operation/qwen_vl_operations.h"
#include "models/qwen3_audio/qwen3_audio.h"
#include <glog/logging.h>

namespace atb_speed {
namespace qwen {

std::map<std::string, std::vector<std::string>>
GetQwen3_Audio_LayerTensorCandidates() {
  std::map<std::string, std::vector<std::string>> qwenLayerTensorCandiadates = {
      {"default_weight",
       {"in_input_norm_weight",
        "in_input_norm_bias",
        "in_post_attn_norm_weight",
        "in_post_attn_norm_bias",
        "in_qkv_weight",
        "in_qkv_bias",
        "in_attn_proj_weight",
        "in_attn_proj_bias",
        "in_linear_fc1_weight",
        "in_linear_fc1_bias",
        "in_linear_fc2_weight",
        "in_linear_fc2_bias",
        "in_q_weight",
        "in_q_bias",
        "in_k_weight",
        "in_k_bias",
        "in_v_weight",
        "in_v_bias"}},
      {"default_input",
       {
           "in_hidden_states",
           "in_seq_len",
       }},
      {"out", {"out"}},
      {"parallel_intermediate_out", {"proj_add_bias_out", "fc2_add_bias_out"}},
      {"intermediate_out",
       {"input_norm_out",
        "qkv_linear_out",
        "intermediate_q",
        "intermediate_k",
        "intermediate_v",
        "intermediate_atten_out",
        "intermediate_proj_out",
        "intermediate_add_out",
        "intermediate_post_atten_norm_out",
        "intermediate_fc1_out",
        "intermediate_gelu_out",
        "intermediate_fc2_out"}},
      {"q_len", {"in_q_len"}}};
  return qwenLayerTensorCandiadates;
}

static std::map<std::string, uint32_t> ConstructTensorMap(
    const AudioEncoderLayerParam& param,
    uint32_t& inTensorNum,
    uint32_t& outTensorNum,
    uint32_t& internalTensorNum) {
  auto qwenLayerTensorCandiadates = GetQwen3_Audio_LayerTensorCandidates();
  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {};
  // weight
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "default_weight", inTensorList);
  // input
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "default_input", inTensorList);
  // out
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "out", outTensorList);
  // intermediate
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "intermediate_out", intermediateTensorList);
  if (param.worldSize > 1) {
    atb_speed::common::AddTensorToList(qwenLayerTensorCandiadates,
                                       "parallel_intermediate_out",
                                       intermediateTensorList);
  }
  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return atb_speed::common::GetTensorMap(
      inTensorList, outTensorList, intermediateTensorList);
}

atb::Status BuildQwen3_Audio_AttentionBlock(
    const AudioEncoderLayerParam& param,
    const std::map<std::string, uint32_t>& tensorMap,
    atb::GraphParam& opGraph,
    bool isTp) {
  atb::Node inputNormNode;
  atb::Node qkvlinearNode;
  atb::Node splitNode;
  atb::Node selfAttentionNode;
  atb::Node outProjNode;
  atb::Node selfResidualAddNode;
  atb::Node proj_add_biasNode;

  // Input Norm
  CreateLayerNormOperation(param,&inputNormNode.operation);
  inputNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "in_input_norm_weight","in_input_norm_bias"});
  inputNormNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"input_norm_out"});
  opGraph.nodes.push_back(inputNormNode);

  // QKV Linear
  CreateLinearOperation(&qkvlinearNode.operation);
  qkvlinearNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"input_norm_out", "in_qkv_weight", "in_qkv_bias"});
  qkvlinearNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  opGraph.nodes.push_back(qkvlinearNode);

  // Split
  CreateSplitOperation(&splitNode.operation);
  splitNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  splitNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "intermediate_k", "intermediate_v"});
  opGraph.nodes.push_back(splitNode);

  // Self Attention
  CreateSelfAttentionOperation(param, &selfAttentionNode.operation);
  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_q",
                                           "intermediate_k",
                                           "intermediate_v",
                                           "in_seq_len"});
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_atten_out"});
  opGraph.nodes.push_back(selfAttentionNode);

  // Output Projection
  if (isTp) {
    CreateLinearParallelOperation(param, &outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_atten_out", "in_attn_proj_weight"});
  } else {
    CreateLinearOperation(&outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_atten_out", "in_attn_proj_weight", "in_attn_proj_bias"});
  }
  outProjNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_proj_out"});
  opGraph.nodes.push_back(outProjNode);

  // Tp Bias Add
  if (isTp) {
    CreateAddOperation(&proj_add_biasNode.operation);
    proj_add_biasNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_proj_out", "in_attn_proj_bias"});
    proj_add_biasNode.outTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"proj_add_bias_out"});
    opGraph.nodes.push_back(proj_add_biasNode);
  }

  // Residual Add
  CreateAddOperation(&selfResidualAddNode.operation);
  std::string residualInput =
      isTp ? "proj_add_bias_out" : "intermediate_proj_out";
  selfResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", residualInput});
  selfResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_add_out"});
  opGraph.nodes.push_back(selfResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status BuildQwen3_Audio_MLPBlock(const AudioEncoderLayerParam& param,
                          const std::map<std::string, uint32_t>& tensorMap,
                          atb::GraphParam& opGraph,
                          bool isTp) {
  atb::Node postNormNode;
  atb::Node fc1Node;
  atb::Node geluNode;
  atb::Node fc2Node;
  atb::Node fc2_add_biasNode;
  atb::Node mlpResidualAddNode;

  // Post Norm
  CreateLayerNormOperation(param,&postNormNode.operation);
  postNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_add_out", "in_post_attn_norm_weight","in_post_attn_norm_bias"});
  postNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_post_atten_norm_out"});
  opGraph.nodes.push_back(postNormNode);

  // fc1 Linear
  CreateLinearOperation(&fc1Node.operation);
  fc1Node.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_post_atten_norm_out",
                                           "in_linear_fc1_weight",
                                           "in_linear_fc1_bias"});
  fc1Node.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_fc1_out"});
  opGraph.nodes.push_back(fc1Node);

  // gelu Activation
  ActivationType act_type = ActivationType::ACTIVATION_GELU;
  CreateActivateOperation(&geluNode.operation,act_type);
  geluNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_fc1_out"});
  geluNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_gelu_out"});
  opGraph.nodes.push_back(geluNode);

  // fc2 Linear
  if (isTp) {
    CreateLinearParallelOperation(param, &fc2Node.operation);
    fc2Node.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_gelu_out", "in_linear_fc2_weight"});
  } else {
    CreateLinearOperation(&fc2Node.operation);
    fc2Node.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_gelu_out", "in_linear_fc2_weight", "in_linear_fc2_bias"});
  }
  fc2Node.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_fc2_out"});
  opGraph.nodes.push_back(fc2Node);

  // Tp Bias Add
  if (isTp) {
    CreateAddOperation(&fc2_add_biasNode.operation);
    fc2_add_biasNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_fc2_out", "in_linear_fc2_bias"});
    fc2_add_biasNode.outTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"fc2_add_bias_out"});
    opGraph.nodes.push_back(fc2_add_biasNode);
  }

  // MLP Residual Add
  std::string downInput = isTp ? "fc2_add_bias_out" : "intermediate_fc2_out";
  CreateAddOperation(&mlpResidualAddNode.operation);
  mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {downInput, "intermediate_add_out"});
  mlpResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status Qwen3_Audio_EncoderLayer(const AudioEncoderLayerParam& param,
                         atb::Operation** operation) {
  atb::GraphParam opGraph;
  opGraph.name = "Qwen3_Audio_Encoder_layer";
  bool isTp = (param.worldSize > 1);

  // tensor names
  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param,
                         opGraph.inTensorNum,
                         opGraph.outTensorNum,
                         opGraph.internalTensorNum);

  // Attention
  BuildQwen3_Audio_AttentionBlock(param, tensorMap, opGraph, isTp);
  // MLP
  BuildQwen3_Audio_MLPBlock(param, tensorMap, opGraph, isTp);
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

}  // namespace qwen
}  // namespace atb_speed
