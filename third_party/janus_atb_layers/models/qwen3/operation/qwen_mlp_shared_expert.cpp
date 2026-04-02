/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "qwen_mlp_shared_expert.h"
#include <atb/atb_infer.h>
#include <memory>

namespace atb_speed {
namespace qwen {
enum QwenMlpSharedExpertTensorId : int {
    IN_HIDDENSTATUS = 0,
    IN_MLP_GATE_UP_WEIGHTTENSOR,
    IN_MLP_DOWN_WEIGHTTENSOR,
    IN_MLP_GATE_LINEAR_WEIGHT_SHARED_EXPERT,
    OUT_SHARED_EXPERT_OUT,
    INTERMIDATE_MATMUL_GATE_UP_OUT,
    INTERMIDATE_MATMUL_GATE_OUT,
    INTERMIDATE_MATMUL_UP_OUT,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_HIDDENSTATUS,
    INTERMIDATE_MLP_OUT_TENSOR,
    INTERMIDATE_GATE_OUT_TENSOR,
    INTERMIDATE_SWISH_TENSOR,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 8;

int64_t AddSharelinearNode(atb::Node &linearNode, const QwenMlpSharedExpertParam &param)
{
    atb::infer::LinearParam linearParam;
    linearParam.transposeB = param.transpose;
    linearParam.transposeA = false;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = { IN_HIDDENSTATUS, IN_MLP_GATE_UP_WEIGHTTENSOR };
    linearNode.outTensorIds = { INTERMIDATE_MATMUL_GATE_UP_OUT };
    return atb::NO_ERROR;
}

int64_t AddShareSplitNode(atb::Node &splitNode)
{
    atb::infer::SplitParam splitParam = { 1, 2, {} };
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = { INTERMIDATE_MATMUL_GATE_UP_OUT };
    splitNode.outTensorIds = { INTERMIDATE_MATMUL_GATE_OUT, INTERMIDATE_MATMUL_UP_OUT };
    return atb::NO_ERROR;
}

int64_t AddShareSwishNode(atb::Node &swishNode)
{
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(activationParam, &swishNode.operation);
    swishNode.inTensorIds = { INTERMIDATE_MATMUL_GATE_OUT };
    swishNode.outTensorIds = { INTERMIDATE_SWISH_OUT };
    return atb::NO_ERROR;
}

int64_t AddShareMulNode(atb::Node &mulNode)
{
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = { INTERMIDATE_SWISH_OUT, INTERMIDATE_MATMUL_UP_OUT };
    mulNode.outTensorIds = { INTERMIDATE_HIDDENSTATUS };
    return atb::NO_ERROR;
}

int64_t AddShareLinearDownNode(atb::Node &linearDownNode, const QwenMlpSharedExpertParam &param)
{
    atb::infer::LinearParam linearDownParam;
    linearDownParam.transposeA = false;
    linearDownParam.transposeB = param.transpose;
    linearDownParam.hasBias = false;
    CreateOperation(linearDownParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = { INTERMIDATE_HIDDENSTATUS, IN_MLP_DOWN_WEIGHTTENSOR };
    if (param.hasSharedExpertGate) {
        linearDownNode.outTensorIds = { INTERMIDATE_MLP_OUT_TENSOR };
    } else {
        linearDownNode.outTensorIds = { OUT_SHARED_EXPERT_OUT };
    }
    return atb::NO_ERROR;
}

int64_t AddShareGateLinearNode(atb::Node &gateLinearNode, const QwenMlpSharedExpertParam &param)
{
    atb::infer::LinearParam gateLinearParam;
    gateLinearParam.transposeA = false;
    gateLinearParam.transposeB = param.transpose;
    gateLinearParam.hasBias = false;
    CreateOperation(gateLinearParam, &gateLinearNode.operation);
    gateLinearNode.inTensorIds = { IN_HIDDENSTATUS, IN_MLP_GATE_LINEAR_WEIGHT_SHARED_EXPERT };
    gateLinearNode.outTensorIds = { INTERMIDATE_GATE_OUT_TENSOR };
    return atb::NO_ERROR;
}

int64_t AddShareSwishActivationNode(atb::Node &swishActivationNode)
{
    atb::infer::ActivationParam swishActivationParam;
    swishActivationParam.activationType = atb::infer::ActivationType::ACTIVATION_SIGMOID;
    CreateOperation(swishActivationParam, &swishActivationNode.operation);
    swishActivationNode.inTensorIds = { INTERMIDATE_GATE_OUT_TENSOR };
    swishActivationNode.outTensorIds = { INTERMIDATE_SWISH_TENSOR };
    return atb::NO_ERROR;
}

int64_t AddShareGateMulNode(atb::Node &gateMulNode)
{
    atb::infer::ElewiseParam gateMulParam;
    gateMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(gateMulParam, &gateMulNode.operation);
    gateMulNode.inTensorIds = { INTERMIDATE_MLP_OUT_TENSOR, INTERMIDATE_SWISH_TENSOR };
    gateMulNode.outTensorIds = { OUT_SHARED_EXPERT_OUT };
    return atb::NO_ERROR;
}

atb::Status CreateQwenMlpSharedExpertOperation(const QwenMlpSharedExpertParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "QwenMlpSharedExpert";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    if (!param.hasSharedExpertGate) {
        opGraph.internalTensorNum -= 3; // gatetranslated3translatedinternalTensor
    }
    atb::Node linearNode;
    atb::Node splitNode;
    atb::Node swishNode;
    atb::Node mulNode;
    atb::Node linearDownNode;
    atb::Node gateLinearNode;
    atb::Node swishActivationNode;
    atb::Node gateMulNode;
    CHECK_OPERATION_STATUS_RETURN(AddSharelinearNode(linearNode, param));
    opGraph.nodes.push_back(linearNode);
    CHECK_OPERATION_STATUS_RETURN(AddShareSplitNode(splitNode));
    opGraph.nodes.push_back(splitNode);
    CHECK_OPERATION_STATUS_RETURN(AddShareSwishNode(swishNode));
    opGraph.nodes.push_back(swishNode);
    CHECK_OPERATION_STATUS_RETURN(AddShareMulNode(mulNode));
    opGraph.nodes.push_back(mulNode);
    CHECK_OPERATION_STATUS_RETURN(AddShareLinearDownNode(linearDownNode, param));
    opGraph.nodes.push_back(linearDownNode);
    if (param.hasSharedExpertGate) {
        CHECK_OPERATION_STATUS_RETURN(AddShareGateLinearNode(gateLinearNode, param));
        opGraph.nodes.push_back(gateLinearNode);
        CHECK_OPERATION_STATUS_RETURN(AddShareSwishActivationNode(swishActivationNode));
        opGraph.nodes.push_back(swishActivationNode);
        CHECK_OPERATION_STATUS_RETURN(AddShareGateMulNode(gateMulNode));
        opGraph.nodes.push_back(gateMulNode);
    }
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDENSTATUS);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed

