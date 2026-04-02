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
#include "qwen_mlp.h"
#include <atb/atb_infer.h>
#include <memory>

namespace atb_speed {
namespace qwen {
enum QwenMlpTensorId : int {
    IN_HIDDENSTATUS = 0,
    IN_MLP_GATE_UP_WEIGHTTENSOR,
    IN_MLP_DOWN_WEIGHTTENSOR,
    IN_EXPERT_MASK_WITH_WEIGHT,
    IN_FINAL_HIDDENS_STATE,
    OUT_MLPRESULTSTENSOR,
    INTERMIDATE_MATMUL_GATE_UP_OUT,
    INTERMIDATE_MATMUL_GATE_OUT,
    INTERMIDATE_MATMUL_UP_OUT,
    INTERMIDATE_SWISH_OUT,
    INTERMIDATE_HIDDENSTATUS,
    INTERMIDATE_MLP_OUT,
    INTERMIDATE_MASKED_MLP_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 7;
static const uint64_t NODE_COUNT = 7;

int64_t AddlinearNode(atb::Node &linearNode, const QwenMlpParam &param)
{
    atb::infer::LinearParam linearParam;
    linearParam.transposeA = false;
    linearParam.transposeB = param.transpose;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = { IN_HIDDENSTATUS, IN_MLP_GATE_UP_WEIGHTTENSOR };
    linearNode.outTensorIds = { INTERMIDATE_MATMUL_GATE_UP_OUT };
    return atb::NO_ERROR;
}

int64_t AddSplitNode(atb::Node &splitNode)
{
    atb::infer::SplitParam splitParam = { 1, 2, {} };
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = { INTERMIDATE_MATMUL_GATE_UP_OUT };
    splitNode.outTensorIds = { INTERMIDATE_MATMUL_GATE_OUT, INTERMIDATE_MATMUL_UP_OUT };
    return atb::NO_ERROR;
}

int64_t AddSwishNode(atb::Node &swishNode)
{
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(activationParam, &swishNode.operation);
    swishNode.inTensorIds = { INTERMIDATE_MATMUL_GATE_OUT };
    swishNode.outTensorIds = { INTERMIDATE_SWISH_OUT };
    return atb::NO_ERROR;
}

int64_t AddMulNode(atb::Node &mulNode)
{
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = { INTERMIDATE_SWISH_OUT, INTERMIDATE_MATMUL_UP_OUT };
    mulNode.outTensorIds = { INTERMIDATE_HIDDENSTATUS };
    return atb::NO_ERROR;
}

int64_t AddlinearDownNode(atb::Node &linearDownNode, const QwenMlpParam &param)
{
    atb::infer::LinearParam linearDownParam;
    linearDownParam.transposeA = false;
    linearDownParam.transposeB = param.transpose;
    linearDownParam.hasBias = false;
    CreateOperation(linearDownParam, &linearDownNode.operation);
    linearDownNode.inTensorIds = { INTERMIDATE_HIDDENSTATUS, IN_MLP_DOWN_WEIGHTTENSOR };
    linearDownNode.outTensorIds = { INTERMIDATE_MLP_OUT };
    return atb::NO_ERROR;
}

int64_t AddMlpMulNode(atb::Node &mlpMulNode)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::infer::ElewiseParam mlpMulParam;
    mlpMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mlpMulParam, &mlpMulNode.operation);
    mlpMulNode.inTensorIds = { INTERMIDATE_MLP_OUT, IN_EXPERT_MASK_WITH_WEIGHT };
    mlpMulNode.outTensorIds = { INTERMIDATE_MASKED_MLP_OUT };
    mlpMulNode.inTensorReshapeFuncs.reserve(mlpMulNode.inTensorIds.size());
    mlpMulNode.inTensorReshapeFuncs.resize(mlpMulNode.inTensorIds.size());
    mlpMulNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 1, 2:translated
        newShape.dims[0] = oldShape.dims[1];
        newShape.dims[1] = 1;
    };
    return atb::NO_ERROR;
}

int64_t AddMlpAddNode(atb::Node &mlpAddNode)
{
    atb::infer::ElewiseParam mlpAddParam;
    mlpAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpAddParam, &mlpAddNode.operation);
    mlpAddNode.inTensorIds = { INTERMIDATE_MASKED_MLP_OUT, IN_FINAL_HIDDENS_STATE };
    mlpAddNode.outTensorIds = { OUT_MLPRESULTSTENSOR };
    return atb::NO_ERROR;
}

atb::Status CreateQwenMlpOperation(const QwenMlpParam &param, atb::Operation **operation)
{
    // std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "QwenMlp";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    atb::Node linearNode;
    atb::Node splitNode;
    atb::Node swishNode;
    atb::Node mulNode;
    atb::Node linearDownNode;
    atb::Node mlpMulNode;
    atb::Node mlpAddNode;
    CHECK_OPERATION_STATUS_RETURN(AddlinearNode(linearNode, param));
    opGraph.nodes.push_back(linearNode);
    CHECK_OPERATION_STATUS_RETURN(AddSplitNode(splitNode));
    opGraph.nodes.push_back(splitNode);
    CHECK_OPERATION_STATUS_RETURN(AddSwishNode(swishNode));
    opGraph.nodes.push_back(swishNode);
    CHECK_OPERATION_STATUS_RETURN(AddMulNode(mulNode));
    opGraph.nodes.push_back(mulNode);
    CHECK_OPERATION_STATUS_RETURN(AddlinearDownNode(linearDownNode, param));
    opGraph.nodes.push_back(linearDownNode);
    CHECK_OPERATION_STATUS_RETURN(AddMlpMulNode(mlpMulNode));
    opGraph.nodes.push_back(mlpMulNode);
    CHECK_OPERATION_STATUS_RETURN(AddMlpAddNode(mlpAddNode));
    opGraph.nodes.push_back(mlpAddNode);
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_FINAL_HIDDENS_STATE);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed

