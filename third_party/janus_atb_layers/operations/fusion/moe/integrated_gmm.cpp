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
#include "integrated_gmm.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/aclnn/ops/grouped_matmul_swiglu_operation.h"
#include "operations/aclnn/ops/moe_grouped_matmul_swiglu_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"
#include "atb_speed/base/event_manager.h"

namespace atb_speed {
namespace common {

static const int IDX2 = 2;
static const int IDX3 = 3;
static const int IDX6 = 6;

std::map<std::string, std::vector<std::string>> GetInteGmmInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> inteGmmInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_weight_expert", "in_bias_expert", "in_descale_expert",
            "in_offset_expert", "in_scale_expert", "in_compress_idx_expert", "in_group_list"},
        },
        {"skip_quant", {"in_dynamic_scale"}},
        {"gmm_swiglu_quant", {
            "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
            "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert"},
        }
    };
    return inteGmmInTensorCandidates;
}

int CalcGmmQuantType(const IntegratedGmmParam &param)
{
    int gmmQuantType = 0;
    int tempQuantType = 0;
    if (param.isUp) {
        tempQuantType = atb_speed::common::GetLinearQuantType(
            param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
                param.packQuantType : param.denseQuantType,
            param.moeLinearQuantType[IntegratedGmmIdx::MOE_MLP_GATE_IDX], false);
    } else {
        tempQuantType = atb_speed::common::GetLinearQuantType(
            param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
                param.packQuantType : param.denseQuantType,
            param.moeLinearQuantType[IntegratedGmmIdx::MOE_MLP_DOWN_IDX], false);
    }
    switch (tempQuantType) {
        case LinearQuantType::NO_QUANT:
            gmmQuantType = GmmQuantType::NONE;
            break;
        case LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT:
        case LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT:
            gmmQuantType = GmmQuantType::W8A8_TOKEN;
            break;
        case LinearQuantType::W8A16:
            gmmQuantType = GmmQuantType::W8A16_CHANNEL;
            break;
        case LinearQuantType::W4A16:
            gmmQuantType = GmmQuantType::W4A16_CHANNEL;
            break;
        case LinearQuantType::LINEAR_W4A8_DYNAMIC_DEQUANT:
        case LinearQuantType::LINEAR_W4A8_DYNAMIC_QUANT:
            gmmQuantType = GmmQuantType::W4A8_GROUP;
            break;
        default:
            gmmQuantType = GmmQuantType::W8A8_CHANNEL;
            break;
    }

    ATB_SPEED_LOG_DEBUG(gmmQuantType);
    return gmmQuantType;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const IntegratedGmmParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto inteGmmInTensorCandidates = GetInteGmmInTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {"out_gmm_result"};

    if (param.enableAtlasGMMFused && !param.enableGMMSwigluQuant) {
        ATB_SPEED_LOG_WARN("Please check if the world size is greater than 16. \
        IntegratedGmmParam enableAtlasGMMFused must set under enableGMMSwigluQuant is True! \
        This may be an unexpected behavior");
    }
    AddTensorToList(inteGmmInTensorCandidates, "default", inTensorList);
    if (param.enableGMMSwigluQuant) {
        AddTensorToList(inteGmmInTensorCandidates, "gmm_swiglu_quant", inTensorList);
    }
    if (param.skipQuant) {
        AddTensorToList(inteGmmInTensorCandidates, "skip_quant", inTensorList);
    }
    
    int gmmQuantType = CalcGmmQuantType(param);
    if ((gmmQuantType == GmmQuantType::W8A8_TOKEN || gmmQuantType == GmmQuantType::W4A8_GROUP) && !param.skipQuant) {
        interTensorList.push_back("intermediate_quant_out");
        interTensorList.push_back("intermediate_dynamic_scale");
    }
    if (param.enableGMMSwigluQuant && !param.enableAtlasGMMFused) {
        interTensorList.push_back("intermediate_dynamic_scale_1");
        interTensorList.push_back("intermediate_swiglu_quant_out");
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

int64_t SetAclnnDynamicQuantNode(
    std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &dynamicQuantNode = opGraph.nodes.at(nodeId++);
    dynamicQuantNode.operation = new atb_speed::common::DynamicQuantOperation("DynamicQuantNode");
    dynamicQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates")};
    dynamicQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_quant_out"),
                                     GetTensorIdx(tensorMap, "intermediate_dynamic_scale")};
    ATB_SPEED_LOG_DEBUG("create dynamic quant");
    return atb::NO_ERROR;
}

atb::Status CreateW8A8Token(
    std::map<std::string, uint32_t> &tensorMap,
    const IntegratedGmmParam &param, atb::Node &gmmNode)
{
    ATB_SPEED_LOG_DEBUG("push back W8A8_TOKEN");
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale_expert"));
    if (param.skipQuant) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_dynamic_scale"));
    } else {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_dynamic_scale"));
    }
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    gmmNode.inTensorReshapeFuncs[IDX2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    if (param.enableGMMSwigluQuant) {
        gmmNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_dynamic_scale_1"));
    }
    ATB_SPEED_LOG_DEBUG("inTensorReshapeFuncs success");
    return atb::NO_ERROR;
}

atb::Status CreateW4A8(
    std::map<std::string, uint32_t> &tensorMap,
    const IntegratedGmmParam &param, atb::Node &gmmNode)
    
{
    ATB_SPEED_LOG_DEBUG("push back W4A8");
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale_expert"));
    if (param.skipQuant) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_dynamic_scale"));
    } else {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_dynamic_scale"));
    }
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("CreateW4A8 success");
    return atb::NO_ERROR;
}

atb::Status CreateA16Channel(
    std::map<std::string, uint32_t> &tensorMap,
    atb::Node &gmmNode, const IntegratedGmmParam &param)
{
    ATB_SPEED_LOG_DEBUG("push back W4A16 or W8A16");
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    if (param.quantGroupSize == 0) {
        ATB_SPEED_LOG_DEBUG("W4A16 or W8A16 quant per-channel");
        int kDim = param.transposeB ? 1 : 2; // number of dim k
        gmmNode.inTensorReshapeFuncs[IDX2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
            newShape.dimNum = IDX2; // dimNum: 2
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[kDim];
        };
        gmmNode.inTensorReshapeFuncs[IDX3] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
            newShape.dimNum = IDX2; // dimNum: 2
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[kDim];
        };
    }

    ATB_SPEED_LOG_DEBUG("inTensorReshapeFuncs success");
    return atb::NO_ERROR;
}

atb::Status CreateGmmMixed(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId,
    const IntegratedGmmParam &param, int gmmQuantType)
{
    atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
    atb::infer::GmmDeqSwigluQuantGmmDeqParam gmmSwigluParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gmmSwigluParam, &gmmNode.operation));
    gmmNode.outTensorIds = {GetTensorIdx(tensorMap, "out_gmm_result")};

    gmmNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                           GetTensorIdx(tensorMap, "in_weight_expert")};
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_dynamic_scale"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    if (gmmQuantType == GmmQuantType::W8A8_TOKEN) {
        gmmNode.inTensorIds.push_back(
            GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"));
    }
    if (param.hasBias) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"));
    }
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"));
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    gmmNode.inTensorReshapeFuncs[IDX2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    gmmNode.inTensorReshapeFuncs[IDX6] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("GmmSwigluQuant calculation success");
    ATB_SPEED_LOG_DEBUG("GmmSwigluQuant calculation success" << param.outDataType);
    return atb::NO_ERROR;
}

// Op1 - GMM
atb::Status CreateGmm(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId,
    const IntegratedGmmParam &param, int gmmQuantType)
{
    atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AclNNGroupedMatmulParam gmmParam;
    gmmParam.quantType = gmmQuantType;
    gmmParam.outDataType = param.outDataType;
    gmmParam.transposeB = param.transposeB;
    gmmParam.hasBias = param.hasBias;
    gmmParam.enableIndexGmm = param.enableIndexGmm;
    if (param.enableGMMSwigluQuant) {
        if (param.enableInitRoutingV3) {
           atb_speed::common::AclNNMoeGroupedSwigluMatmulParam gmmSwigluParam;
           gmmSwigluParam.quantType = gmmQuantType;
           gmmSwigluParam.outDataType = param.outDataType;
           gmmSwigluParam.transposeB = param.transposeB;
           gmmNode.operation = new atb_speed::common::MoeGroupedMatmulSwigluQuantOperation("gmmSwigluNode", gmmSwigluParam);
        } else{
           atb_speed::common::AclNNGroupedSwigluMatmulParam gmmSwigluParam;
           gmmSwigluParam.quantType = gmmQuantType;
           gmmSwigluParam.outDataType = param.outDataType;
           gmmSwigluParam.transposeB = param.transposeB;
           gmmNode.operation = new atb_speed::common::GroupedMatmulSwigluOperation("gmmSwigluNode", gmmSwigluParam);
        }
        gmmNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swiglu_quant_out")};
    } else {
        gmmNode.operation = new atb_speed::common::GroupedMatmulOperation("gmmNode", gmmParam);
        gmmNode.outTensorIds = {GetTensorIdx(tensorMap, "out_gmm_result")};
    }
    if ((gmmQuantType == GmmQuantType::W8A8_TOKEN || gmmQuantType == GmmQuantType::W4A8_GROUP) && !param.skipQuant) {
        gmmNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_quant_out"),
                               GetTensorIdx(tensorMap, "in_weight_expert")};
    } else {
        gmmNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                               GetTensorIdx(tensorMap, "in_weight_expert")};
    }
    if (param.hasBias) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias_expert"));
    }
    switch (gmmParam.quantType) {
        case GmmQuantType::W8A16_CHANNEL:
        case GmmQuantType::W4A16_CHANNEL:
            CHECK_OPERATION_STATUS_RETURN(CreateA16Channel(tensorMap, gmmNode, param));
            break;
        case GmmQuantType::W8A8_CHANNEL:
            ATB_SPEED_LOG_ERROR("MoE does not support W8A8_CHANNEL");
            gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale_expert"));
            gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_compress_idx_expert"));
            gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
            break;
        case GmmQuantType::W8A8_TOKEN:
            CHECK_OPERATION_STATUS_RETURN(CreateW8A8Token(tensorMap, param, gmmNode));
            break;
        case GmmQuantType::W4A8_GROUP:
            CHECK_OPERATION_STATUS_RETURN(CreateW4A8(tensorMap, param, gmmNode));
            break;
        default:
            gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
            break;
    }
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGmm1(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, size_t &nodeId,
    const IntegratedGmmParam &param, int gmmQuantType)
{
    atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AclNNGroupedMatmulParam gmmParam;
    gmmParam.quantType = gmmQuantType;
    gmmParam.outDataType = param.outDataType;
    gmmParam.transposeB = param.downTransposeB;
    gmmParam.hasBias = param.hasBias;
    gmmParam.enableIndexGmm = param.enableIndexGmm;
    if (param.enableInitRoutingV3) {
       gmmParam.groupListType = 2;
    }
    ATB_SPEED_LOG_DEBUG("Calc GmmQuantType success");
    gmmNode.operation = new atb_speed::common::GroupedMatmulOperation("gmmNode1", gmmParam);
    gmmNode.outTensorIds = {GetTensorIdx(tensorMap, "out_gmm_result")};
    gmmNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_swiglu_quant_out"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert")};
    if (param.hasBias) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"));
    }
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_dynamic_scale_1"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    gmmNode.inTensorReshapeFuncs[IDX2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateRecord(const IntegratedGmmParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                         atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap) {
        atb::Node &recordNode = opGraph.nodes.at(nodeId++);
        recordNode.inTensorIds = {};
        recordNode.outTensorIds ={};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
            recordNode.operation,
            eventAction,
            cvKey));
        ATB_SPEED_LOG_DEBUG("Record event success");
    }
    return atb::NO_ERROR;
}

atb::Status CreateIntegratedGmmOperation(const IntegratedGmmParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "integrated_gmm";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    int gmmQuantType = CalcGmmQuantType(param);
    uint64_t nodeCount = uint64_t(1);
    if ((gmmQuantType == GmmQuantType::W8A8_TOKEN || gmmQuantType == GmmQuantType::W4A8_GROUP) && !param.skipQuant) {
        nodeCount = uint64_t(2); // 2: the number of nodes needed to compelte the calculation
    }
    if (param.enableGMMSwigluQuant && !param.enableAtlasGMMFused) {
        nodeCount += 1;
    }
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    if ((gmmQuantType == GmmQuantType::W8A8_TOKEN || gmmQuantType == GmmQuantType::W4A8_GROUP) && !param.skipQuant) {
        CHECK_OPERATION_STATUS_RETURN(SetAclnnDynamicQuantNode(tensorMap, opGraph, nodeId));
    }
    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateRecord(
            param, opGraph, nodeId, atb_speed::EventAction::POP, atb_speed::common::VECTOR_CONTROL));
    }
    if (param.enableAtlasGMMFused) {
        CHECK_OPERATION_STATUS_RETURN(CreateGmmMixed(tensorMap, opGraph, nodeId, param, gmmQuantType));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, nodeId, param, gmmQuantType));
        if (param.enableGMMSwigluQuant && gmmQuantType == GmmQuantType::W8A8_TOKEN) {
            CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, nodeId, param, gmmQuantType));
        }
    }
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed
