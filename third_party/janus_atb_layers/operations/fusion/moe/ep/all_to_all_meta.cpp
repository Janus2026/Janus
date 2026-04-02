/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "all_to_all_meta.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/fusion/utils.h"
#include "operations/aclnn/ops/cast_operation.h"

namespace atb_speed {
namespace common {
std::map<std::string, std::vector<std::string>> GetAllToAllMetaInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllMetaInTensorCandidates = {
        {"default", {
            "in_group_count", "in_idx", "in_zero_hot"}
        },
    };
    return allToAllMetaInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAllToAllMetaIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllMetaIntermediateTensorCandidates = {
        {"default", {
            "intermediate_buffer_idx", "intermediate_buffer_idx_int64", "intermediate_group_count_int64",
            "intermediate_shuffle_idx_int64", "intermediate_zero_hot_int64", "intermediate_shuffle_filter_mask",
            "intermediate_shuffle_idx_int32", "intermediate_shuffle_idx_float16", "intermediate_one_mask",
            "intermediate_shuffle_weight"}
        },
    };
    return allToAllMetaIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAllToAllMetaOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> allToAllMetaOutTensorCandidates = {
        {"default", {
            "out_shuffle_idx_for_device_buffer", "out_shuffle_weight_mask", "out_valid_idx"}
        },
    };
    return allToAllMetaOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructAllToAllMetaTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto allToAllMetaInTensorCandidates = GetAllToAllMetaInTensorCandidates();
    auto allToAllMetaIntermediateTensorCandidates = GetAllToAllMetaIntermediateTensorCandidates();
    auto allToAllMetaOutTensorCandidates = GetAllToAllMetaOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};

    AddTensorToList(allToAllMetaInTensorCandidates, "default", inTensorList);
    AddTensorToList(allToAllMetaIntermediateTensorCandidates, "default", interTensorList);
    AddTensorToList(allToAllMetaOutTensorCandidates, "default", outTensorList);

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateOutValidIdx(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    // translated+1 done translatedpythontranslated+1
    atb::infer::SliceParam sliceParam;
    atb::Node &sliceNode = opGraph.nodes.at(nodeId++);
    sliceParam.offsets.resize(1);
    sliceParam.offsets[0] = -1;
    sliceParam.size.resize(1);
    sliceParam.size[0] = 1;
    CreateOperation(sliceParam, &sliceNode.operation);
    sliceNode.inTensorIds = {GetTensorIdx(tensorMap, "in_idx")};
    sliceNode.outTensorIds = {GetTensorIdx(tensorMap, "out_valid_idx")};
    return atb::NO_ERROR;
}

atb::Status CreateBufferIdx(std::map<std::string, uint32_t> &tensorMap,
    const AllToAllMetaParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::infer::SliceParam sliceParam;
    atb::Node &sliceNode = opGraph.nodes.at(nodeId++);
    sliceParam.offsets.resize(2); // 2: dimNum
    sliceParam.offsets[0] = 0;
    sliceParam.offsets[1] = 0;
    sliceParam.size.resize(2); // 2: dimNum
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    CreateOperation(sliceParam, &sliceNode.operation);
    sliceNode.inTensorIds = {GetTensorIdx(tensorMap, "in_idx")};
    sliceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
    sliceNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dimNum
        newShape.dims[0] = param.worldSize;
        newShape.dims[1] = oldShape.dims[0] / param.worldSize;
    };
    return atb::NO_ERROR;
}

atb::Status CreateBufferIdx64(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_buffer_idx_int64")};
    return atb::NO_ERROR;
}

atb::Status CreateGroupCount64(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "in_group_count")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_count_int64")};
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdx64(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    // translatedshuffleidx translated translatedfiltertranslated
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_SUB;
    CreateOperation(param, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_count_int64"),
                        GetTensorIdx(tensorMap, "intermediate_buffer_idx_int64")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_int64")};
    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dimNum
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    return atb::NO_ERROR;
}

atb::Status CreateZeroHot64(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "in_zero_hot")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_zero_hot_int64")};
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdxFilterMask(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam lessParam;
    lessParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_LESS;
    CreateOperation(lessParam, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_int64"),
                        GetTensorIdx(tensorMap, "intermediate_zero_hot_int64")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_filter_mask")};
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdx32(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT32;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_int64")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_int32")};
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdx(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::FillParam param;
    param.withMask = true;
    param.value.resize(1);
    param.value[0] = 0;
    CreateOperation(param, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_int32"),
                        GetTensorIdx(tensorMap, "intermediate_shuffle_filter_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "out_shuffle_idx_for_device_buffer")};
    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    node.inTensorReshapeFuncs[1] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdx16(std::map<std::string, uint32_t> &tensorMap,
    const AllToAllMetaParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AclNNCastParam castParam;
    castParam.dtype =  param.isBF16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
    castNode.operation = new atb_speed::common::CastOperation("CastNode", castParam);
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_int32")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_float16")};
    castNode.inTensorReshapeFuncs.resize(castNode.inTensorIds.size());
    castNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    return atb::NO_ERROR;
}

atb::Status CreateOneMask(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT;
    CreateOperation(param, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_filter_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_one_mask")};
    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    return atb::NO_ERROR;
}

atb::Status CreateShuffleWeightZero(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::FillParam param;
    param.withMask = true;
    param.value.resize(1);
    param.value[0] = 0;
    CreateOperation(param, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_float16"),
                        GetTensorIdx(tensorMap, "intermediate_shuffle_filter_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_weight")};
    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    return atb::NO_ERROR;
}

atb::Status CreateShuffleWeightOne(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::FillParam param;
    param.withMask = true;
    param.value.resize(1);
    param.value[0] = 1;
    CreateOperation(param, &node.operation);
    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_weight"),
                        GetTensorIdx(tensorMap, "intermediate_one_mask")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "out_shuffle_weight_mask")};
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllMetaOperation(const AllToAllMetaParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateAllToAllMetaOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "AllToAllMeta";
    std::map<std::string, uint32_t> tensorMap = ConstructAllToAllMetaTensorMap(
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    uint64_t nodeCount = 13;
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    // output 1
    CHECK_OPERATION_STATUS_RETURN(CreateOutValidIdx(tensorMap, nodeId, opGraph));
    // output 2
    CHECK_OPERATION_STATUS_RETURN(CreateBufferIdx(tensorMap, param, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateBufferIdx64(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateGroupCount64(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdx64(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateZeroHot64(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdxFilterMask(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdx32(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdx(tensorMap, nodeId, opGraph));
    // output 3
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdx16(tensorMap, param, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateOneMask(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleWeightZero(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleWeightOne(tensorMap, nodeId, opGraph));

    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(1);
        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(1).dtype = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
        outTensorDescs.at(2) = inTensorDescs.at(1); // 2: dim 2
        outTensorDescs.at(2).shape.dimNum = 1; // 2: dim 2
        outTensorDescs.at(2).shape.dims[0] = 1; // 2: dim 2
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    ATB_SPEED_LOG_DEBUG("CreateAllToAllMetaOperation success");
    return atb::NO_ERROR;
}
}
} // namespace atb_speed