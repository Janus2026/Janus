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
#include <gflags/gflags.h>

#include <cmath>
#include "atb_speed/base/event_manager.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/argmax_operation.h"
#include "operations/aclnn/ops/matmul_allreduce_operation.h"
#include "operations/aclnn/ops/max_v2_operation.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "models/base/util.h"

DECLARE_bool(enable_atb_comm_multiprocess);
DECLARE_bool(enable_multi_stream_parallel);

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetLinearParallelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearPrallelInTensorCandidates = {
        {"default", {
            "in_input", "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"}
        },
        {"reduce_quant", {
            "in_reduce_quant_scale", "in_reduce_quant_offset", "in_gather_quant_scale", "in_gather_quant_offset"}
        },
        {"lora", {"in_seq_len_cum_sum", "in_lora_a", "in_lora_b"}},
        {"lora_with_mask", {"in_im_mask"}},
        {"swiglu_quant", {"intermediate_swiglu_dynamic_scale"}},
        {"flash_comm", {"send_counts", "sdispls", "recv_count", "fake_rs_shape"}},
    };
    return linearPrallelInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLinearParallelIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearPrallelIntermediateTensorCandidates = {
        {"linear_out", {"intermediate_linear_out"}},
        {"sync_out", {"intermediate_sync_out"}},
        {"quant_out", {"intermediate_quant_out"}},
        {"argmax", {"argmax_out", "argmax_withvalue_out", "transpose_argmax_out", "transpose_argmax_withvalue_out"}},
        {"inner_tp", {"intermediate_inner_tp_input", "intermediate_inner_linear_out"}},
        {"inner_tp_prefill", {"intermediate_tp_allgather"}},
    };
    return linearPrallelIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLinearParallelOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearPrallelOutTensorCandidates = {
        {"argmax_out", {"all_argmax_out", "all_argmaxwithvalue_out"}},
    };
    return linearPrallelOutTensorCandidates;
}

bool IsDownDynamicDeQuant(const LinearParallelParam &param)
{
    return param.fusionLinearParam.isDownLinear && param.fusionLinearParam.isPrefill &&
        param.fusionLinearParam.enableSwigluQuant &&
        (param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT ||
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::LINEAR_W4A8_DYNAMIC_DEQUANT);
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const LinearParallelParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum,
    bool enableLcoc)
{
    auto linearPrallelInTensorCandidates = GetLinearParallelInTensorCandidates();
    auto linearPrallelIntermediateTensorCandidates = GetLinearParallelIntermediateTensorCandidates();
    auto linearPrallelOutTensorCandidates =  GetLinearParallelOutTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {};
    if (!param.isArgmaxlogits) {
        outTensorList = {"out"};
    }

    // translatedTensor
    AddTensorToList(linearPrallelInTensorCandidates, "default", inTensorList);

    // translatedTensor
    if (enableLcoc) {
        if (param.biasAfterSync && !param.isArgmaxlogits) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "sync_out", intermediateTensorList);
        }
    } else if (param.innerTensorParallelInfo.rankIds.size() > 1) {    // translatedTensor
        AddTensorToList(linearPrallelIntermediateTensorCandidates, "inner_tp", intermediateTensorList);
        if (param.isPrefill) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "inner_tp_prefill", intermediateTensorList);
        }
    } else {
        AddTensorToList(linearPrallelIntermediateTensorCandidates, "linear_out", intermediateTensorList);
        if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "quant_out", intermediateTensorList);
            }
        // All gathertranslated
        if (param.parallelType == COLUMN_PARALLEL && !param.isArgmaxlogits) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "sync_out", intermediateTensorList);
        }
    }

    // translatedLoratranslatedTensor
    if (param.fusionLinearParam.supportLora) {
        if (param.fusionLinearParam.useImMask) {
            AddTensorToList(linearPrallelInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(linearPrallelInTensorCandidates, "lora", inTensorList);
    }
    // translatedlccl reduce int8translatedTensor
    if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
        AddTensorToList(linearPrallelInTensorCandidates, "reduce_quant", inTensorList);
    }
    if (IsDownDynamicDeQuant(param)) {
        AddTensorToList(linearPrallelInTensorCandidates, "swiglu_quant", inTensorList);
    }
    // translatedtensor
    if (param.isArgmaxlogits) {
        AddTensorToList(linearPrallelIntermediateTensorCandidates, "argmax", intermediateTensorList);
        AddTensorToList(linearPrallelOutTensorCandidates, "argmax_out", outTensorList);
    }
    // translatedflashcomm1.0translatedtensor
    if (param.parallelType == REDUCE_SCATTER) {
        AddTensorToList(linearPrallelInTensorCandidates, "flash_comm", inTensorList);
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddAllReduceOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.tensorParallelInfo.rank;
    allReduceParam.rankSize = param.tensorParallelInfo.worldSize;
    allReduceParam.backend = param.tensorParallelInfo.backend;
    allReduceParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
    allReduceParam.quantType = param.tensorParallelInfo.quantType;
    allReduceParam.outDataType = param.tensorParallelInfo.outDataType;
    allReduceParam.commDomain = param.tensorParallelInfo.commDomain;
    allReduceParam.hcclComm = param.tensorParallelInfo.hcommInfo;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allReduceParam, &allReduceNode.operation));
    if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL) {
        bool isQuant = param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT;
        std::vector<std::string> allReduceInTensors = {
            isQuant ? "intermediate_quant_out" : "intermediate_linear_out", \
            "in_reduce_quant_scale", "in_gather_quant_offset"
        };
        allReduceNode.inTensorIds = {GetTensorIdxList(tensorMap, allReduceInTensors)};
    } else {
        allReduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
    }
    allReduceNode.outTensorIds = {
        GetTensorIdx(tensorMap, param.biasAfterSync ? "intermediate_linear_out" : "out")
    };
    if (param.fusionLinearParam.isPrefill && FLAGS_enable_multi_stream_parallel) {
        atb::SetExecuteStreamId(allReduceNode.operation, 1);
    }
    opGraph.nodes.push_back(allReduceNode);

    if (param.biasAfterSync) {
        atb::Node addNode;
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
        addNode.inTensorIds = GetTensorIdxList(tensorMap, {
            "intermediate_linear_out", "in_bias"
        });
        addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(addNode);
    }
    return atb::NO_ERROR;
}

int64_t AddReduceScatterOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node reduceScatterVNode;
    atb::infer::ReduceScatterVParam reduceScatterVParam;
    reduceScatterVParam.rank = param.tensorParallelInfo.rank;
    reduceScatterVParam.rankSize = param.tensorParallelInfo.worldSize;
    reduceScatterVParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
    reduceScatterVParam.backend = param.tensorParallelInfo.backend;
    reduceScatterVParam.commDomain = param.tensorParallelInfo.commDomain;
    reduceScatterVParam.hcclComm = param.tensorParallelInfo.hcommInfo;
    if (!FLAGS_enable_atb_comm_multiprocess) {
      reduceScatterVParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(reduceScatterVParam, &reduceScatterVNode.operation));
    reduceScatterVNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
    reduceScatterVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "send_counts"));
    reduceScatterVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "sdispls"));
    reduceScatterVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "recv_count"));
    reduceScatterVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "fake_rs_shape"));
    reduceScatterVNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    if (param.fusionLinearParam.isPrefill && FLAGS_enable_multi_stream_parallel) {
        atb::SetExecuteStreamId(reduceScatterVNode.operation, 1);
    }
    opGraph.nodes.push_back(reduceScatterVNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    return atb::NO_ERROR;
}

atb::Status CreateArgmax(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node argmaxNode;
    atb_speed::common::AclNNArgMaxParam argMaxParam;
    argMaxParam.keepdim = true;
    argmaxNode.operation = new atb_speed::common::ArgMaxOperation("argmaxNode", argMaxParam);
    argmaxNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
    argmaxNode.outTensorIds = {GetTensorIdx(tensorMap, "argmax_out")};
    opGraph.nodes.push_back(argmaxNode);

    return atb::NO_ERROR;
}

atb::Status CreateArgmaxwithValue(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node maxNode;
    atb_speed::common::AclNNMaxV2Param maxV2Param;
    maxV2Param.keepdim = true;
    maxNode.operation = new atb_speed::common::MaxV2Operation("maxNode", maxV2Param);
    maxNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
    maxNode.outTensorIds = {GetTensorIdx(tensorMap, "argmax_withvalue_out")};
    opGraph.nodes.push_back(maxNode);

    return atb::NO_ERROR;
}

int64_t AddCommunicationOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(AddDapEventsBeforeComm(opGraph));
    if (param.fusionLinearParam.isPrefill && FLAGS_enable_multi_stream_parallel) {
        janus::atb_utils::insert_pop_events(opGraph);
    }
    if (param.parallelType == ROW_PARALLEL) {
        CHECK_OPERATION_STATUS_RETURN(AddAllReduceOp(param, opGraph, tensorMap));
    } else if (param.parallelType == REDUCE_SCATTER) {
        CHECK_OPERATION_STATUS_RETURN(AddReduceScatterOp(param, opGraph, tensorMap));
    } else {
        atb::Node allGatherNode;
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.tensorParallelInfo.rank;
        allGatherParam.rankSize = param.tensorParallelInfo.worldSize;
        allGatherParam.backend = param.tensorParallelInfo.backend;
        allGatherParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
        allGatherParam.hcclComm = param.tensorParallelInfo.hcommInfo;
        allGatherParam.commDomain = param.tensorParallelInfo.commDomain;
        if (!FLAGS_enable_atb_comm_multiprocess) {
          allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
        }
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
        allGatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
        allGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sync_out")};
        if (param.fusionLinearParam.isPrefill && FLAGS_enable_multi_stream_parallel) {
            atb::SetExecuteStreamId(allGatherNode.operation, 1);
        }        
        opGraph.nodes.push_back(allGatherNode);
    }
    CHECK_OPERATION_STATUS_RETURN(AddDapEventsAfterComm(opGraph));

    if (param.parallelType == COLUMN_PARALLEL) {
        atb::Node transposeNode;
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = {1, 0, 2};
        } else {
            transposeParam.perm = {1, 2, 0, 3};
        }
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
        transposeNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_sync_out")};
        transposeNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(transposeNode);
    }
    return atb::NO_ERROR;
}

int64_t AddCommunicationArgmaxOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
        atb::Node allGatherNode;
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.tensorParallelInfo.rank;
        allGatherParam.rankSize = param.tensorParallelInfo.worldSize;
        allGatherParam.backend = param.tensorParallelInfo.backend;
        allGatherParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
        allGatherNode.inTensorIds = {GetTensorIdx(tensorMap, "argmax_out")};
        allGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "transpose_argmax_out")};
        opGraph.nodes.push_back(allGatherNode);
        atb::Node transposeNode;
        atb::infer::TransposeParam transposeParam;
        transposeParam.perm = {1, 0, 2};
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
        transposeNode.inTensorIds = {GetTensorIdx(tensorMap, "transpose_argmax_out")};
        transposeNode.outTensorIds = {GetTensorIdx(tensorMap, "all_argmax_out")};
        opGraph.nodes.push_back(transposeNode);
    return atb::NO_ERROR;
}

int64_t AddCommunicationMaxOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
        atb::Node allGatherNode;
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.tensorParallelInfo.rank;
        allGatherParam.rankSize = param.tensorParallelInfo.worldSize;
        allGatherParam.backend = param.tensorParallelInfo.backend;
        allGatherParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
        if (!FLAGS_enable_atb_comm_multiprocess) {
          allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
        }
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
        allGatherNode.inTensorIds = {GetTensorIdx(tensorMap, "argmax_withvalue_out")};
        allGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "transpose_argmax_withvalue_out")};
        opGraph.nodes.push_back(allGatherNode);
        atb::Node transposeNode;
        atb::infer::TransposeParam transposeParam;
        transposeParam.perm = {1, 0, 2};
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
        transposeNode.inTensorIds = {GetTensorIdx(tensorMap, "transpose_argmax_withvalue_out")};
        transposeNode.outTensorIds = {GetTensorIdx(tensorMap, "all_argmaxwithvalue_out")};
        opGraph.nodes.push_back(transposeNode);

    return atb::NO_ERROR;
}

atb::Status AddInnerPreAllGatherNode(const LinearParallelParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.innerTensorParallelInfo.rank;
    allGatherParam.rankSize = param.innerTensorParallelInfo.rankIds.size();
    allGatherParam.backend = param.innerTensorParallelInfo.defaultBackend;
    param.innerTensorParallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    if (!FLAGS_enable_atb_comm_multiprocess) {
          allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    }
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input")};
    allGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_tp_allgather")};
    opGraph.nodes.push_back(allGatherNode);
    ATB_SPEED_LOG_DEBUG("Fusion linear Inner pre AllGather calculation success");
    return atb::NO_ERROR;
}

atb::Status AddInnerPreAllGatherSliceNode(const LinearParallelParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sliceNode;
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets.resize(2); // 2: dimNum
    sliceParam.offsets[0] = 0;
    sliceParam.offsets[1] = param.innerTensorParallelInfo.rank * param.innerTpShape;
    sliceParam.size.resize(2); // 2: dimNum
    sliceParam.size[0] = -1;
    sliceParam.size[1] = param.innerTpShape;
    CreateOperation(sliceParam, &sliceNode.operation);

    sliceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_tp_allgather")};
    sliceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_inner_tp_input")};

    sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
    sliceNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim num
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim
    };
    opGraph.nodes.push_back(sliceNode);
    ATB_SPEED_LOG_DEBUG("Inner pre AllGather Slice calculation success");
    return atb::NO_ERROR;
}

atb::Status AddInnerPreAllToAllNode(const LinearParallelParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allToAllNode;
    atb::infer::AllToAllParam allToAllParam;
    allToAllParam.rank = param.innerTensorParallelInfo.rank;
    allToAllParam.rankSize = param.innerTensorParallelInfo.rankIds.size();
    allToAllParam.backend = "lccl";
    allToAllParam.transpose = true;
    param.innerTensorParallelInfo.InitCommDomain(allToAllParam.hcclComm, allToAllParam.commDomain, "lccl");
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allToAllParam, &allToAllNode.operation));
    allToAllNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input")};
    allToAllNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_inner_tp_input")};
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allToAllNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    ATB_SPEED_LOG_DEBUG("Inner pre AllToAll calculation success");
    return atb::NO_ERROR;
}

atb::Status AddInnerPostReduceScatterNode(const LinearParallelParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node reduceScatterNode;
    atb::infer::ReduceScatterParam reduceScatterParam;
    reduceScatterParam.rank = param.innerTensorParallelInfo.rank;
    reduceScatterParam.rankSize = param.innerTensorParallelInfo.rankIds.size();
    reduceScatterParam.backend = param.innerTensorParallelInfo.defaultBackend;
    param.innerTensorParallelInfo.InitCommDomain(reduceScatterParam.hcclComm, reduceScatterParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceScatterParam, &reduceScatterNode.operation));
    reduceScatterNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_inner_linear_out")};
    reduceScatterNode.outTensorIds = {GetTensorIdx(tensorMap, param.tensorParallelInfo.worldSize > 1 ?
        "intermediate_linear_out" : "out")};
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(reduceScatterNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    ATB_SPEED_LOG_DEBUG("Inner Post Reduce Scatter calculation success");
    return atb::NO_ERROR;
}

atb::Status AddFusionLinearNode(const LinearParallelParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node linearNode;
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(linearParam, &linearNode.operation));
    std::vector<std::string> linearInTensor = {
        param.innerTensorParallelInfo.rankIds.size() > 1 ? "intermediate_inner_tp_input" : "in_input",
        "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"
    };
    if (param.fusionLinearParam.supportLora) {
        if (param.fusionLinearParam.useImMask) {
            linearInTensor.push_back("in_im_mask");
        }
        linearInTensor.push_back("in_seq_len_cum_sum");
        linearInTensor.push_back("in_lora_a");
        linearInTensor.push_back("in_lora_b");
    }
    if (IsDownDynamicDeQuant(param)) {
        linearInTensor.push_back("intermediate_swiglu_dynamic_scale");
    }
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, linearInTensor);
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, param.innerTensorParallelInfo.rankIds.size() > 1 ?
        "intermediate_inner_linear_out" : "intermediate_linear_out")};
    opGraph.nodes.push_back(linearNode);
    return atb::NO_ERROR;
}

atb::Status CreateLinearParallelMC2(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum, true);
    atb::Node linearParallelNode;
    linearParallelNode.operation = new atb_speed::common::MatmulAllreduceOperation("matmulAllReduce",
            param.tensorParallelInfo.hcommInfo);
    linearParallelNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "in_input", "in_weight"
    });

    linearParallelNode.outTensorIds = {
        GetTensorIdx(tensorMap, param.biasAfterSync ? "intermediate_sync_out" : "out")
    };
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(linearParallelNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));

    if (param.biasAfterSync) {
        atb::Node addNode;
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
        addNode.inTensorIds = GetTensorIdxList(tensorMap, {
                        "intermediate_sync_out", "in_bias"
        });
        addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(addNode);
    }

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

void LinearParallelInferShape(atb::GraphParam &opGraph, const LinearParallelParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        uint32_t inputIdx = GetTensorIdx(tensorMap, "in_input");
        uint32_t weightIdx = GetTensorIdx(tensorMap, "in_weight");
        uint32_t biasIdx = GetTensorIdx(tensorMap, "in_bias");
        uint32_t resultDim = 2;
        if (param.isArgmaxlogits) {
            outTensorDescs.at(0).dtype = aclDataType::ACL_INT32;
            outTensorDescs.at(0).format = inTensorDescs.at(inputIdx).format;
            outTensorDescs.at(0).shape.dimNum = resultDim; // translated [batch_size,1]
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(inputIdx).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = param.worldSize;
            outTensorDescs.at(1).dtype = inTensorDescs.at(inputIdx).dtype;
            outTensorDescs.at(1).format = inTensorDescs.at(inputIdx).format;
            outTensorDescs.at(1).shape.dimNum = resultDim; // translated [batch_size,1]
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(inputIdx).shape.dims[0];
            outTensorDescs.at(1).shape.dims[1] = param.worldSize;
        } else {
            outTensorDescs.at(0) = inTensorDescs.at(inputIdx);
            if (param.fusionLinearParam.isDownLinear && param.fusionLinearParam.enableSwigluQuant) {
                outTensorDescs.at(0).dtype = param.fusionLinearParam.isBF16 ? \
                aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
            }
            CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(inputIdx).shape.dimNum);
            auto dimLast = inTensorDescs.at(inputIdx).shape.dimNum - 1;
            int nDim = param.fusionLinearParam.transposeType == TransposeType::TRANSPOSE ? 0 : 1;
            if (param.parallelType == COLUMN_PARALLEL) {
                outTensorDescs.at(0).shape.dims[dimLast] = \
                    CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[nDim],
                                        param.tensorParallelInfo.worldSize);
            } else if (param.parallelType == REDUCE_SCATTER) {
                uint32_t fakeRsShapeIdx = GetTensorIdx(tensorMap, "fake_rs_shape");
                outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(fakeRsShapeIdx).shape.dims[0];
                outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(weightIdx).shape.dims[nDim];
            } else {
                if (param.fusionLinearParam.quantType == LINEAR_W8A8_SC_DEQUANT || \
                    param.fusionLinearParam.quantType == LINEAR_W8A8_SC_QUANT) {
                    outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(biasIdx).shape.dims[0];
                } else if (param.fusionLinearParam.quantType == W4A16) {
                    if (param.fusionLinearParam.transposeType == TransposeType::TRANSPOSE) {
                        outTensorDescs.at(0).shape.dims[dimLast] = \
                            inTensorDescs.at(weightIdx).shape.dims[0];  // 0: ntranslatedshape
                    } else {
                        outTensorDescs.at(0).shape.dims[dimLast] = \
                            CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[1], 2);  // 1, 2: translatedshape * 2
                    }
                } else {
                    outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(weightIdx).shape.dims[nDim];
                }
            }
        }
        return atb::NO_ERROR;
    };
}

atb::Status CreateLinearParallel(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    if (param.parallelType == REDUCE_SCATTER) {
        opGraph.name = "LinearReduceScatter";
    } else if (param.parallelType == ROW_PARALLEL && !param.biasAfterSync) {
        opGraph.name = "LinearRowParallelNoAdd";
    } else {
        opGraph.name = param.parallelType == COLUMN_PARALLEL ?  "LinearColumnParallel" : "LinearRowParallelAndAdd";
    }

    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum, false);

    if (param.innerTensorParallelInfo.rankIds.size() > 1) {
        if (param.isPrefill) {
            CHECK_OPERATION_STATUS_RETURN(AddInnerPreAllGatherNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddInnerPreAllGatherSliceNode(param, opGraph, tensorMap));
        } else {
            CHECK_OPERATION_STATUS_RETURN(AddInnerPreAllToAllNode(param, opGraph, tensorMap));
        }
    }

    CHECK_OPERATION_STATUS_RETURN(AddFusionLinearNode(param, opGraph, tensorMap));

    if (param.innerTensorParallelInfo.rankIds.size() > 1) {
        CHECK_OPERATION_STATUS_RETURN(AddInnerPostReduceScatterNode(param, opGraph, tensorMap));
    }

    if (param.tensorParallelInfo.worldSize > 1) {
        if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL && \
                param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
            atb::Node quantNode;
            atb::infer::ElewiseParam quantParam;
            quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(quantParam, &quantNode.operation));
            quantNode.inTensorIds = GetTensorIdxList(tensorMap, {
                "intermediate_linear_out", "in_reduce_quant_scale", "in_reduce_quant_offset"
            });
            quantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_quant_out")};
            opGraph.nodes.push_back(quantNode);
        }
        if (param.isArgmaxlogits) {
            CHECK_OPERATION_STATUS_RETURN(CreateArgmax(opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(CreateArgmaxwithValue(opGraph, tensorMap));
        }
        if (!param.isArgmaxlogits) {
            CHECK_OPERATION_STATUS_RETURN(AddCommunicationOp(param, opGraph, tensorMap));
        } else {
            CHECK_OPERATION_STATUS_RETURN(AddCommunicationArgmaxOp(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddCommunicationMaxOp(param, opGraph, tensorMap));
        }
    }
    LinearParallelInferShape(opGraph, param, tensorMap);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

atb::Status CreateLinearParallelLcoc(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "LinearParallelLcoc";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum, true);
    ATB_SPEED_LOG_DEBUG("linear parallel lcoc opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("linear parallel lcoc opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("linear parallel lcoc opGraph.internalTensorNum " << opGraph.internalTensorNum);

    atb::Node linearParallelNode;
    atb::infer::LinearParallelParam linearParallelParam;
    linearParallelParam.transWeight = param.fusionLinearParam.transposeType == TransposeType::TRANSPOSE;
    linearParallelParam.rank = param.tensorParallelInfo.rank;
    linearParallelParam.rankSize = param.tensorParallelInfo.worldSize;
    linearParallelParam.hasResidual = false;
    linearParallelParam.backend = "lcoc";
    linearParallelParam.commDomain = param.tensorParallelInfo.commDomain;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParallelParam, &linearParallelNode.operation));

    linearParallelNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "in_input", "in_weight"
    });
    linearParallelNode.outTensorIds = {
        GetTensorIdx(tensorMap, param.biasAfterSync ? "intermediate_sync_out" : "out")
    };
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(linearParallelNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));

    if (param.biasAfterSync) {
        atb::Node addNode;
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
        addNode.inTensorIds = GetTensorIdxList(tensorMap, {
            "intermediate_sync_out", "in_bias"
        });
        addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(addNode);
    }

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

atb::Status LinearParallel(const LinearParallelParam &param, atb::Operation **operation)
{
    if (param.tensorParallelInfo.worldSize <= 1 && param.innerTensorParallelInfo.rankIds.size() <= 1) {
        return FusionLinear(param.fusionLinearParam, operation);
    } else if (param.parallelType == ROW_PARALLEL) {
        if (param.tensorParallelInfo.backend == "hccl" && param.enableMC2 && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
                return CreateLinearParallelMC2(param, operation);
        } else if (param.tensorParallelInfo.backend == "lccl" && \
            param.supportLcoc && !param.fusionLinearParam.supportLora && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
                return CreateLinearParallelLcoc(param, operation);
        }
        return CreateLinearParallel(param, operation);
    } else if (param.parallelType == COLUMN_PARALLEL || param.parallelType == REDUCE_SCATTER) {
        return CreateLinearParallel(param, operation);
    } else {
        ATB_SPEED_LOG_ERROR("LinearParallel operation doesn't support parallelType: " << param.parallelType
            << " Possible values are 1 (row parallel) or 2 (column parallel).");
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed