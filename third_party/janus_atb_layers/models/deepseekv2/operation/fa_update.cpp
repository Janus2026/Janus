/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "models/deepseekv2/operation/fa_update.h"
#include "operations/aclnn/ops/attn_operation.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace deepseekV2 {
using namespace atb_speed::common;
atb::Status AddGoLseConcat(atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap)
{
    // [B, N, dc] [B, N, 1] -> [B, N, dc+1]
    atb::Node catNode;
    atb::infer::ConcatParam catParam;
    catParam.concatDim = -1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(catParam, &catNode.operation));
    catNode.inTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_go", "intermediate_lse"})};
    catNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_lse")};
    opGraph.nodes.push_back(catNode);

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddAll2All(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
                       std::map<std::string, uint32_t>& tensorMap)
{
    // CP / CP+SP: [B, N, dc+1] -> [B, N*(dc+1)] -> [sp*B, N/sp*(dc+1)]
    atb::Node all2AllNode;
    atb::infer::AllToAllParam all2AllParam;
    all2AllParam.rank = param.attnSpRank;
    all2AllParam.rankSize = param.attnSpSize;
    all2AllParam.backend = "lccl"; // all2all transpose translated lccl
    all2AllParam.rankTableFile = param.attnSpRankTableFile;
    all2AllParam.commDomain = param.attnSpDomain;
    all2AllParam.hcclComm = param.attnSpHcclComm;
    all2AllParam.transpose = true;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(all2AllParam, &all2AllNode.operation));
    all2AllNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_lse")};
    all2AllNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_lse_all2all")};
    all2AllNode.inTensorReshapeFuncs.resize(all2AllNode.inTensorIds.size());
    all2AllNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: [B, N, dc+1] -> [B, N*(dc+1)]
    };
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(all2AllNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddAllGather(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
                         std::map<std::string, uint32_t>& tensorMap)
{
    // CP: [B, N, dc+1] -> [B, N*(dc+1)] -> [cp, B, N*(dc+1)]
    // CP+SP: [sp*B, N/sp*(dc+1)] -> [cp, sp*B, N/sp*(dc+1)]
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.contextParallelInfo.rank;
    allGatherParam.rankSize = param.contextParallelInfo.rankIds.size();
    allGatherParam.backend = param.contextParallelInfo.defaultBackend;
    param.contextParallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = {GetTensorIdx(tensorMap, param.hasAttnInnerSp ? \
                                              "intermediate_go_lse_all2all" : "intermediate_go_lse")};
    allGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_lse_allgather")};
    allGatherNode.inTensorReshapeFuncs.resize(allGatherNode.inTensorIds.size());
    allGatherNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum == 3) { // 3: number of dimensions of the old shape
            // CP: [B, N, dc+1] -> [B, N*(dc+1)]
            newShape.dimNum = 2; // 2: number of dimensions of the new shape
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: [B, N, dc+1] -> [B, N*(dc+1)]
        } else {
            newShape = oldShape;
        }
    };
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddGoLseSplit(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
                          std::map<std::string, uint32_t>& tensorMap)
{
    // CP: [cp, B, N*(dc+1)] -> [cp, B, N, dc+1] -> [cp, B, N, dc] [cp, B, N, 1]
    // SP: [sp*B, N/sp*(dc+1)] -> [sp, B, N/sp, dc+1] -> [sp, B, N/sp, dc] [sp, B, N/sp, 1]
    // CP+SP: [cp, sp*B, N/sp*(dc+1)] -> [cp*sp, B, N/sp, dc+1] -> [cp*sp, B, N/sp, dc] [cp*sp, B, N/sp, 1]
    atb::Node splitNode;
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 3; // 3: position os dc+1
    splitParam.splitSizes = {param.kvLoraRank, 1};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitParam, &splitNode.operation));
    splitNode.inTensorIds = {GetTensorIdx(tensorMap, param.contextParallelInfo.IsEnabled() ? \
        "intermediate_go_lse_allgather" : "intermediate_go_lse_all2all")};
    splitNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_t"),
                              GetTensorIdx(tensorMap, "intermediate_lse_t")};
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // 4: number of dimensions of the new shape
        if (oldShape.dimNum == 2) { // 2: number of dimensions of the old shape
            // SP: [sp*B, N/sp*(dc+1)] -> [sp, B, N/sp, dc+1]
            newShape.dims[0] = param.attnSpSize;
            newShape.dims[1] = oldShape.dims[0] / param.attnSpSize; // 1: B
            newShape.dims[2] = param.pageAttentionParam.headNum / param.attnSpSize; // 2: N/sp
            newShape.dims[3] = param.kvLoraRank + 1; // 3: dc+1
        } else if (oldShape.dimNum == 3) {
            // CP: [cp, B, N*(dc+1)] -> [cp, B, N, dc+1]
            // CP+SP: [cp, sp*B, N/sp*(dc+1)] -> [cp*sp, B, N/sp, dc+1]
            newShape.dims[0] = param.attnSpSize * param.contextParallelInfo.rankIds.size();
            newShape.dims[1] = oldShape.dims[1] / param.attnSpSize; // 1: B
            newShape.dims[2] = oldShape.dims[2] / (param.kvLoraRank + 1); // 2: N/sp
            newShape.dims[3] = param.kvLoraRank + 1; // 3: dc+1
        }
    };
    opGraph.nodes.push_back(splitNode);
    return atb::NO_ERROR;
}

atb::Status CreateCastToFP32(atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap)
{
    atb::Node lseCastNode;
    atb::infer::ElewiseParam lseCastParam;
    lseCastParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    lseCastParam.outTensorType = ACL_FLOAT;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(lseCastParam, &lseCastNode.operation));
    lseCastNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_lse_t")};
    lseCastNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_lse_fp32")};
    opGraph.nodes.push_back(lseCastNode);

    atb::Node goCastNode;
    atb::infer::ElewiseParam goCastParam;
    goCastParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    goCastParam.outTensorType = ACL_FLOAT;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(goCastParam, &goCastNode.operation));
    goCastNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_t")};
    goCastNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_go_fp32")};
    opGraph.nodes.push_back(goCastNode);

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddFaUpdate(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
    std::map<std::string, uint32_t>& tensorMap)
{
    atb::Node faUpdateNode;
    atb::infer::FaUpdateParam faUpdateParam;
    faUpdateParam.sp = param.contextParallelInfo.IsEnabled() ?
        param.attnSpSize * param.contextParallelInfo.rankIds.size() : param.attnSpSize;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(faUpdateParam, &faUpdateNode.operation));
    faUpdateNode.inTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_lse_fp32", "intermediate_go_fp32"})};
    faUpdateNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_fa_update_out_fp32")};
    faUpdateNode.inTensorReshapeFuncs.resize(faUpdateNode.inTensorIds.size());

    faUpdateNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: [sp, b, n/tp] -> [sp, b*n/tp]
    };
    faUpdateNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: [sp, b, n/tp] -> [sp, b*n/tp]
        newShape.dims[2] = oldShape.dims[3]; // 2,3: [sp, b, n/tp, dc] -> [sp, b*n/tp, dc]
    };
    opGraph.nodes.push_back(faUpdateNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status CreateCastToFP16(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
    std::map<std::string, uint32_t>& tensorMap)
{
    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_fa_update_out_fp32")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_o")};
    castNode.inTensorReshapeFuncs.resize(castNode.inTensorIds.size());
    // faUpdateNode.inTensorIds.size() is 4, set 0~2 reshape func
    castNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        int tpHeadNum = param.pageAttentionParam.headNum / param.attnSpSize;
        // [B*N/Tp, D] -> [B, N/tp, D]
        newShape.dimNum = 3; // 3: number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0] / tpHeadNum;
        newShape.dims[1] = tpHeadNum;
        newShape.dims[2] = oldShape.dims[1]; // 2: dimensions of D
    };
    opGraph.nodes.push_back(castNode);
    return atb::NO_ERROR;
}

// SP: GO [B,N,dc] + LSE [B,N,dc] -> O [B,N/tp,dc]
// CP: GO [B,N/tp,dc] + LSE [B,N/tp,dc] -> O [B,N/tp,dc]
// CP+SP: GO [B,N,dc] + LSE [B,N,dc] -> O [B,N/tp,dc]
template <typename NormParamType>
atb::Status AddDecodeUpdate(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
    std::map<std::string, uint32_t>& tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(AddGoLseConcat(opGraph, tensorMap));
    if (param.hasAttnInnerSp) {
        CHECK_OPERATION_STATUS_RETURN(AddAll2All(param, opGraph, tensorMap));
    }
    if (param.contextParallelInfo.IsEnabled()) {
        CHECK_OPERATION_STATUS_RETURN(AddAllGather(param, opGraph, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(AddGoLseSplit(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(CreateCastToFP32(opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddFaUpdate(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(CreateCastToFP16(param, opGraph, tensorMap));
    return atb::NO_ERROR;
}

template atb::Status AddDecodeUpdate(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddAll2All(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddAllGather(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddGoLseSplit(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddFaUpdate(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status CreateCastToFP16(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);

template atb::Status AddDecodeUpdate(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddAll2All(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddAllGather(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddGoLseSplit(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status AddFaUpdate(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);
template atb::Status CreateCastToFP16(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam& opGraph, std::map<std::string, uint32_t>& tensorMap);

} // namespace common
} // namespace atb_speed