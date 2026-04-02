
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

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"

#include "operations/aclnn/ops/argmax_operation.h"
#include "operations/fusion/lmhead/parallel_lmhead_all2all.h"
#include "operations/fusion/lmhead/lmhead.h"

namespace atb_speed {
namespace common {

enum LmHeadTensorIdx : uint32_t {
    IN_HIDDENSTATES = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    IN_BIAS,
    IN_COMPRESS_IDX,
    IN_INDICES,
    IN_LOGITS_OFFSET,
    OUT_LOGITS,
};

static const uint64_t IN_TENSOR_COUNT = 9;
static const uint64_t OUT_TENSOR_COUNT = 1;

template <class T>
int64_t AddSlice(atb::GraphParam &opGraph, const LmHeadParam &param, T &config)
{
    atb::Node sliceNode;
    atb::infer::SliceParam slicePassParam;
    if (param.unpadInputs) {
        slicePassParam.offsets = {
            0, CheckIntMulOverFlow(param.hiddenSizePerAttentionHead,
                                   param.linearParallelParam.tensorParallelInfo.rank)
        };
        slicePassParam.size = {-1, param.hiddenSizePerAttentionHead};
    } else {
        slicePassParam.offsets = {
            0, 0, CheckIntMulOverFlow(param.hiddenSizePerAttentionHead,
                                      param.linearParallelParam.tensorParallelInfo.rank)
        };
        slicePassParam.size = {-1, -1, param.hiddenSizePerAttentionHead};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(slicePassParam, &sliceNode.operation));
    if (param.gatherAhead) {
        sliceNode.inTensorIds = {config.INTERMEDIATE_GATHER_OUT};
    } else {
        sliceNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES};
    }
    sliceNode.outTensorIds = {config.INTERMEDIATE_SLICE_OUT};
    opGraph.nodes.push_back(sliceNode);

    return atb::NO_ERROR;
}

template <class T>
int64_t AddLinearParallel(
    atb::GraphParam &opGraph, const LmHeadParam &param,
    T &config, atb_speed::common::LinearParallelType parallelType)
{
    atb::Node linearParallelNode;
    if (param.enableDpOut && param.linearParallelParam.tensorParallelInfo.worldSize > 1 && !param.gatherAhead) {
        CHECK_OPERATION_STATUS_RETURN(ParallelLmHeadAllToAll(param, &linearParallelNode.operation));
    } else {
        CHECK_OPERATION_STATUS_RETURN(LinearParallel(param.linearParallelParam, &linearParallelNode.operation));
    }
    if (parallelType == ROW_PARALLEL) {
        linearParallelNode.inTensorIds = {
            config.INTERMEDIATE_SLICE_OUT, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_BIAS,
            LmHeadTensorIdx::IN_COMPRESS_IDX,
        };
    } else if (param.gatherAhead) {
        linearParallelNode.inTensorIds = {
            config.INTERMEDIATE_GATHER_OUT, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_BIAS,
            LmHeadTensorIdx::IN_COMPRESS_IDX,
        };
    } else {
        linearParallelNode.inTensorIds = {
            LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_BIAS,
            LmHeadTensorIdx::IN_COMPRESS_IDX,
        };
    }
    if (!param.linearParallelParam.isArgmaxlogits) {
        linearParallelNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS};
    } else {
    if (param.gatherAhead) {
        linearParallelNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS + 2,
                                        LmHeadTensorIdx::OUT_LOGITS + 3};
    } else {
        linearParallelNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS + 1,
                                        LmHeadTensorIdx::OUT_LOGITS + 2};
    }
    }
    opGraph.nodes.push_back(linearParallelNode);

    return atb::NO_ERROR;
}

int64_t AddLogitsOffset(atb::GraphParam &opGraph, const LmHeadParam &param)
{
    uint32_t argmaxOutId = LmHeadTensorIdx::OUT_LOGITS + 1;
    if (param.gatherAhead) {
        argmaxOutId = argmaxOutId + 1;
    }
    uint32_t logitSoffsetId = argmaxOutId + 2;
    atb::Node addNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
    addNode.inTensorIds = {argmaxOutId, LmHeadTensorIdx::IN_LOGITS_OFFSET};
    addNode.outTensorIds = {logitSoffsetId};
    opGraph.nodes.push_back(addNode);
    return atb::NO_ERROR;
}

int64_t AddArgMax(atb::GraphParam &opGraph, const LmHeadParam &param)
{
    uint32_t maxOutId = LmHeadTensorIdx::OUT_LOGITS + 2;
    if (param.gatherAhead) {
        maxOutId = maxOutId + 1;
    }
    const uint32_t argmaxMaxlogitsOut = maxOutId + 2;
    atb::Node argmaxNode;
    atb_speed::common::AclNNArgMaxParam argMaxParam;
    argMaxParam.keepdim = true;
    argmaxNode.operation = new atb_speed::common::ArgMaxOperation("argmaxNode", argMaxParam);
    argmaxNode.inTensorIds = {maxOutId};
    argmaxNode.outTensorIds = {argmaxMaxlogitsOut};
    opGraph.nodes.push_back(argmaxNode);
    return atb::NO_ERROR;
}

int64_t AddGatherLogits(atb::GraphParam &opGraph, const LmHeadParam &param)
{
    uint32_t  argmaxMaxlogitsOut = LmHeadTensorIdx::OUT_LOGITS + 4;
    if (param.gatherAhead) {
        argmaxMaxlogitsOut = argmaxMaxlogitsOut + 1;
    }
    uint32_t logitSoffsetId = argmaxMaxlogitsOut - 1;
    uint32_t inT32Logits = argmaxMaxlogitsOut + 1;
    atb::Node gatherNode;
    atb::infer::GatherParam gatherparam;
    gatherparam.axis = 1;
    gatherparam.batchDims = 1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherparam, &gatherNode.operation));
    gatherNode.inTensorIds = {logitSoffsetId, argmaxMaxlogitsOut};
    gatherNode.outTensorIds = {inT32Logits};
    opGraph.nodes.push_back(gatherNode);

    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {inT32Logits};
    castNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS};
    opGraph.nodes.push_back(castNode);
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateLmHead(
    const LmHeadParam &param, atb::Operation **operation, T config,
    atb_speed::common::LinearParallelType parallelType)
{
    uint32_t RESULT_OFFSET_4 = 4;
    uint32_t RESULT_OFFSET_5 = 5;
    uint32_t RESULT_DIM_2 = 2;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    if (param.linearParallelParam.isArgmaxlogits) {
        opGraph.internalTensorNum =
            param.gatherAhead ? config.intermediateTensorCount + RESULT_OFFSET_5
                            : config.intermediateTensorCount + RESULT_OFFSET_4;
    } else {
        opGraph.internalTensorNum =
            param.gatherAhead ? config.intermediateTensorCount : config.intermediateTensorCount - 1;
    }
    opGraph.name = "LmHead";

    if (param.gatherAhead) {
        atb::Node gatherNode;
        atb::infer::GatherParam gatherParam;
        gatherParam.axis = param.unpadInputs ? 0 : 1;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherParam, &gatherNode.operation));
        gatherNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_INDICES};
        gatherNode.outTensorIds = {config.INTERMEDIATE_GATHER_OUT};
        opGraph.nodes.push_back(gatherNode);
    }
    if (parallelType == ROW_PARALLEL) {
        CHECK_OPERATION_STATUS_RETURN(AddSlice(opGraph, param, config));
    }
    CHECK_OPERATION_STATUS_RETURN(AddLinearParallel(opGraph, param, config, parallelType));
    if (param.linearParallelParam.isArgmaxlogits) {
        CHECK_OPERATION_STATUS_RETURN(AddLogitsOffset(opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(AddArgMax(opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(AddGatherLogits(opGraph, param));
    }
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (param.linearParallelParam.isArgmaxlogits) {
            outTensorDescs.at(0).format = inTensorDescs.at(IN_HIDDENSTATES).format;
            outTensorDescs.at(0).dtype = aclDataType::ACL_INT64;
            outTensorDescs.at(0).shape.dimNum = RESULT_DIM_2; // translated [batch_size,1]
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_INDICES).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = 1;
        } else {
            outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDENSTATES);
            CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(IN_HIDDENSTATES).shape.dimNum);
            auto dimLast = inTensorDescs.at(IN_HIDDENSTATES).shape.dimNum - 1;
            if (param.gatherAhead) {
                outTensorDescs.at(0).shape.dims[param.unpadInputs ? 0 : 1] = inTensorDescs.at(IN_INDICES).shape.dims[0];
            } else if (param.enableDpOut && param.linearParallelParam.tensorParallelInfo.worldSize > 1) {
                outTensorDescs.at(0).shape.dims[0] = \
                    inTensorDescs.at(IN_HIDDENSTATES).shape.dims[0] / \
                                                                param.linearParallelParam.tensorParallelInfo.worldSize;
            }
            if (parallelType == COLUMN_PARALLEL) {
                int nDim =
                    param.linearParallelParam.fusionLinearParam.transposeType == TransposeType::TRANSPOSE ? 0 : 1;
                outTensorDescs.at(0).shape.dims[dimLast] = \
                    CheckIntMulOverFlow(inTensorDescs.at(IN_WEIGHT).shape.dims[nDim],
                    param.linearParallelParam.tensorParallelInfo.worldSize);
            } else {
                outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(IN_WEIGHT).shape.dims[0];
            }
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

class LmHeadNoParallelConfig {
public:
    uint64_t nodeCount = 2;
    uint64_t intermediateTensorCount = 1;

    enum LmHeadNoParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

class LmHeadRowParallelConfig {
public:

    uint64_t nodeCount = 3;
    uint64_t intermediateTensorCount = 2;

    enum LmHeadRowParallelTensorIdx : uint32_t {
        INTERMEDIATE_SLICE_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_GATHER_OUT,
    };
};

class LmHeadColumnParallelConfig {
public:

    uint64_t nodeCount = 2;
    uint64_t intermediateTensorCount = 1;

    enum LmHeadColumnParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

atb::Status LmHead(const LmHeadParam &param, atb::Operation **operation)
{
    if (param.linearParallelParam.tensorParallelInfo.worldSize <= 1) {
        LmHeadNoParallelConfig lmHeadNoParallelConfig;
        return CreateLmHead(param, operation, lmHeadNoParallelConfig, UNDEFINED);
    } else if (param.linearParallelParam.parallelType == ROW_PARALLEL) {
        LmHeadRowParallelConfig lmHeadRowParallelConfig;
        return CreateLmHead(param, operation, lmHeadRowParallelConfig, ROW_PARALLEL);
    } else if (param.linearParallelParam.parallelType == COLUMN_PARALLEL) {
        LmHeadColumnParallelConfig lmHeadColumnParallelConfig;
        return CreateLmHead(param, operation, lmHeadColumnParallelConfig, COLUMN_PARALLEL);
    } else {
        ATB_SPEED_LOG_ERROR("LmHead operation doesn't support parallelType: "
            << param.linearParallelParam.parallelType
            << " Possible values are 1 (row parallel) or 2 (column parallel).");
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed