/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 */

#include <atb/atb_infer.h>

#include "operations/fusion/utils.h"
#include "parallel_lmhead_all2all.h"

namespace atb_speed {
namespace common {

template <class T>
atb::Status CreateLmHeadLinearNode(const LmHeadParam &param, atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &lmHeadLinearNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam lmHeadLinearParam;
    lmHeadLinearParam.transposeB = param.linearParallelParam.fusionLinearParam.transposeType == TRANSPOSE;
    if (!lmHeadLinearParam.transposeB) {
        ATB_SPEED_LOG_ERROR("The lmhead linear node in lmhead-all2all doesn't support transposeType: "
            << param.linearParallelParam.fusionLinearParam.transposeType
            << " The value must be " << TRANSPOSE << ".");
        return atb::ERROR_INVALID_PARAM;
    }
    lmHeadLinearParam.hasBias = false;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(lmHeadLinearParam, &lmHeadLinearNode.operation));
    lmHeadLinearNode.inTensorIds = {config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_ID};
    lmHeadLinearNode.outTensorIds = {config.INTERMEDIATE_LMLINEAR_OUT_ID};
    
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateTransPose1Node(const LmHeadParam &param, atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &transPose1Node = opGraph.nodes.at(nodeId++);
    atb::infer::TransposeParam transParam1;
    transParam1.perm = { 0, 2, 1 };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transParam1, &transPose1Node.operation));
    transPose1Node.inTensorIds = { config.INTERMEDIATE_LMLINEAR_OUT_ID };
    transPose1Node.outTensorIds = { config.INTERMEDIATE_TRANS1_OUT_ID };
    transPose1Node.inTensorReshapeFuncs.resize(transPose1Node.inTensorIds.size());
    transPose1Node.inTensorReshapeFuncs.at(0) = [param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: rank, token, vocab_size
        newShape.dims[0] = param.linearParallelParam.tensorParallelInfo.worldSize;
        newShape.dims[1] = oldShape.dims[0] / param.linearParallelParam.tensorParallelInfo.worldSize;
        newShape.dims[2] = oldShape.dims[1]; // 2: vocab_size
    };
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateAllToAllNode(const LmHeadParam &param, atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &allToAllNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllToAllParam allToAllParam;
    allToAllParam.rank = param.linearParallelParam.tensorParallelInfo.rank;
    allToAllParam.rankSize = param.linearParallelParam.tensorParallelInfo.worldSize;
    allToAllParam.backend = param.linearParallelParam.tensorParallelInfo.backend;
    allToAllParam.hcclComm = param.linearParallelParam.tensorParallelInfo.hcommInfo;
    allToAllParam.commDomain = param.linearParallelParam.tensorParallelInfo.commDomain;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allToAllParam, &allToAllNode.operation));
    allToAllNode.inTensorIds = {config.INTERMEDIATE_TRANS1_OUT_ID};
    allToAllNode.outTensorIds = {config.INTERMEDIATE_ALLTOALLTP_OUT_ID};
    allToAllNode.inTensorReshapeFuncs.resize(allToAllNode.inTensorIds.size());
    allToAllNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: rank* vocab_size, token
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: token
    };
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateTransPose2Node(const LmHeadParam &param, atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &transPose2Node = opGraph.nodes.at(nodeId++);
    atb::infer::TransposeParam trans2Param;
    trans2Param.perm = { 1, 0 };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(trans2Param, &transPose2Node.operation));
    transPose2Node.inTensorIds = { config.INTERMEDIATE_ALLTOALLTP_OUT_ID };
    transPose2Node.outTensorIds = { config.OUT_LOGITS_ID };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
        outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0] * \
                                                                param.linearParallelParam.tensorParallelInfo.worldSize;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] / \
                                                                param.linearParallelParam.tensorParallelInfo.worldSize;
        return atb::NO_ERROR;
    };
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateParallelLmHeadAllToAllBase(const LmHeadParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);
    opGraph.name = "Parallel_LmHead";

    size_t nodeId = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateLmHeadLinearNode(param, opGraph, config, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateTransPose1Node(param, opGraph, config, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateAllToAllNode(param, opGraph, config, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateTransPose2Node(param, opGraph, config, nodeId));

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

class ParallelLmHeadAllToAllConfig {
public:

    uint64_t inTensorNum = 7;
    uint64_t outTensorNum = 1;
    uint64_t interTensorNum = 3;
    uint64_t nodeCount = 4;

    enum ParallelLmHeadAllToAllId : unsigned int {
        IN_HIDDENSTATES_ID = 0,
        IN_WEIGHT_ID,
        IN_SCALE,
        IN_OFFSET,
        IN_DESCALE,
        IN_BIAS,
        IN_COMPRESS_IDX,
        OUT_LOGITS_ID,
        INTERMEDIATE_LMLINEAR_OUT_ID,
        INTERMEDIATE_TRANS1_OUT_ID,
        INTERMEDIATE_ALLTOALLTP_OUT_ID,
    };
};

atb::Status ParallelLmHeadAllToAll(const LmHeadParam &param, atb::Operation **operation)
{
    ParallelLmHeadAllToAllConfig parallelLmHeadAllToAllConfig;
    return CreateParallelLmHeadAllToAllBase(param, operation, parallelLmHeadAllToAllConfig);
}
} // namespace common
} // namespace atb_speed