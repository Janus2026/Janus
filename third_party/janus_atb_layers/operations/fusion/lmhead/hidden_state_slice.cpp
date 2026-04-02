/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 */

#include "atb_speed/utils/check_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/lmhead/hidden_state_slice.h"

namespace atb_speed {
namespace common {

enum HiddenStateSliceTensorIdx : uint32_t {
    IN_HIDDENSTATES = 0,
    OUT_HIDDEN_STATES,
};

static const uint64_t IN_TENSOR_COUNT = 1;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 1;
static const uint64_t NUM3 = 3; // num3

atb::Status HiddenStateSlice(const HiddenStateSliceParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = 0;

    atb::Node sliceNode;
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets.resize(NUM3);
    sliceParam.offsets[0] = param.rank;
    sliceParam.offsets[1] = 0;
    sliceParam.offsets[2] = 0; // 2: hidden_state
    sliceParam.size.resize(NUM3);
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    sliceParam.size[2] = -1; // 2
    CREATE_OPERATION(sliceParam, &sliceNode.operation);
    sliceNode.inTensorIds = {HiddenStateSliceTensorIdx::IN_HIDDENSTATES};
    sliceNode.outTensorIds = {HiddenStateSliceTensorIdx::OUT_HIDDEN_STATES};

    opGraph.nodes.push_back(sliceNode);

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = inTensorDescs.at(IN_HIDDENSTATES).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(IN_HIDDENSTATES).format;
        outTensorDescs.at(0).shape.dimNum = 2;  // 2: translatedtensortranslated2
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_HIDDENSTATES).shape.dims[0] / param.world_size;
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_HIDDENSTATES).shape.dims[1];
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}
}  // namespace common
}  // namespace atb_speed