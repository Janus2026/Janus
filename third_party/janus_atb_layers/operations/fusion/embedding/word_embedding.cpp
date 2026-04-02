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

#include "operations/fusion/embedding/word_embedding.h"
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {

enum WordEmbeddingTensorIdx : uint32_t {
    IN_EMBEDDING_WEIGHTS = 0,
    IN_INPUT_IDS,
    OUT_HIDDEN_STATES,
    INTERMEDIATE_GATHER,
    INTERMEDIATE_ALLGATHER_OUT_ID,
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_NO_ALL_GATHER_COUNT = 0;
static const uint64_t INTERMEDIATE_TENSOR_ALL_GATHER_COUNT = 2;
static const uint64_t NODE_NO_ALL_GATHER_COUNT = 1;
static const uint64_t NODE_ALL_GATHER_COUNT = 3;

atb::Status WordEmbedding(const WordEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.tensorParallelInfo.worldSize > 1 ? \
        INTERMEDIATE_TENSOR_ALL_GATHER_COUNT : INTERMEDIATE_TENSOR_NO_ALL_GATHER_COUNT;
    opGraph.name = "WordEmbedding";

    atb::Node inputIdEmbeddingNode;
    atb::infer::GatherParam inputembedinggatherparam;
    inputembedinggatherparam.axis = param.axis;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(inputembedinggatherparam, &inputIdEmbeddingNode.operation));
    inputIdEmbeddingNode.inTensorIds = {
        WordEmbeddingTensorIdx::IN_EMBEDDING_WEIGHTS, WordEmbeddingTensorIdx::IN_INPUT_IDS
    };
    inputIdEmbeddingNode.outTensorIds = {
        param.tensorParallelInfo.worldSize > 1 ? \
        WordEmbeddingTensorIdx::INTERMEDIATE_GATHER : WordEmbeddingTensorIdx::OUT_HIDDEN_STATES
    };
    opGraph.nodes.push_back(inputIdEmbeddingNode);

    if (param.tensorParallelInfo.worldSize > 1) {
        LinearParallelParam parallelParam;
        parallelParam.parallelType = COLUMN_PARALLEL;
        parallelParam.tensorParallelInfo = param.tensorParallelInfo;
        parallelParam.unpadInputs = param.unpadInputs;

        std::map<std::string, uint32_t> tensorMap = {
            {"intermediate_linear_out", WordEmbeddingTensorIdx::INTERMEDIATE_GATHER},
            {"intermediate_sync_out", WordEmbeddingTensorIdx::INTERMEDIATE_ALLGATHER_OUT_ID},
            {"out", WordEmbeddingTensorIdx::OUT_HIDDEN_STATES}};

        CHECK_OPERATION_STATUS_RETURN(AddCommunicationOp(parallelParam, opGraph, tensorMap));
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = inTensorDescs.at(IN_EMBEDDING_WEIGHTS).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(IN_EMBEDDING_WEIGHTS).format;
        if (param.unpadInputs) {
            outTensorDescs.at(0).shape.dimNum = 2;  // 2: translatedtensortranslated2
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_INPUT_IDS).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = CheckIntMulOverFlow(
                inTensorDescs.at(IN_EMBEDDING_WEIGHTS).shape.dims[1], param.tensorParallelInfo.worldSize);
        } else {
            outTensorDescs.at(0).shape.dimNum = 3;  // 3: translatedtensortranslated3
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_INPUT_IDS).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_INPUT_IDS).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] =  // 2: translated2translated
                CheckIntMulOverFlow(
                    inTensorDescs.at(IN_EMBEDDING_WEIGHTS).shape.dims[1],
                    param.tensorParallelInfo.worldSize);
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}
}  // namespace common
}  // namespace atb_speed