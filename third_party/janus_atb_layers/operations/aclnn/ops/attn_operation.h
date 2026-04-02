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
#ifndef ATTN_OPERATION_H
#define ATTN_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"
#include "cstring"
namespace atb_speed {
namespace common {
struct AclNNAttnParam {
    /// A flag indicating whether the model use mask
    bool hasMask = false;
    /// A flag indicating whether the model use FA
    bool isFA = false;
    /// A flag indicating whether the model prefills
    bool isPrefill = false;
    /// A flag indicating whether the model is kvcache int8 compressed
    bool hasKVQuant = false;
    /// A flag indicating whether the model has kvcache compressed offset weight
    bool hasQuantOffset = false;
    /// enable Prefix Attn
    bool enablePrefixAttn = false;
    /// the number of head
    int64_t headNum = 0;
    /// the number of kvHead
    int64_t kvHeadNum = 0;
    /// the number of headDim
    int64_t headDim = 0;
    /// represent high performance/accuracy, dafault 1 (high performance)
    int64_t innerPrecise = 1;
    /// max number of tokens in each block page attention stored in KV cache
    int64_t blockSize = 128;
};

/// This class defines an operator that calculates the attention including FA and PA.
///
/// This class makes uses of `aclnnFusedInferAttentionScoreV2GetWorkspaceSize` and
/// `aclnnFusedInferAttentionScoreV2` from AscendCL Api.
///
/// Inputs to the operator:
/// Name                      | Dtype                       | Shape                               |
/// --------------------------|-----------------------------|-------------------------------------|
/// input                     | *                           | [batchsize, headNum, dim]           |
/// query                     | float16, bfloat16 or int8   | [batchsize, headNum, dim]           |
/// key                       | float16, bfloat16 or int8   | [blocknum, blocksize, headNum, dim] |
/// value                     | float16, bfloat16 or int8   | [blocknum, blocksize, headNum, dim] |
/// actualSeqLengthsOptional  | int64                       | [bs]                                |
/// blockTableOptional        | float16, bfloat16 or float32| [bs,blocknum]                       |
/// antiquantScaleOptional    | float16, bfloat16 or float32| [bs,dim]                            |
/// antiquantOffsetOptional   | float16, bfloat16 or float32| [bs,dim]                            |
///
/// Outputs of the operator:
/// Name                      | Dtype                       | Shape                               |
/// --------------------------|-----------------------------|-------------------------------------|
/// output                    | float16, bfloat16 or int8   | [batchsize, headNum, dim]           |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///    QUERY,
///    KEY,
///    VALUE,
///    SEQ_LEN,
///    BLOCK_TABLE,
///    DEQUANT_SCALE,
///    DEQUANT_OFFSET,
///    OUT,
///};
///
/// atb::Node &attnNode = opGraph.nodes.at(nodeId++);
/// attnNode.operation = new atb_speed::common::AttnOperation("AttentionNode");
/// attnNode.inTensorIds = {QUERY, KEY, VALUE, SEQ_LEN, BLOCK_TABLE, DEQUANT_SCALE, DEQUANT_OFFSET};
/// attnNode.outTensorIds = {OUT};
/// \endcode

class AttnOperation : public AclNNOperation {
public:
    explicit AttnOperation(const std::string &name, AclNNAttnParam param);
    ~AttnOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    int ProcessSeqLengthTensor(atb::Tensor &tensor);

private:
    aclTensor *tensorsOfValue[1]{nullptr};
    aclTensor *tensorsOfKey[1]{nullptr};
    AclNNAttnParam param_;
    std::string opName_;
};
} // namespace common
} // namespace atb_speed
#endif
