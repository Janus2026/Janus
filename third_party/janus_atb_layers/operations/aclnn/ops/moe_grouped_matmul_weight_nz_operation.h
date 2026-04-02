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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_GROUPED_MATMUL_WEIGHT_NZ_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_GROUPED_MATMUL_WEIGHT_NZ_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "aclnn_moe_grouped_matmul_weight_nz.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
/// This class defines an customized operator that consists of a group of matrix multiplications.
/// Meanwhile, this operator supports different quantization types.
///
/// This class makes uses of `aclnnMoeGroupedMatmulWeightNZGetWorkspaceSize` and `aclnnMoeGroupedMatmulWeightNZ`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name                    | Dtype | Shape |
/// ------------------------|-------|-------|
/// input                   | *     | [m,k] |
/// weight                  | *     | [e,n,k] if `transposeB` is true; otherwise, [e,k,n] |
/// groupList               | int64 | [e,2] |
/// * Note: the format type of weight must be fractal_nz; group list be output of initroutingv3 key-value mode
///
/// Outputs of the operator:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// output | float16 or bfloat16 | [m,n] |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_GROUP_LIST,
///     OUT,
/// };
///
/// atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
/// transposeB = transposeB;
/// gmmNode.operation = new atb_speed::common::MoeGroupedMatmulWeightNZOperation("gmmNode", gmmParam);
/// gmmNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_GROUP_LIST};
/// gmmNode.outTensorIds = {OUT};
/// \endcode

class MoeGroupedMatmulWeightNZOperation: public AclNNOperation {
public:
   explicit MoeGroupedMatmulWeightNZOperation(const std::string &name, bool transposeB);
    ~MoeGroupedMatmulWeightNZOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc,bool is_bf16);

    std::vector<aclTensor *> yTensorVector;
    std::vector<std::vector<aclTensor *>> inputVectorOfTensor;
    bool transposeB;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_GROUPED_MATMUL_WEIGHT_NZ_OPERATION_H
