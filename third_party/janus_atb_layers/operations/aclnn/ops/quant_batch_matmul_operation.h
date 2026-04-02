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
#ifndef ATB_SPEED_PLUGIN_ACLNN_QAUNT_BATCH_MATMUL_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_QAUNT_BATCH_MATMUL_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
/// A struct defines the parameter of `W8A16Operation` and `W4A16Operation`.
struct AclNNWeightQuantBatchMatmulParam {
    /// A flag indicating whether the matmul operation includes a bias tensor.
    bool hasBias = false;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach.
    int quantGroupSize = 0;
    /// A flag indicating whether the second matrix in the matmul operation is transposed.
    bool transposeB = false;
};

/// This class defines a matrix operation that supports per-channel and per-group weight quantization
/// while keeping activations in floating-point format.
///
/// This class makes use of `aclnnQuantMatmulV4GetWorkspaceSize` and `aclnnQuantMatmulV4` from the AscendCL API.
/// This class contains a virtual function called `PreprocessATBInTensor`, which cannot be invoked directly.
/// The `W8A16Operation` and `W4A16Operation` classes inherit from this base class
/// and implement the `PreprocessATBInTensor` function to handle various tensor data types.
class QuantBatchMatmulOperation : public AclNNOperation {
public:
    explicit QuantBatchMatmulOperation(const std::string &name, AclNNWeightQuantBatchMatmulParam param);
    ~QuantBatchMatmulOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    virtual atb::Tensor PreprocessATBInTensor(atb::Tensor atbTensor, int index) = 0;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc);

private:
    AclNNWeightQuantBatchMatmulParam param_;
};
} // namespace common
} // namespace atb_speed
#endif