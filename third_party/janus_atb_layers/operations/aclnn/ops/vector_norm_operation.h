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

#ifndef ATB_SPEED_PLUGIN_ACLNN_VECTOR_NORM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_VECTOR_NORM_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {
    /// A struct defines `VectorNorm`'s parameter.
    struct AclNNVectorNormParam {
        /// scalar param, default is null
        aclScalar *ord = nullptr;
    };

    /// vector norm operation is used for moe scenarios, for example:
    /// Operation's Inputs:
    /// Name   | Dtype               | Shape |
    /// -------|---------------------|-------|
    /// input  | float16 or bfloat16 | [m,n] |
    ///
    /// Operation's Outputs:
    /// Name   | Dtype               | Shape |
    /// -------|---------------------|-------|
    /// output | float16 or bfloat16 | [m,n] |
    ///
    /// Example:
    /// \code
    /// enum TensorIdx : uint32_t {
    ///     INPUT = 0,
    ///     OUT,
    /// };
    /// atb::Node &vectorNormNode = opGraph.nodes.at(nodeId++);
    /// atb_speed::common::AclNNVectorNormParam vectorNormParam;
    /// vectorNormNode.operation = new atb_speed::common::VectorNormOperation("vectorNormOperation", vectorNormParam);
    /// vectorNormNode.inTensorIds = {INPUT};
    /// vectorNormNode.outTensorIds = {OUT};
    /// \endcode
    class VectorNormOperation : public AclNNOperation {
    public:
        /// Class constructor.
        /// Initialize an `VectorNormOperation` pointer.
        /// \param name The name of the AclNN operation.
        /// \param param The param of the AclNN operation.
        explicit VectorNormOperation(const std::string &name, AclNNVectorNormParam param);

        ~VectorNormOperation() override;

        /// infer shape function.
        /// \param inTensorDesc inTensorDesc of AclNN operation.
        /// \param outTensorDesc outTensorDesc of the AclNN operation.
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDesc,
            atb::SVector<atb::TensorDesc> &outTensorDesc
        ) const override;

        /// get input num
        /// \return operation input num
        [[nodiscard]] uint32_t GetInputNum() const override;

        /// get output num
        /// \return operation output num
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        /// Prepare the operation's input tensors.
        /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
        /// \return A status code that indicates whether variantPack was created successfully.
        atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;

        /// Call AclNN operation's first phase API to get work space size and `aclOpExecutor`.
        /// \return The return value of AclNN's first phase API.
        int SetAclNNWorkspaceExecutor() override;

        /// Call AclNN operation's second phase API to execute the operation.
        /// \return The return value of AclNN's second phase API.
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

        /// Prepare the operation's input tensors.
        /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
        /// \return A status code that indicates whether variantPack was created successfully.
        atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

        /// Prepare the operation's output tensors.
        /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
        /// \return A status code that indicates whether variantPack was created successfully.
        atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

        /// Prepare the operation's create tensors.
        /// \param atbTensor An `atb::Tensor` object containing tensor info passed through ATB framework.
        /// \param tensorIdx The id of tensor.
        /// \return A status code that indicates whether variantPack was created successfully.
        virtual std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, size_t tensorIdx);

        /// The dims of vector_norm.
        aclIntArray *dims = nullptr;

    private:
        /// An `AclNNVectorNormParam` object that can be reused within the current operation object.
        AclNNVectorNormParam param_;

        /// A human identifiable name for the operation's name.
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_VECTOR_NORM_OPERATION_H
