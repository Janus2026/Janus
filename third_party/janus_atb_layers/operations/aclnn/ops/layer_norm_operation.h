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

#ifndef ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {

    /// A struct defines `aclnnLayerNormWithImplModeGetWorkspaceSize` operation parameter.
    struct AclNNLayerNormParam {
        /// Indicates a value added to the denominator for numerical stability.
        float layerNormEps = 0;
        /// Indicates the start of normalization axis.
        int beginNormAxis = 0;
        /// Indicates the number of normalization axes.
        int normAxes = 1;
        /// Indicates the accuracy implementation mode in execution.
        ///
        /// 0: high accuracy mode.
        /// 1: high performance mode.
        /// 2: keep dtype of `float16` in execution.
        int64_t layerNormImplMode = 0;
        /// Indicates whether inputs include a bias tensor.
        bool hasBias = true;
    };

    /// This class defines a matrix operation that applies Layer Normalization over a mini-batch of inputs.
    ///
    /// This class makes use of `aclnnLayerNormGetWorkspaceSize` and `aclnnLayerNormWithImplModeGetWorkspaceSize`
    /// from the AscendCL API.
    ///
    /// Operation's Inputs: \n
    /// | Name   | Dtype                    | Shape                   | \n
    /// |--------|--------------------------|-------------------------| \n
    /// | input  | float32/float16/bfloat16 | [-1,…,-1]               | \n
    /// | weight | float32/float16/bfloat16 | [beginNormAxis:]/[1:-1] | \n
    /// | bias   | float32/float16/bfloat16 | [beginNormAxis:]/[1:-1] | \n
    ///
    /// Operation's Outputs: \n
    /// | Name   | Dtype                    | Shape                   | \n
    /// |--------|--------------------------|-------------------------| \n
    /// | output | float32/float16/bfloat16 | [-1,…,-1]               | \n
    ///
    /// Example:
    /// \code
    /// enum TensorIdx : uint32_t {
    ///     IN_INPUT = 0,
    ///     IN_WEIGHT,
    ///     IN_BIAS,
    ///     OUT,
    /// };
    ///
    /// atb::Node layerNormNode;
    /// AclNNLayerNormParam layerNormParam;
    /// layerNormParam.layerNormEps = 1e-5;
    /// layerNormParam.beginNormAxis = -1;
    /// layerNormParam.normAxes = 1;
    /// layerNormParam.hasBias = true;
    /// layerNormNode.inTensorIds = { IN_INPUT, IN_WEIGHT, IN_BIAS };
    /// layerNormNode.outTensorIds = { OUT };
    /// layerNormNode.operation = new atb_speed::common::LayerNormOperation("layerNormNode", layerNormParam);
    ///
    /// atb::GraphParam opGraph;
    /// opGraph.nodes.push_back(layerNormNode);
    /// \endcode
    class LayerNormOperation : public AclNNOperation {
    public:
        explicit LayerNormOperation(const std::string &name, AclNNLayerNormParam param);
        ~LayerNormOperation() override;
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDesc,
            atb::SVector<atb::TensorDesc> &outTensorDesc
        ) const override;
        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        int SetAclNNWorkspaceExecutor() override;
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
        atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
        atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
        atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
        virtual std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, size_t tensorIdx);

    private:
        AclNNLayerNormParam param_;
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H
