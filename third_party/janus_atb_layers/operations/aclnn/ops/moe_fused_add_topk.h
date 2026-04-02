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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_FUSED_ADD_TOPK_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_FUSED_ADD_TOPK_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"


namespace atb_speed::common {

    /// A struct defines `aclnnMoeFusedAddTopkWithImplModeGetWorkspaceSize` operation parameter.
    
    struct AclNNMoeFusedAddTopkParam {
        uint32_t groupNum = 1;      //translated0,translated.
        uint32_t groupTopk = 1;     //translated0,translated.
        uint32_t n = 1;     //topN:translated. translated0,
        uint32_t k = 0;     //topK:translated. translated0.
        uint32_t activationType = 0;  //translated,translated0 (ACTIVATION_SIGMOID);
        bool isNorm = true; //translated
        float scale = 1.0f; //isNormtranslatedtruetranslated,translated.
        bool enableExpertMapping = false;   //translated.
    };
    
    class MoeFusedAddTopkOperation : public AclNNOperation {
    public:
        explicit MoeFusedAddTopkOperation(const std::string &name, AclNNMoeFusedAddTopkParam param);
        ~MoeFusedAddTopkOperation() override;
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

    private:
        AclNNMoeFusedAddTopkParam param_;
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_FUSED_ADD_TOPK_OPERATION_H

    // aclnnMoeFusedAddTopkGetWorkspaceSize(
    //     const aclTensor* x, const aclTensor* addNum, const aclTensor* mappingNum, const aclTensor* mappingTable,
    //     uint32_t groupNum, uint32_t groupTopk, uint32_t topN, uint32_t topK, uint32_t activateType, 
    //     bool isNorm, float scale, bool enableExpertMapping, aclTensor* y, aclTensor* indices,
    //     uint64_t* workspaceSize, aclOpExecutor** executor);

    /// This class defines a matrix operation that applies Layer Normalization over a mini-batch of inputs.
    ///
    /// This class makes use of `aclnnMoeFusedAddTopkGetWorkspaceSize` and `aclnnMoeFusedAddTopkWithImplModeGetWorkspaceSize`
    /// from the AscendCL API.
    ///
    /// Operation's Inputs: \n
    /// | Name   | Dtype                    | Shape                   | \n
    /// |--------|--------------------------|-------------------------| \n
    /// | x      | float32/float16/bfloat16 | [a,b]                   | \n
    /// | addNum | float32/float16/bfloat16 | [b]                     | \n
    /// | mappingNum     | INT32 | ?   | \n
    /// | mappingTable   | INT32 | ?   | \n
    ///
    /// Operation's Outputs: \n
    /// | Name   | Dtype                    | Shape                   | \n
    /// |--------|--------------------------|-------------------------| \n
    /// | y      | float32                   | [a,b]                  | \n
    ///
