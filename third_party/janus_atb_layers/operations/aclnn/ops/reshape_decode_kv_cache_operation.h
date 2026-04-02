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

#ifndef ATB_SPEED_PLUGIN_ACLNN_RESHAPE_DECODE_KV_CACHE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_RESHAPE_DECODE_KV_CACHE_OPERATION_H
 
 #include "operations/aclnn/core/acl_nn_operation.h"
 #include "operations/aclnn/core/acl_nn_operation_cache.h"
 
 
 namespace atb_speed::common {
 
     /// A struct defines `aclnnMoeFusedAddTopkWithImplModeGetWorkspaceSize` operation parameter.
     
     struct AclNNReshapeDecodeKvCacheParam {
        bool enableXattention = false;
     };
     
     class ReshapeDecodeKvCacheOperation : public AclNNOperation {
     public:
         explicit ReshapeDecodeKvCacheOperation(const std::string &name, AclNNReshapeDecodeKvCacheParam param);
         ~ReshapeDecodeKvCacheOperation() override;
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
        AclNNReshapeDecodeKvCacheParam param_;
        std::string opName_;
     };
 }  // namespace atb_speed::common
 
#endif  // ATB_SPEED_PLUGIN_ACLNN_RESHAPE_DECODE_KV_CACHE_OPERATION_H