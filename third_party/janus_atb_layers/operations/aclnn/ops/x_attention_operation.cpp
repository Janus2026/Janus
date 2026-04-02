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

 #include <cstring>
 #include <iostream>
 #include <securec.h>
 #include <sstream>
 #include <vector>
 #include <unistd.h>
 #include "acl/acl.h"
 #include "atb_speed/log.h"
 #include "atb_speed/utils/timer.h"
 #include "operations/aclnn/utils/utils.h"
 
 #include "x_attention_operation.h"
 #include "aclnn_x_attention.h"
 
 
 
 namespace atb_speed {
 namespace common {
 
 XAttentionOperation::XAttentionOperation(
     const std::string &name,
     AclNNXAttentionParam param) : AclNNOperation(name), param_(param) {}
 
 XAttentionOperation::~XAttentionOperation()
 {
     ATB_SPEED_LOG_DEBUG("XAttentionOperation deconstructor");
     this->DestroyOperation();
 }
 
 uint32_t XAttentionOperation::GetInputNum() const
 {
     uint32_t inputNum = 8;
     return inputNum;
 }
 
 uint32_t XAttentionOperation::GetOutputNum() const
 {
     return DIM1;
 }
 
 atb::Status XAttentionOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                         atb::SVector<atb::TensorDesc> &outTensorDescs) const
 {
     ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");

    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (uint32_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
     ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
     return 0;
 }
 
 
 int XAttentionOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
 {
     AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
     aclnnVariantPack.aclInTensors.resize(GetInputNum());
     for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
             
         std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
         aclnnTensor->tensorIdx = i;
         aclnnTensor->needUpdateTensorDataPtr = true;
         aclnnTensor->atbTensor = variantPack.inTensors.at(i);
         aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);
 
         aclnnTensor->tensor = aclCreateTensor(
             aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
             aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
             aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
             aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);
 
         if (aclnnTensor->tensor == nullptr) {
             ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
             return atb::ERROR_INTERNAL_ERROR;
         }
         aclnnVariantPack.aclInTensors[i] = aclnnTensor;
     }
     return atb::NO_ERROR;
 }
 
 int XAttentionOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
 {
     AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
     aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
     for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
         std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
         aclnnTensor->tensorIdx = i;
         aclnnTensor->needUpdateTensorDataPtr = true;
         aclnnTensor->atbTensor = variantPack.outTensors.at(i);
         aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);
         aclnnTensor->tensor = aclCreateTensor(
             aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
             aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
             aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
             aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);
         if (aclnnTensor->tensor == nullptr) {
             ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
             return atb::ERROR_INTERNAL_ERROR;
         }
         aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
     }
     return atb::NO_ERROR;
 }
 
 
 int XAttentionOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
 {
     ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
     int ret = 0;
     ret = CreateAclNNInTensorVariantPack(variantPack);
     if (ret != 0) {
         ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
         return ret;
     }
     ret = CreateAclNNOutTensorVariantPack(variantPack);
     if (ret != 0) {
         ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
         return ret;
     }
     ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
     return atb::NO_ERROR;
 }
 
 
 int XAttentionOperation::SetAclNNWorkspaceExecutor()
 {
     ATB_SPEED_LOG_DEBUG(opName_ << " x attention SetAclNNWorkspaceExecutor start");
     AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
     aclTensor *shared_block_table = nullptr;
     int ret = aclnnXAttentionGetWorkspaceSize(
             aclnnVariantPack.aclInTensors[DIM0]->tensor, 
             aclnnVariantPack.aclInTensors[DIM1]->tensor,
             aclnnVariantPack.aclInTensors[DIM2]->tensor,
             aclnnVariantPack.aclInTensors[DIM3]->tensor,
             aclnnVariantPack.aclInTensors[4]->tensor,
             aclnnVariantPack.aclInTensors[5]->tensor,
             aclnnVariantPack.aclInTensors[6]->tensor,
             aclnnVariantPack.aclInTensors[7]->tensor,
             shared_block_table,
             aclnnVariantPack.aclOutTensors.at(0)->tensor, 
             &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
 
     ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                   << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                   << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
     return ret;
 }
 
 
 int XAttentionOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
 {
     ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddmm start");
     int ret = aclnnXAttention(
         workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
     
     ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddmm end, ret:" << ret);
     if (ret != 0) {
         ATB_SPEED_LOG_ERROR("aclrtSynchronizeStream failed, ret: " << ret);
         return ret;
     }
     return ret;
 }
 
 }  // namespace common
 }  // namespace atb_speed
 
 