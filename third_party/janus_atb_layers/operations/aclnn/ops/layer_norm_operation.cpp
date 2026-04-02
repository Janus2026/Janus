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

#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_layer_norm.h"
#include "layer_norm_operation.h"


namespace atb_speed::common {

    LayerNormOperation::LayerNormOperation(
        const std::string &name,
        atb_speed::common::AclNNLayerNormParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    LayerNormOperation::~LayerNormOperation()
    {
        ATB_SPEED_LOG_DEBUG("LayerNormOperation deconstruct");
        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @param[in] outTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @return atb::Status
     */
    atb::Status LayerNormOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape start");
        outTensorDesc.at(0).format = inTensorDesc.at(0).format;
        outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
        outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;

        ATB_SPEED_LOG_DEBUG("Check " << opName_ << " input dimNum=" << inTensorDesc.at(0).shape.dimNum);
        for (uint64_t dim = 0; dim < inTensorDesc.at(0).shape.dimNum; ++dim) {
            ATB_SPEED_LOG_DEBUG("input dim" << dim << " shape=" << inTensorDesc.at(0).shape.dims[dim]);
            outTensorDesc.at(0).shape.dims[dim] = inTensorDesc.at(0).shape.dims[dim];
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");
        return atb::NO_ERROR;
    }

    uint32_t LayerNormOperation::GetInputNum() const
    {
        return NUM3;  // inputTensorNum = 3
    }

    uint32_t LayerNormOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status LayerNormOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
        int ret;

        ret = CreateAclNNInTensorVariantPack(variantPack);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
            return ret;
        }

        ret = CreateAclNNOutTensorVariantPack(variantPack);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
        return atb::NO_ERROR;
    }

    atb::Status LayerNormOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    atb::Status LayerNormOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclNnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclNnVariantPack.aclOutTensors.resize(GetOutputNum());
        for (size_t i = 0; i < aclNnVariantPack.aclOutTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " outTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclNnVariantPack.aclOutTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    std::shared_ptr<AclNNTensor> LayerNormOperation::CreateTensor(atb::Tensor atbTensor, size_t tensorIdx)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor start");
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = atbTensor;
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(atbTensor);
        ATB_SPEED_LOG_DEBUG(opName_ << " tensor dtype: " << squeezedAtbTensor.desc.dtype);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype,
            aclnnTensor->strides.data(),
            0,
            squeezedAtbTensor.desc.format,
            squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.deviceData);
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor end");
        return aclnnTensor;
    }

    int LayerNormOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(
            opName_ << " SetAclNNWorkspaceExecutor start"
                    << ", layerNormEps: " << param_.layerNormEps
                    << ", beginNormAxis: " << param_.beginNormAxis
                    << ", layerNormImplMode: " << param_.layerNormImplMode
        );
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        int beginNormAxis =
            param_.beginNormAxis < 0 ?
            aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dimNum + param_.beginNormAxis :
            param_.beginNormAxis;
        uint64_t normalizedMaxDimNum = static_cast<uint64_t>(beginNormAxis + param_.normAxes);
        uint64_t inTensorDimNum = aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dimNum;
        if (normalizedMaxDimNum > inTensorDimNum) {
            std::stringstream ss;
            ss << this->opName_ << " normalized max dimNum " << normalizedMaxDimNum
                                << " > inTensor dimNum " << inTensorDimNum;
            ATB_SPEED_LOG_ERROR(
                this->opName_ << " normalized max dimNum " << normalizedMaxDimNum
                              << " > inTensor dimNum " << inTensorDimNum;
            );
            throw std::runtime_error(ss.str());
        }
        int64_t normalizedShapeValue[param_.normAxes];
        for (int i = 0; i < param_.normAxes; ++i) {
            normalizedShapeValue[i] = aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dims[
                beginNormAxis + i
            ];
        }
        aclIntArray *normalizedShape = aclCreateIntArray(normalizedShapeValue, param_.normAxes);
        int ret = aclnnLayerNormWithImplModeGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,                                // input
            normalizedShape,                                                            // normalizedShape
            aclnnVariantPack.aclInTensors.at(1)->tensor,                                // weight
            param_.hasBias ? aclnnVariantPack.aclInTensors.at(2)->tensor : nullptr,     // bias
            param_.layerNormEps,                                                        // eps
            aclnnVariantPack.aclOutTensors.at(0)->tensor,                               // out
            nullptr,                                                                    // meanOut
            nullptr,                                                                    // rstdOut
            static_cast<int32_t>(param_.layerNormImplMode),                             // implMode
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        ATB_SPEED_LOG_DEBUG(
            opName_ << " SetAclNNWorkspaceExecutor end"
                    << ", ret: " << ret
                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor
        );
        return ret;
    }

    int LayerNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnLayerNormWithImplMode(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end" << ", ret: " << ret);
        return ret;
    }

}  // namespace atb_speed::common
