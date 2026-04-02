/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef ATB_SPEED_PLUGIN_ACLNN_SPARSE_FLASH_ATTENTION_H
#define ATB_SPEED_PLUGIN_ACLNN_SPARSE_FLASH_ATTENTION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

struct SparseFlashAttentionParam {
    std::string queryLayout = "BSND";
    std::string kvLayout = "BSND";
    double scaleValue = 1.0;
    int64_t sparseBlockSize = 2048;
    int64_t sparseMode = 3;
    bool hasBlockTable = false;
};

class SparseFlashAttentionOperation : public AclNNOperation {
public:
    explicit SparseFlashAttentionOperation(const std::string &name, SparseFlashAttentionParam param);
    ~SparseFlashAttentionOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                           atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    SparseFlashAttentionParam param_;
};
} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_SPARSE_FLASH_ATTENTION_H
