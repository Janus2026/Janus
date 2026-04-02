/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_PLUGIN_ACLNN_MINIMUM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MINIMUM_OPERATION_H

#include "aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

class MinimumOperation : public AclNNOperation {
public:
    explicit MinimumOperation(const std::string &name);
    ~MinimumOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                           atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, int tensorIdx) const;
};
} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_MINIMUM_OPERATION_H
