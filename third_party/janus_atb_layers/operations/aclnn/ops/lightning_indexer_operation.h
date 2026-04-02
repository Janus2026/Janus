/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_PLUGIN_ACLNN_LIGHTING_INDEXER_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_LIGHTING_INDEXER_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

struct LightningIndexerParam {
    std::string queryLayout = "TND";
    std::string keyLayout = "PA_BSND";
    int64_t selectedCount = 2048;
    int64_t sparseMode = 3;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    // bool returnValues = false;
};

class LightningIndexerOperation : public AclNNOperation {
public:
    explicit LightningIndexerOperation(const std::string &name, LightningIndexerParam param);
    ~LightningIndexerOperation() override;
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
    LightningIndexerParam param_;
};
} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_LIGHTING_INDEXER_OPERATION_H
