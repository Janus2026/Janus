/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef ATB_SPEED_PLUGIN_ACLNN_MLA_PREPROCESS_V2_H
#define ATB_SPEED_PLUGIN_ACLNN_MLA_PREPROCESS_V2_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

struct Mlapreprocessv2OperationParam {
    int64_t wdqDim = 0;
    int64_t qRopeDim = 0;
    int64_t kRopeDim = 0;
    double epsilon = 1e-5;
    int64_t qRotaryCoeff = 2;
    int64_t kRotaryCoeff =2;
    bool transeposeWdq = true;
    bool transeposeWuq = true;
    bool transeposeWuk = true;
    int64_t cacheMode = 1;
    int64_t quantMode = 0;
    bool doRmsNorm = true;
    int64_t wdkvSplitCount = 1;
    bool qDownOutFlag = true;
};

class Mlapreprocessv2Operation : public AclNNOperation {
public:
    explicit Mlapreprocessv2Operation(const std::string &name, Mlapreprocessv2OperationParam param);
    ~Mlapreprocessv2Operation() override;
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
    Mlapreprocessv2OperationParam param_;
};
} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_MLA_PREPROCESS_V2_H

