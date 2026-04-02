/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_COMBINE_OPERATION_V2_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_COMBINE_OPERATION_V2_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeDistributeCombineV2Param {
    int32_t epRankId = 0;
    int32_t epRankSize = 1;
    int32_t tpRankId = 0;
    int32_t tpRankSize = 1;
    int32_t expertSharedType = 0;
    int32_t maxDecodeDpTokenSize = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 1;
    int64_t localMoeExpertNum = 1;
    int64_t topk = 8;
    int64_t globalBS = 0; // tilingtranslatedBS*world_size
    std::string tpCommName;
    std::string epCommName;
    std::string commAlg;
    std::string rankTableFile = "";
    HcclComm hcclComm = nullptr;
    int64_t commQuantMode = 0;
};

class MoeDistributeCombineV2Operation : public AclNNOperation {
public:
    explicit MoeDistributeCombineV2Operation(const std::string &name, MoeDistributeCombineV2Param param);
    ~MoeDistributeCombineV2Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    int32_t GetGlobalBS(const atb::TensorDesc &inTensorDesc) const;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeDistributeCombineV2Param param_;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_COMBINE_OPERATION_H