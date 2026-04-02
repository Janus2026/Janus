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
#ifndef ATB_SPEED_MODELS_QWEN_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_MOE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen {
class MoeDecoderLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    bool isPack = true;
    int quantType = 0;
    int maskStartIdx = 0;
    int layerId = 0;
    int rank = 0;
    int worldSize = 1;
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = true;
    bool hasMoe = true;
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    // Use Aclnn RmsNorm instead of ATB RmsNorm.
    bool enableAclnnRmsNorm = false;
    bool enableAclGraphPagedAttention = false;
    bool enableAclnnAddRmsNorm = false;
    bool enableAclnnExternelAddRmsNorm = false;
    bool enableFusedReducesumDiv = false;
};

atb::Status MoeDecoderLayer(const MoeDecoderLayerParam &param, atb::Operation **operation);

class MoeDecoderLayer : public HostTensorBinder {
public:
    MoeDecoderLayer();
    ~MoeDecoderLayer() override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
}  // namespace qwen
}  // namespace atb_speed
#endif
