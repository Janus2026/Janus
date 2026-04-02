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
#ifndef ATB_SPEED_MODELS_GLM_DECODER_LAYER_H
#define ATB_SPEED_MODELS_GLM_DECODER_LAYER_H

#include "atb/atb_infer.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/base/param/layer_param.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace glm {

class MoeLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    /// A flag indicating whether this layer is last layer.
    bool useMLA = false;
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    
    // h3p
    bool enableQkvdownDp = false;
    bool enableSharedExpertDp = false;
    bool enableGatingDp = false;
    bool enableSharedExpertOverlap = false;
    
    bool enableInfNan = true;
    bool enableOutLcocTp = false;
    bool enablePreprocessLcocTp = false;
    bool enableLcocAll2All = false;
    bool mixSharedRouting = false;
    bool enableFusedMLA = false;
    bool enableIndexGmm = false;

    bool enableLoadBalance = false;
    bool maskfree = true;

    // MLAtranslated
    int qLoraRank = 768;
    int kvLoraRank = 512;
    int headNum = 32;
    int actual_headNum = 0;
    int qkNopeHeadDim = 192;
    int qkRopeHeadDim = 64;
    float softmaxScale = 0;
    bool enableMlaPreprocess = false;
    bool enableCustomizeMla = false;
    bool isNzCache = false;
    bool enablePrefixCache = false;
    bool enableExtraOprojTp = false;
    
    // lightindexertranslated  translated
    int index_head_dim = 0; // 128
    int index_n_heads = 0; // 64
    int index_topk = 0; // 2048
};


template <typename NormType>
class MoeDecoderLayer : public atb_speed::moe::MoeDecoderLayer<NormType> {
public:
    explicit MoeDecoderLayer(const MoeLayerParam &param);
    ~MoeDecoderLayer() override {};

    atb::Status BuildGraph(atb::Operation **operation) override;

protected:
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam) override;

    void ConstructInTensorMap() override;
    
    atb::Status AddFusionAttention(bool is_auxiliary, uint64_t stream_id) override; 
    atb::Status CreateFusionAttentionOperation(atb::Operation **op);

    atb_speed::glm::MoeLayerParam param;

};

}  // namespace glm
}  // namespace atb_speed
#endif  // ATB_SPEED_MODELS_GLM_DECODER_LAYER_H
