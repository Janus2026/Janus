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
#ifndef ATB_SPEED_MODELS_ATTENTION_H
#define ATB_SPEED_MODELS_ATTENTION_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"

#include "operations/fusion/parallel_layer_v2.h"

namespace atb_speed {
namespace common {

struct AttentionParam {
    float normEps = 0; /// The epsilon used by the layer normalization layers.
    int layerId = 0;  /// The current layer Id.
    int numHiddenLayers = 0;  /// The number of hidden layers.
    int numAttentionHeads = 8;  /// The number of attention heads.
    int numKeyValueHeads = 1;  /// The number of key/value heads.
    int hiddenSize = 0;  /// The size of hidden layers.
    int seqLength = 1;  // The input sequence length.
    bool isPrefill = false;  // A flag indicating whether the  prefill phase.
    bool isGQA = false;  /// A flag indicating whether attention type is GQA.
    bool isQuant = false;  /// A flag indicating whether quantified or not.
    bool isHasQKVBias = false; /// A flag indicating whether qkv has bias or not.
    bool useQKNorm = false;
    int hiddenSizePerAttentionHead = 0;
    QuantParam quantParam;  /// The parm of quant , it is struct.
};

/// This function helps us build an attention based on 310B, It is used only on 310B.
atb::Status AttentionEdge(const AttentionParam &param, atb::Operation **operation);

}  // namespace common
}  // namespace atb_speed
#endif