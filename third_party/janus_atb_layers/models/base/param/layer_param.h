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
#ifndef ATB_SPEED_BASE_LAYER_PARAM_H
#define ATB_SPEED_BASE_LAYER_PARAM_H
#include <vector>
#include <nlohmann/json.hpp>
#include "models/base/param/param.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace base {

/// Parameters for the base layer, inherited from the `Param` class.
///
/// In addition to the parameters defined in the Param class,
/// this class introduces additional parameters specific to the base `DecoderLayer` class.
class LayerParam : public Param {
public:
    LayerParam() {};
    ~LayerParam() override {};
    void PrintParam() override;
    void CheckParam() override;

    /// The layer index, starting from 0
    int layerId = 0;
    /// Number of hidden layers
    int numHiddenLayers = 0;
    /// Information for tensor parallelism
    atb_speed::common::TensorParallelInfo tensorParallelInfo;
    /// Indicates the pack type and the quantization type of the qkv linear and gate up linear.
    std::vector<int> packQuantType = {
        common::PackQuantType::PACK_QUANT_UNDEFINED, common::PackQuantType::PACK_QUANT_UNDEFINED
    };
    /// Specifies the quantization type for the following linear module:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    std::vector<int> linearQuantType = {
        common::LinearType::INVALID, common::LinearType::INVALID, common::LinearType::INVALID,
        common::LinearType::INVALID, common::LinearType::INVALID, common::LinearType::INVALID,
        common::LinearType::INVALID
    };
    /// Defines the transpose type of the second matrix in the matmul operation for the following linear module:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    std::vector<int> linearTransposeType = {};
    /// Specifies whether the following linear module has bias:
    /// qkv linear, dense linear, gateup linear and down linear.
    std::vector<bool> linearHasBias = {false, false, false, false};
    /// Specifies the weight description of the following linear module:
    /// qkv linear, dense linear, gateup linear and down linear.
    std::vector<int> linearDescs = {
        common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC,
        common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC,
        common::LinearDesc::INVALID_DESC
    };
    /// Specifies whether the input norm and post attention norm enable antioutlier
    std::vector<bool> isAntiOutlier = {false, false};
    /// A flag indicating whether currentlayer is compressed
    bool isomnicompressed = false;
};
} // namespace base
} // namespace atb_speed
#endif