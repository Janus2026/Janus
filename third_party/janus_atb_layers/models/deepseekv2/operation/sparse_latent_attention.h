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

#ifndef ATB_SPEED_MODELS_COMMON_SPARSE_LATENT_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_SPARSE_LATENT_ATTENTION_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "models/deepseekv2/operation/latent_attention.h"

namespace atb_speed {
namespace deepseekV2 {

// template <typename NormParamType>
// std::map<std::string, uint32_t> ConstructTensorMap(
//     uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

template <typename NormParamType>
atb::Status SparseAttention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace deepseekV2
} // namespace atb_speed
#endif