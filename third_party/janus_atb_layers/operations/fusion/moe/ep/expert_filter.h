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
#ifndef ATB_SPEED_MODELS_EXPERT_FILTER_OPERATION_H
#define ATB_SPEED_MODELS_EXPERT_FILTER_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
struct ExpertFilterParam {
    bool shiftedTopK = true;
    bool isBF16 = false;
    bool enableGatingDp = false;
    long unsigned int numOfExperts = 8;
    std::vector<int32_t> deviceExpert = {0, 1, 2, 3, 4, 5, 6, 7};
};

atb::Status CreateExpertFilterOperation(const ExpertFilterParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif