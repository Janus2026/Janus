/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_ALL2ALL_MATMUL_OPERATION_H
#define ATB_SPEED_MODELS_ALL2ALL_MATMUL_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
constexpr int DEFAULT_TOPK_SCALE = -1;

struct All2AllMatmulParam {
    int32_t topk = 2;
    uint32_t numOfExperts = 8;
    uint32_t numOfDeviceExperts = 8;
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    int32_t scaledTopk = -1;
    int moeEpRank = 0;
    int moeEpSize = 1;
    std::string lcclMoeEpDomain = "";
    HcclComm lcclMoeEpHcclComm = nullptr;
};

atb::Status CreateAll2AllMatmulOperation(const All2AllMatmulParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif