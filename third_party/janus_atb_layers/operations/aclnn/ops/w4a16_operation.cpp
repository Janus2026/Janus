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

#include <iostream>
#include <sstream>
#include <memory>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/utils/utils.h"
#include "w4a16_operation.h"

namespace atb_speed {
namespace common {

W4A16Operation::W4A16Operation(
    const std::string &name,
    AclNNWeightQuantBatchMatmulParam param) : QuantBatchMatmulOperation(name, param), param_(param) {}

atb::Tensor W4A16Operation::PreprocessATBInTensor(atb::Tensor atbTensor, int index)
{
    atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(atbTensor);
    if (index == 1) {  // 1: weight
        squeezedAtbTensor.desc.dtype = ACL_INT4;
        squeezedAtbTensor.desc.shape.dims[DIM1] = CheckIntMulOverFlow(
            squeezedAtbTensor.desc.shape.dims[DIM1], 2);  // 2: translatedshape * 2
    }
    return squeezedAtbTensor;
}

} // namespace common
} // namespace atb_speed