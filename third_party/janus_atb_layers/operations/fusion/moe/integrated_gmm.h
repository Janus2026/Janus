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
#ifndef ATB_SPEED_MODELS_INTEGRATED_GMM_OPERATION_H
#define ATB_SPEED_MODELS_INTEGRATED_GMM_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
enum IntegratedGmmIdx : int {
    ROUTER_IDX = 0,
    MOE_MLP_GATE_IDX,
    MOE_MLP_UP_IDX,
    MOE_MLP_DOWN_IDX
};
struct IntegratedGmmParam {
    /// The quantization tpe of the linear transformation of this sub-graph
    std::vector<int> moeLinearQuantType = {};
    /// A flag indicating whether there is bias to the linear transformation of this sub-graph
    bool hasBias = false;
    /// A flag indicating whether the linear transformation is the `UP` or the `DOWN` stage of FFN
    bool isUp = true;
    /// The data type of the output of the linear transformation
    aclDataType outDataType = ACL_FLOAT16;
    /// A flag indicating whether the second matrix of the matrix multiplication needs to be transposed
    bool transposeB = false;
    /// A flag indicating whether the second matrix of the matrix multiplication needs to be transposed
    bool downTransposeB = false;
    /// The quantization type of the packed weights
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    /// The quantization type used to facilitate the calculation of the quantization type of the linear operation
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach
    int quantGroupSize = 0;
    /// A flag indicating whether or not to skip the quantization step
    bool skipQuant = false;
    /// A flag indicating whether the model use Moe parallel
    bool enableMoeParallel = false;
    /// A flag indicating whether the model use cube and vector parallel
    bool enableCVOverlap = false;
    /// A flag indicating whether or not to use integrated GMM+Swiglu+quant operators.
    bool enableGMMSwigluQuant = false;
    /// A flag indicating whether or not to use fused atb GMM+Swiglu+quant operators instead of aclnn.
    bool enableAtlasGMMFused = false;
    /// A flag indicating whether to use index GMM
    bool enableIndexGmm = false;
    bool enableInitRoutingV3 = false;
};

/// This function creates a sub-graph that performs grouped-matmul.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateIntegratedGmmOperation(const IntegratedGmmParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif
