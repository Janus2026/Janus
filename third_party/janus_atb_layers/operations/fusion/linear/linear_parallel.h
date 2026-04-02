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

#ifndef ASCEND_SPEED_INFERENCE_COMMON_LINEAR_PARALLEL_H
#define ASCEND_SPEED_INFERENCE_COMMON_LINEAR_PARALLEL_H

#include <atb/atb_infer.h>
#include "acl/acl.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/parallel_info.h"
#include "operations/fusion/linear/linear.h"

namespace atb_speed {
namespace common {

/// Types of tensor parallelism, categorized based on how weight matrix is split across devices.
enum LinearParallelType : uint32_t {
    /// No parallelism is applied to the weight matrix
    UNDEFINED = 0,
    /// The weight matrix is split along its rows
    ROW_PARALLEL,
    /// The weight matrix is split along its columns
    COLUMN_PARALLEL,
    /// The weight matrix is split along its rows (0 dim)
    REDUCE_SCATTER,
};

/// Parameters for the linear parallel module
struct LinearParallelParam {
    /// Parameters for the linear module
    atb_speed::common::FusionLinearParam fusionLinearParam;
    /// Types of tensor parallelism. Refer to the `LinearParallelType`
    /// in the `operations/fusion/linear/linear_parallel.h`.
    int parallelType = UNDEFINED;
    /// A flag indicating whether to add bias after the all-reduce operation.
    bool biasAfterSync = false;
    /// A flag that indicates whether the input includes padding.
    /// It is applicable only when the `parallelType` is set to `ROW_PARALLEL`.
    bool unpadInputs = false;
    /// A flag that indicates whether low-latency computation over communication is enabled
    bool supportLcoc = false;
    bool enableMC2 = false;
    /// A flag indicating whether a mask is used before apply lora adapter.
    bool useImMask = false;
    /// Details about tensor parallelism
    bool isArgmaxlogits = false;
    /// A flag indicating whether argmax every card logits.
    int worldSize = 1;
    /// A flag indicating the prefill and decode phases
    bool isPrefill = false;
    /// The shape for inner tp slice
    int innerTpShape = 0;
    TensorParallelInfo tensorParallelInfo;
    atb_speed::common::ParallelInfo innerTensorParallelInfo;
};

/// This function is the main entrance for all types of linear parallel modules.
/// It will call different operations based on the `parallelType`.
///
/// \param param Parameters for the linear parallel module
/// \param operation the address of a pointer to a default operation
/// \return A flag that indicates whether operation has been successfully created.
///
/// Operations's inputs and outpus follow the same specifications as the inputs of the linear module.
/// See `operations/fusion/linear/linear.h` for more details.
///
/// In addtion, this operation supports the following optional inputs. They are required when
/// `param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL` and
/// `param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT`.
/// Name                   | Dtype    | Shape |
/// -----------------------|----------|-------|
/// in_reduce_quant_scale  | float16  | [k]   |
/// in_reduce_quant_offset | int8     | [k]   |
/// in_gather_quant_scale  | float16  | [k]   |
/// in_gather_quant_offset | float16  | [1]   |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_PLACEHOLDER,
///     OUT,
/// };
///
/// atb::Node linearParallelNode;
/// atb_speed::common::LinearParallelParam linearParallelParam;
/// // Modify linearParallelParam's attribute if needed.
/// linearParallelParam.parallelType = atb_speed::common::ROW_PARALLEL;
/// linearParallelParam.tensorParallelInfo,worldSize = 4;
/// LinearParallel(linearParallelParam, &linearParallelNode.operation);
/// linearParallelNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER};
/// linearParallelNode.outTensorIds = {OUT};
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearParallelNode);
/// \endcode
atb::Status LinearParallel(const LinearParallelParam &param, atb::Operation **operation);

/// This function adds communication operations to the graph.
/// It will call different operations based on the `parallelType`.
///
/// \param param Parameters for the `LinearParallelParam` module
/// \param opGraph A reference to the graph
/// \param tensorMap Defines all the required tensors for the current graph, with the key representing
/// the input tensor name and the value corresponding to the tensor index.
/// \return A flag indicating whether the operation has been successfully created.
///
/// This function will use the following tensors if `parallelType` equals to `COLUMN_PARALLEL`:
/// Key in `tensorMap`      | Requirements | Dtype            | Shape | Description |
/// ------------------------|--------------|------------------|-------|----------|
/// intermediate_linear_out | Required     | float16/bfloat16 | [m, n] or [m, k, n] | Hidden states |
/// intermediate_sync_out   | ^            | float16/bfloat16 | [group_size, m, n] or [group_size, m, k, n] | Hidden
/// states of all communication groups | out                     | ^            | ^                | [m, n * group_size]
/// or [m, k, n * group_size] | Hidden states of all communication groups after tensor reorder |
int64_t AddCommunicationOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
                           std::map<std::string, uint32_t> &tensorMap);
} // namespace common
} // namespace atb_speed

#endif
