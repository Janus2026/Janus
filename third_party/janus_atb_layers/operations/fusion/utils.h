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

#ifndef ATB_SPEED_MODELS_COMMON_UITLS_H
#define ATB_SPEED_MODELS_COMMON_UITLS_H

#include <vector>
#include <unordered_set>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/base/event_manager.h"


namespace atb_speed {
namespace common {

/// The pack and quantization type of linear operations.
/// Q, k and v linear may be packed. Gate and up linear may be packed.
///
/// Each value except `PACK_QUANT_UNDEFINED` represents a combination of pack type and quantization type.
/// Explaination of each key word:
/// - `PACK_QUANT_UNDEFIEND`: undefined pack and quantization type.
/// - `ALL`: all linear in the pack are using same quantization, weights will be combined to accelerate computation.
/// - `MIX`: linears in the pack are mixture of quantization and float. Computation will be performed separately.
/// - `W8A8`: weights and activation values are quantized to int8.
/// - `W4A8`: weights are quantized to int4 and activation values are quantized to int8.
/// - `W8A16`: weights are quantized to int8 and activation values are holded in float16/bfloat16.
/// - `W4A16`: weights are quantized to int4 and activation values are holded in float16/bfloat16.
/// - `ANTI`: quantization with anti-outlier.
/// - `SC`: quantization with model sparsing and compression. Exclusively supported by Atlas 300I Duo.
/// - `DYNAMIC`: using per-token quantization.
enum PackQuantType : unsigned int {
    PACK_QUANT_UNDEFINED = 0,
    ALL_FP = 1,
    ALL_W8A8 = 2,
    ALL_W8A8_ANTI = 3,
    MIX_W8A8 = 4,
    MIX_W8A8_ANTI = 5,
    ALL_W8A16 = 6,
    ALL_W8A8SC = 7,
    MIX_W8A8SC = 8,
    ALL_W8A8SC_ANTI = 9,
    MIX_W8A8SC_ANTI = 10,
    ALL_W4A16 = 11,
    ALL_W8A16_ANTI = 12,
    ALL_W4A16_ANTI = 13,
    MIX_W4A16 = 14,
    MIX_W4A16_ANTI = 15,
    MIX_W8A16 = 16,
    MIX_W8A16_ANTI = 17,
    ALL_W8A8_DYNAMIC = 18,
    ALL_W8A8_DYNAMIC_ANTI = 19,
    MIX_W8A8_DYNAMIC = 20,
    MIX_W8A8_DYNAMIC_ANTI = 21,
    ALL_W4A8 = 22,
    MIX_W4A8 = 23,
    ALL_W4A8_ANTI = 24,
    MIX_W4A8_ANTI = 25
};

/// An listing of operations backend.
enum OpBackend: unsigned int {
    /// Ascend Transformer Boost backend.
    ATB = 0,
    /// Ascend Computing Language Neural Network backend.
    ACLNN = 1,
};

/// An enumeration of quantization types for the linear operations.
enum LinearQuantType : unsigned int {
    /// No quantization.
    NO_QUANT = 0,
    /// Weights are quantized to int8 and activation values are quantized to int8.
    /// Quantization is performed in normalization operation and dequantization is performed in linear operation.
    LINEAR_W8A8_DEQUANT,
    /// Weights are quantized to int8 and activation values are quantized to int8.
    /// Quantization and dequantization are both performed in linear operation.
    LINEAR_W8A8_QUANT,
    /// Weights are quantized to int4.
    /// Quantization and dequantization are both performed in linear operation.
    W4A16,
    /// Weights are quantized to int8.
    /// Quantization and dequantization are both performed in linear operation.
    W8A16,
    /// Weights are quantized to int8 and activation values are quantized to int8, using sparse compression.
    /// Quantization is performed in normalization operation and dequantization is performed in linear operation.
    LINEAR_W8A8_SC_DEQUANT,
    /// Weights are quantized to int8 and activation values are quantized to int8, using sparse compression.
    /// Quantization and dequantization are both performed in linear operation.
    LINEAR_W8A8_SC_QUANT,
    /// Weights are quantized to int8 and activation values are quantized to int8, using per-token quantization.
    /// Quantization is performed in normalization operation and dequantization is performed in linear operation.
    LINEAR_W8A8_DYNAMIC_DEQUANT,
    /// Weights are quantized to int8 and activation values are quantized to int8, using per-token quantization..
    /// Quantization and dequantization are both performed in linear operation.
    LINEAR_W8A8_DYNAMIC_QUANT,
    LINEAR_W4A8_DYNAMIC_DEQUANT,
    LINEAR_W4A8_DYNAMIC_QUANT
};

/// An enum of linear dtype.
enum LinearType : int {
    /// Invalid type.
    INVALID = -1,
    /// Float type.
    FP = 0,
    /// Integer type.
    INT = 1,
};

enum LinearDesc : int {
    INVALID_DESC = -1,
    FLOAT16_DESC = 0,
    BFLOAT16_DESC = 1,
    W4A16_DESC = 2,
    W8A16_DESC = 3,
    W8A8_PER_TENSOR_DESC = 4,
    W8A8S_DESC = 5,
    W8A8SC_DESC = 6,
    W8A8_DYNAMIC_DESC = 7,
    W8A8_PDMIX_DESC = 8,
    W4A8_DESC = 9
};

/// Transpose type of B matrix in matmul operation.
enum TransposeType : int {
    /// Invalid type.
    TRANSPOSE_INVALID = -1,
    /// Do not transpose B matrix in matmul operation.
    NOT_TRANSPOSE = 0,
    /// Do transpose B matrix in matmul operation.
    TRANSPOSE = 1,
};

/// Details about tensor parallelism
///
/// Parameters will be directly passed to `AllReduceOperation`, `AllGatherOperation`
/// or `LinearParallelOperation` defined in `atb/atb_infer.h`.
struct TensorParallelInfo {
    /// Rank of the current process
    int rank = 0;
    /// Number of processes participating in the job
    int worldSize = 1;
    /// Communication backend. Options: `hccl`, `lccl`
    std::string backend = "hccl";
    /// Path of the cluster information config file. Use for single-node or multi-node communcation.
    std::string rankTableFile = "";
    HcclComm hcommInfo = nullptr;
    /// A communication device group is identified by a communication domain name.
    std::string commDomain = "";
    /// Quant type
    atb::infer::AllReduceParam::QuantType quantType = atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED;
    /// The data type of the output tensor
    aclDataType outDataType = ACL_DT_UNDEFINED;
};

extern const std::string CMO_COMPUTE;
extern const std::string CMO_OPROJ;
extern const std::string CMO_MLAPO;
extern const std::string CV_START;
extern const std::string VECTOR_CONTROL;
extern const std::string CUBE_CONTROL;
extern const std::string COMPUTE_EVENT;
extern const std::string COMM_EVENT;
extern const std::string END_EVENT;
extern const std::string CC_START;
extern const std::string COMM_CONTROL;
extern const std::string COMP_CONTROL;

enum DapRole : uint32_t {
    UNDEFINED_ROLE = 0,
    PRECEDER = 1,
    SUCCESSOR = 2,
};

class DapManager {
public:
    void SetRole(DapRole role);
    DapRole GetRole() const;
    std::string GetSuccessorSuffix() const;
    uint32_t GetStreamId() const;

private:
    DapRole currentRole = DapRole::UNDEFINED_ROLE;
};

class CommOpCounter {
public:
    int32_t Increment();
    int32_t GetCount();
    void Reset();

private:
    std::map<DapRole, int32_t> count = {};
};

atb::Status AddDapEventsBeforeComm(atb::GraphParam &opGraph);

atb::Status AddDapEventsAfterComm(atb::GraphParam &opGraph);

/// Assgin indices to tensors based on the provided tensorCandidates and targetKey.
/// Indices start from the provided tensorIndex.
///
/// \param tensorCanditates A map where each tensor key maps to a list of tensor name.
/// \param targetKey The tensor key to identify which tensor name list to be assigned.
/// \param tensorIdx The initial index to assgin.
/// \param tensorMap A map store the tensor name to index mapping.
void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, uint32_t &tensorIdx, std::map<std::string, uint32_t> &tensorMap);

/// Assgin indices to tensors based on the provided tensorCandidates and targetKey.
/// Indices start from the size of the existing tensorMap.
///
/// \param tensorCanditates A map where each tensor key maps to a list of tensor name.
/// \param targetKey The tensor key to identify which tensor list to be assigned.
/// \param tensorMap A map store the tensor name to index mapping.
void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, std::map<std::string, uint32_t> &tensorMap);

/// Add tensorCandidates.at(targetKey) to tensorList.
///
/// \tparam T The type of elements in the tensor list. This can be any container type that supports push_back.
/// \param tensorCandidates A map where each tensor key maps to a list of tensor.
/// \param targetKey The tensor key to identify which tensor list to be assigned.
/// \param tensorList A list of type T where to add the tensor.
template <typename T>
void AddTensorToList(
    const std::map<std::string, T> &tensorCandidates,
    std::string targetKey, T &tensorList);

/// Return a map of tensor name to tensor index.
///
/// Assume length of in/out/intermediateTensorList is n1/n2/n3, the returned map is
/// \code
/// {
///     {inTensorList[0], 0}, {inTensorList[1]: 1},..., {inTensorList[n1-1], n1-1},
///     {outTensorList[0], n1}, {outTensorList[1]: n1+1},..., {outTensorList[n2-1], n1+n2-1},
///     {intermediateTensorList[0], n1+n2},..., {intermediateTensorList[n3-1], n1+n2+n3-1}
/// };
/// \endcode
///
/// \param inTensorList A list of input tensors of an operation.
/// \param outTensorList A list of output tensors of an operation.
/// \param intermediateTensorList A list of intermediate tensors of an operation.
/// \return A map of tensor name to tensor index.
std::map<std::string, uint32_t> GetTensorMap(
    std::vector<std::string> &inTensorList, std::vector<std::string> &outTensorList,
    std::vector<std::string> &intermediateTensorList);

/// Retrieve the tensor index using the `tensorName` from the `tensorMap`.
///
/// \param tensorMap A map of tensor name to tensor index.
/// \param tensorName The name of the tensor.
/// \return The index of the tensor.
uint32_t GetTensorIdx(const std::map<std::string, uint32_t> &tensorMap, std::string tensorName);

/// Return a list of tensor indices from the `tensorMap` referenced by `tensorNames`.
///
/// \param tensorMap A map of tensor name to tensor index.
/// \param tensorNames A list of tensor names.
/// \return A list of tensor indices.
atb::SVector<uint32_t> GetTensorIdxList(const std::map<std::string, uint32_t> &tensorMap,
    std::vector<std::string>tensorNames);

/// Verify if `packQuantType` supports quantization with anti-outlier.
///
/// \param packQuantType The pack and quantization type of linear operations.
/// Refer to `atb_speed::common::PackQuantType` in the `operations/fusion/utils.h` for more details.
/// \return True if `packQuantType` supports quantization with anti-outlier.
bool CheckAntiOutlier(const int &packQuantType);

/// Check whether linear weights are packed.
/// \param packQuantType The pack and quantization type of linear operations.
/// Refer to `atb_speed::common::PackQuantType` in the `operations/fusion/utils.h` for more details.
/// \param linearDescs weight description of linear module
/// \param linearIndex A list of index of the target linear module
/// \return True if linear weights are packed.
bool CheckPack(const int &packQuantType = PackQuantType::PACK_QUANT_UNDEFINED,
    const std::vector<int> &linearDescs = {},
    const std::vector<int> &linearIndex = {});

/// Valide the size of `vector`. It should not be smaller than `threshold`.
/// \param vector The vector to be checked.
/// \param threshold The threshold of the size of `vector`.
/// \return A flag indicating whether the size of `vector` is valid.
atb::Status CheckParamVectorSize(const std::vector<int> &vector, size_t threshold);

atb::Status CreateRecordWithoutNodeId(atb::GraphParam &opGraph,
                                      atb_speed::EventAction eventAction, const std::string &cvKey);

atb::Status CreateWaitWithoutNodeId(atb::GraphParam &opGraph,
                                    atb_speed::EventAction eventAction, const std::string &cvKey);

/// Convert quantType to packType. e.g. `"w8a8"` -> `ALL_W8A8`.
/// \param quantType The quantization type. Valid values are `float`/`w8a8`/`w8a8s`/`w8a8sc`/`w8a8_dynamic`/`w8a16`/
///     `w4a16`, other input values will return `PACK_QUANT_UNDEFINED`.
/// \return The corresponding pack type.
PackQuantType ConvertQuantTypeToPackType(std::string quantType);
} // namespace common
} // namespace atb_speed
#endif