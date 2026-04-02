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
#include <cstring>
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include "atb/svector.h"
#include "hccl/hccl.h"

#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/external_comm_manager.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/hccl_runner.h"
#include "operation_factory.h"
#include "operations/aclnn/ops/indexput_operation.h"
#include "operations/aclnn/ops/index_select_operation.h"
#include "operations/aclnn/ops/w8a16_operation.h"
#include "operations/aclnn/ops/w4a16_operation.h"
#include "operations/aclnn/ops/w8a8_operation.h"
#include "operations/aclnn/ops/w4a8_operation.h"
#include "operations/aclnn/ops/std_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"
#include "operations/aclnn/ops/vector_norm_operation.h"
#include "operations/aclnn/ops/argsort_operation.h"
#include "operations/aclnn/ops/moe_topk_softmax_operation.h"
#include "operations/aclnn/ops/moe_init_routing_operation.h"
#include "operations/aclnn/ops/moe_compute_expert_tokens_operation.h"
#include "operations/aclnn/ops/moetoken_unpermute_operation.h"
#include "operations/aclnn/ops/matmul_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/aclnn/ops/prompt_flash_attention_operation.h"
#include "operations/aclrt/ops/aclrt_cmo_async.h"
#include "operations/aclnn/ops/cast_operation.h"
#include "operations/aclnn/ops/scatter_operation.h"
#include "operations/aclnn/ops/dequant_rope_quant_kvcache_operation.h"
#include "operations/aclnn/ops/dequant_swiglu_quant_operation.h"
#include "operations/aclnn/ops/inplace_nan_to_num_operation.h"

namespace atb {
namespace infer {
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::LayerNormParam::LayerNormType, {
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_UNDEFINED, "LAYER_NORM_UNDEFINED"},
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM, "LAYER_NORM_NORM"},
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_PRENORM, "LAYER_NORM_PRENORM"},
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_POSTNORM, "LAYER_NORM_POSTNORM"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::RmsNormParam::RmsNormType, {
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_UNDEFINED, "RMS_NORM_UNDEFINED"},
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM, "RMS_NORM_NORM"},
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM, "RMS_NORM_PRENORM"},
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_POSTNORM, "RMS_NORM_POSTNORM"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::RmsNormParam::PrecisionMode, {
        {atb::infer::RmsNormParam::PrecisionMode::HIGH_PRECISION_MODE, "HIGH_PRECISION_MODE"},
        {atb::infer::RmsNormParam::PrecisionMode::HIGH_PERFORMANCE_MODE, "HIGH_PERFORMANCE_MODE"},
    })
    
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::RmsNormParam::ModelType, {
        {atb::infer::RmsNormParam::ModelType::LLAMA_MODEL, "LLAMA_MODEL"},
        {atb::infer::RmsNormParam::ModelType::GEMMA_MODEL, "GEMMA_MODEL"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::DynamicQuantType, {
        {atb::infer::DynamicQuantType::DYNAMIC_QUANT_UNDEFINED, "DYNAMIC_QUANT_UNDEFINED"},
        {atb::infer::DynamicQuantType::DYNAMIC_QUANT_SYMMETRIC, "DYNAMIC_QUANT_SYMMETRIC"},
        {atb::infer::DynamicQuantType::DYNAMIC_QUANT_ASYMMETRIC, "DYNAMIC_QUANT_ASYMMETRIC"},
    })
    
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::InputLayout, {
        {atb::infer::InputLayout::TYPE_BSND, "TYPE_BSND"},
        {atb::infer::InputLayout::TYPE_BNSD, "TYPE_BNSD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::QuantType, {
        {atb::infer::QuantType::QUANT_UNDEFINED, "QUANT_UNDEFINED"},
        {atb::infer::QuantType::QUANT_INT4, "QUANT_INT4"},
        {atb::infer::QuantType::QUANT_INT8, "QUANT_INT8"},
        {atb::infer::QuantType::QUANT_INT16, "QUANT_INT16"},
        {atb::infer::QuantType::QUANT_FLOAT8, "QUANT_FLOAT8"},
        {atb::infer::QuantType::QUANT_FLOAT16, "QUANT_FLOAT16"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ActivationType, {
        {atb::infer::ActivationType::ACTIVATION_UNDEFINED, "ACTIVATION_UNDEFINED"},
        {atb::infer::ActivationType::ACTIVATION_RELU, "ACTIVATION_RELU"},
        {atb::infer::ActivationType::ACTIVATION_GELU, "ACTIVATION_GELU"},
        {atb::infer::ActivationType::ACTIVATION_FAST_GELU, "ACTIVATION_FAST_GELU"},
        {atb::infer::ActivationType::ACTIVATION_SWISH, "ACTIVATION_SWISH"},
        {atb::infer::ActivationType::ACTIVATION_LOG, "ACTIVATION_LOG"},
        {atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD, "ACTIVATION_SWIGLU_FORWARD"},
        {atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD, "ACTIVATION_SWIGLU_BACKWARD"},
        {atb::infer::ActivationType::ACTIVATION_SIGMOID, "ACTIVATION_SIGMOID"},
        {atb::infer::ActivationType::ACTIVATION_MAX, "ACTIVATION_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ActivationParam::GeLUMode, {
        {atb::infer::ActivationParam::GeLUMode::TANH_MODE, "TANH_MODE"},
        {atb::infer::ActivationParam::GeLUMode::NONE_MODE, "NONE_MODE"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::CommMode, {
        {atb::infer::CommMode::COMM_UNDEFINED, "COMM_UNDEFINED"},
        {atb::infer::CommMode::COMM_MULTI_PROCESS, "COMM_MULTI_PROCESS"},
        {atb::infer::CommMode::COMM_MULTI_THREAD, "COMM_MULTI_THREAD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ElewiseParam::ElewiseType, {
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_UNDEFINED, "ELEWISE_UNDEFINED"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST, "ELEWISE_CAST"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS, "ELEWISE_MULS"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_COS, "ELEWISE_COS"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_SIN, "ELEWISE_SIN"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_NEG, "ELEWISE_NEG"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT, "ELEWISE_QUANT"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, "ELEWISE_LOGICAL_NOT"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD, "ELEWISE_ADD"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL, "ELEWISE_MUL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, "ELEWISE_REALDIV"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, "ELEWISE_LOGICAL_AND"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, "ELEWISE_LOGICAL_OR"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LESS, "ELEWISE_LESS"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, "ELEWISE_GREATER"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_SUB, "ELEWISE_SUB"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, "ELEWISE_EQUAL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL, "ELEWISE_QUANT_PER_CHANNEL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL, "ELEWISE_DEQUANT_PER_CHANNEL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT, "ELEWISE_DYNAMIC_QUANT"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_TANH, "ELEWISE_TANH"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::LinearParallelParam::ParallelType, {
        {atb::infer::LinearParallelParam::ParallelType::UNDEFINED, "UNDEFINED"},
        {atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE, "LINEAR_ALL_REDUCE"},
        {atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER, "LINEAR_REDUCE_SCATTER"},
        {atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR, "ALL_GATHER_LINEAR"},
        {atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR, "PURE_LINEAR"},
        {atb::infer::LinearParallelParam::ParallelType::MAX, "MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::LinearParallelParam::QuantType, {
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED, "QUANT_TYPE_UNDEFINED"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TENSOR, "QUANT_TYPE_PER_TENSOR"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL, "QUANT_TYPE_PER_CHANNEL"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_GROUP, "QUANT_TYPE_PER_GROUP"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX, "QUANT_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::AllReduceParam::QuantType, {
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED, "QUANT_TYPE_UNDEFINED"},
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_TENSOR, "QUANT_TYPE_PER_TENSOR"},
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL, "QUANT_TYPE_PER_CHANNEL"},
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_MAX, "QUANT_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::CalcType, {
        {atb::infer::SelfAttentionParam::CalcType::UNDEFINED, "UNDEFINED"},
        {atb::infer::SelfAttentionParam::CalcType::ENCODER, "ENCODER"},
        {atb::infer::SelfAttentionParam::CalcType::DECODER, "DECODER"},
        {atb::infer::SelfAttentionParam::CalcType::PA_ENCODER, "PA_ENCODER"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::KernelType, {
        {atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_DEFAULT, "KERNELTYPE_DEFAULT"},
        {atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_HIGH_PRECISION, "KERNELTYPE_HIGH_PRECISION"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::ClampType, {
        {atb::infer::SelfAttentionParam::ClampType::CLAMP_TYPE_UNDEFINED, "CLAMP_TYPE_UNDEFINED"},
        {atb::infer::SelfAttentionParam::ClampType::CLAMP_TYPE_MIN_MAX, "CLAMP_TYPE_MIN_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::MaskType, {
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED, "MASK_TYPE_UNDEFINED"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM, "MASK_TYPE_NORM"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI, "MASK_TYPE_ALIBI"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM_COMPRESS, "MASK_TYPE_NORM_COMPRESS"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS, "MASK_TYPE_ALIBI_COMPRESS"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS_SQRT, "MASK_TYPE_ALIBI_COMPRESS_SQRT"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::ScaleType, {
        {atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_TOR, "SCALE_TYPE_TOR"},
        {atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_LOGN, "SCALE_TYPE_LOGN"},
        {atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_MAX, "SCALE_TYPE_MAX"},
    })
    
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::KvCacheCfg, {
        {atb::infer::SelfAttentionParam::KvCacheCfg::K_CACHE_V_CACHE, "K_CACHE_V_CACHE"},
        {atb::infer::SelfAttentionParam::KvCacheCfg::K_BYPASS_V_BYPASS, "K_BYPASS_V_BYPASS"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::QuantType, {
        {atb::infer::SelfAttentionParam::QuantType::TYPE_QUANT_UNDEFINED, "TYPE_QUANT_UNDEFINED"},
        {atb::infer::SelfAttentionParam::QuantType::TYPE_DEQUANT_FUSION, "TYPE_DEQUANT_FUSION"},
        {atb::infer::SelfAttentionParam::QuantType::TYPE_QUANT_QKV_OFFLINE, "TYPE_QUANT_QKV_OFFLINE"},
        {atb::infer::SelfAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE, "TYPE_QUANT_QKV_ONLINE"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::MaskType, {
        {atb::infer::PagedAttentionParam::MaskType::UNDEFINED, "MASK_TYPE_UNDEFINED"},
        {atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM, "MASK_TYPE_NORM"},
        {atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI, "MASK_TYPE_ALIBI"},
        {atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC, "MASK_TYPE_SPEC"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::QuantType, {
        {atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_UNDEFINED, "TYPE_QUANT_UNDEFINED"},
        {atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION, "TYPE_DEQUANT_FUSION"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::CalcType, {
        {atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED, "CALC_TYPE_UNDEFINED"},
        {atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC, "CALC_TYPE_SPEC"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::CompressType, {
        {atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED, "COMPRESS_TYPE_UNDEFINED"},
        {atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD, "COMPRESS_TYPE_KVHEAD"},
        {atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE, "COMPRESS_TYPE_KVHEAD_ROPE"},
        {atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_MAX, "COMPRESS_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::ScaleType, {
        {atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_TOR, "SCALE_TYPE_TOR"},
        {atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN, "SCALE_TYPE_LOGN"},
        {atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_MAX, "SCALE_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ReshapeAndCacheParam::CompressType, {
        {atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED, "COMPRESS_TYPE_UNDEFINED"},
        {atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD, "COMPRESS_TYPE_KVHEAD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ReshapeAndCacheParam::KvCacheCfg, {
        {atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE, "K_CACHE_V_CACHE"},
        {atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS, "K_CACHE_V_BYPASS"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::TransdataParam::TransdataType, {
        {atb::infer::TransdataParam::TransdataType::UNDEFINED, "UNDEFINED"},
        {atb::infer::TransdataParam::TransdataType::FRACTAL_NZ_TO_ND, "FRACTAL_NZ_TO_ND"},
        {atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ, "ND_TO_FRACTAL_NZ"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ReduceParam::ReduceType, {
        {atb::infer::ReduceParam::ReduceType::REDUCE_UNDEFINED, "REDUCE_UNDEFINED"},
        {atb::infer::ReduceParam::ReduceType::REDUCE_MAX, "REDUCE_MAX"},
        {atb::infer::ReduceParam::ReduceType::REDUCE_MIN, "REDUCE_MIN"},
        {atb::infer::ReduceParam::ReduceType::REDUCE_SUM, "REDUCE_SUM"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::TopkToppSamplingParam::TopkToppSamplingType, {
        {atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SAMPLING_UNDEFINED, "SAMPLING_UNDEFINED"},
        {atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING, "SINGLE_TOPK_SAMPLING"},
        {atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING,
            "BATCH_TOPK_MULTINOMIAL_SAMPLING"},
        {atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING,
            "BATCH_TOPK_EXPONENTIAL_SAMPLING"},
        {atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SAMPLING_MAX, "SAMPLING_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::IndexAddParam::IndexType, {
        {atb::infer::IndexAddParam::IndexType::INDEX_UNDEFINED, "INDEX_UNDEFINED"},
        {atb::infer::IndexAddParam::IndexType::INDEX_ADD, "INDEX_ADD"},
        {atb::infer::IndexAddParam::IndexType::INDEX_ADD_VALID, "INDEX_ADD_VALID"},
    })
}  // namespace infer
}  // namespace atb

NLOHMANN_JSON_SERIALIZE_ENUM(aclDataType, {
    {ACL_DT_UNDEFINED, "ACL_DT_UNDEFINED"},
    {ACL_FLOAT, "ACL_FLOAT"},
    {ACL_FLOAT16, "ACL_FLOAT16"},
    {ACL_INT8, "ACL_INT8"},
    {ACL_INT32, "ACL_INT32"},
    {ACL_UINT8, "ACL_UINT8"},
    {ACL_INT16, "ACL_INT16"},
    {ACL_UINT16, "ACL_UINT16"},
    {ACL_UINT32, "ACL_UINT32"},
    {ACL_INT64, "ACL_INT64"},
    {ACL_UINT64, "ACL_UINT64"},
    {ACL_DOUBLE, "ACL_DOUBLE"},
    {ACL_BOOL, "ACL_BOOL"},
    {ACL_STRING, "ACL_STRING"},
    {ACL_COMPLEX64, "ACL_COMPLEX64"},
    {ACL_COMPLEX128, "ACL_COMPLEX128"},
    {ACL_BF16, "ACL_BF16"},
    {ACL_INT4, "ACL_INT4"},
    {ACL_UINT1, "ACL_UINT1"},
    {ACL_COMPLEX32, "ACL_COMPLEX32"},
})

namespace atb_torch {

template <typename Param>
static atb::Operation *OperationCreate(Param &param, std::string operationName) {
    atb::Operation *operation = nullptr;
    atb::Status st = atb::CreateOperation(param, &operation);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("create atb " << operationName << " operation fail, error:" << st);
    }
    return operation;
}

static atb::Operation *ActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ActivationParam param;
    param.activationType = \
        paramJson.value("activationType", R"("ACTIVATION_UNDEFINED")"_json).get<atb::infer::ActivationType>();
    param.scale = paramJson.value("scale", 1.0f);
    param.dim = paramJson.value("dim", -1);
    param.geluMode = paramJson.value("geluMode", R"("TANH_MODE")"_json).get<atb::infer::ActivationParam::GeLUMode>();

    return OperationCreate(param, "Activation");
}

static atb::Operation *AsStridedOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AsStridedParam param;
    for (auto item : paramJson.value("size", R"([])"_json).get<std::vector<int64_t>>()) {
        param.size.push_back(item);
    }
    for (auto item : paramJson.value("stride", R"([])"_json).get<std::vector<int64_t>>()) {
        param.stride.push_back(item);
    }
    for (auto item : paramJson.value("offset", R"([])"_json).get<std::vector<int64_t>>()) {
        param.offset.push_back(item);
    }

    return OperationCreate(param, "AsStrided");
}

static atb::Operation *CumsumOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::CumsumParam param;
    for (auto item : paramJson.value("axes", R"([])"_json).get<std::vector<int64_t>>()) {
        param.axes.push_back(item);
    }
    param.exclusive = paramJson.value("exclusive", false);
    param.reverse = paramJson.value("reverse", false);

    return OperationCreate(param, "Cumsum");
}

static atb::Operation *DynamicNTKOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::DynamicNTKParam param;
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "DynamicNTK");
}

static atb::Operation *GatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::GatherParam param;
    param.axis = paramJson.value("axis", 0);
    param.batchDims = paramJson.value("batchDims", 0);

    return OperationCreate(param, "Gather");
}

static atb::Operation *MultinomialOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::MultinomialParam param;
    param.numSamples = paramJson.value("numSamples", 1);
    param.randSeed = paramJson.value("randSeed", 0);

    return OperationCreate(param, "Multinomial");
}

static atb::Operation *SplitOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SplitParam param;
    param.splitDim = paramJson.value("splitDim", 0);
    param.splitNum = paramJson.value("splitNum", 2);  // 2: translated
    for (auto item : paramJson.value("splitSizes", R"([])"_json).get<std::vector<int32_t>>()) {
        param.splitSizes.push_back(item);
    }

    return OperationCreate(param, "Split");
}

static atb::Operation *ConcatOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ConcatParam param;
    param.concatDim = paramJson.value("concatDim", 0);

    return OperationCreate(param, "Concat");
}

static atb::Operation *SliceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SliceParam param;
    for (auto item : paramJson.value("offsets", R"([])"_json).get<std::vector<int64_t>>()) {
        param.offsets.push_back(item);
    }
    for (auto item : paramJson.value("size", R"([])"_json).get<std::vector<int64_t>>()) {
        param.size.push_back(item);
    }

    return OperationCreate(param, "Slice");
}

static atb::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson.value("perm", R"([])"_json).get<std::vector<int32_t>>()) {
        param.perm.push_back(item);
    }

    return OperationCreate(param, "Transpose");
}

static atb::Operation *ElewiseOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ElewiseParam param;
    param.elewiseType = \
        paramJson.value("elewiseType", R"("ELEWISE_UNDEFINED")"_json).get<atb::infer::ElewiseParam::ElewiseType>();
    param.quantParam.inputScale = paramJson.value("quantParam", R"({})"_json).value("inputScale", 1.0f);
    param.quantParam.inputOffset = paramJson.value("quantParam", R"({})"_json).value("inputOffset", 0);
    param.quantParam.asymmetric  = paramJson.value("quantParam", R"({})"_json).value("asymmetric", false);
    param.mulsParam.varAttr = paramJson.value("mulsParam", R"({})"_json).value("varAttr", 0.0f);
    param.outTensorType = paramJson.value("outTensorType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "Elewise");
}

static atb::Operation *KvCacheOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in KvCache Operation");
    atb::infer::KvCacheParam param;

    return OperationCreate(param, "KvCache");
}

static atb::Operation *ReshapeAndCacheOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReshapeAndCacheParam param;
    param.compressType = paramJson.value("compressType", \
        R"("COMPRESS_TYPE_UNDEFINED")"_json).get<atb::infer::ReshapeAndCacheParam::CompressType>();
    param.kvCacheCfg = paramJson.value("kvCacheCfg", \
        R"("K_CACHE_V_CACHE")"_json).get<atb::infer::ReshapeAndCacheParam::KvCacheCfg>();
    return OperationCreate(param, "ReshapeAndCache");
}

static atb::Operation *LayerNormOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LayerNormParam param;
    param.layerType = paramJson.value("layerType", \
        R"("LAYER_NORM_UNDEFINED")"_json).get<atb::infer::LayerNormParam::LayerNormType>();
    param.normParam.quantType = paramJson.value("normParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.normParam.epsilon = paramJson.value("normParam", R"({})"_json).value("epsilon", 1e-5);
    param.normParam.beginNormAxis = paramJson.value("normParam", R"({})"_json).value("beginNormAxis", 0);
    param.normParam.beginParamsAxis = \
        paramJson.value("normParam", R"({})"_json).value("beginParamsAxis", 0);
    param.normParam.dynamicQuantType = paramJson.value("normParam", \
        R"({})"_json).value("dynamicQuantType", \
        R"("DYNAMIC_QUANT_UNDEFINED")"_json).get<atb::infer::DynamicQuantType>();
    param.preNormParam.quantType = paramJson.value("preNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.preNormParam.epsilon = paramJson.value("preNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.preNormParam.opMode = paramJson.value("preNormParam", R"({})"_json).value("opMode ", 0);
    param.preNormParam.zoomScaleValue = paramJson.value("preNormParam", R"({})"_json).value("zoomScaleValue", 1.0f);
    param.postNormParam.quantType = paramJson.value("postNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.postNormParam.epsilon = paramJson.value("postNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.postNormParam.opMode = paramJson.value("postNormParam", R"({})"_json).value("opMode", 0);
    param.postNormParam.zoomScaleValue = paramJson.value("postNormParam", R"({})"_json).value("zoomScaleValue", 1.0f);

    return OperationCreate(param, "LayerNorm");
}

static atb::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RmsNormParam param;
    param.layerType = paramJson.value("layerType", \
        R"("RMS_NORM_UNDEFINED")"_json).get<atb::infer::RmsNormParam::RmsNormType>();
    param.normParam.quantType = paramJson.value("normParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.normParam.epsilon = paramJson.value("normParam", R"({})"_json).value("epsilon", 1e-5);
    param.normParam.layerNormEps = paramJson.value("normParam", R"({})"_json).value("layerNormEps", 1e-5);
    param.normParam.rstd = paramJson.value("normParam", R"({})"_json).value("rstd", false);
    param.normParam.precisionMode = paramJson.value("normParam", \
        R"({})"_json).value("precisionMode", \
        R"("HIGH_PRECISION_MODE")"_json).get<atb::infer::RmsNormParam::PrecisionMode>();
    param.normParam.modelType = paramJson.value("normParam", \
        R"({})"_json).value("modelType", R"("LLAMA_MODEL")"_json).get<atb::infer::RmsNormParam::ModelType>();
    param.normParam.dynamicQuantType = paramJson.value("normParam", \
        R"({})"_json).value("dynamicQuantType", \
        R"("DYNAMIC_QUANT_UNDEFINED")"_json).get<atb::infer::DynamicQuantType>();
    param.preNormParam.quantType = paramJson.value("preNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.preNormParam.epsilon = paramJson.value("preNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.preNormParam.hasBias = paramJson.value("preNormParam", R"({})"_json).value("hasBias", false);
    param.postNormParam.quantType = paramJson.value("postNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.postNormParam.epsilon = paramJson.value("postNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.postNormParam.hasBias = paramJson.value("postNormParam", R"({})"_json).value("hasBias", false);

    return OperationCreate(param, "RmsNorm");
}

static atb::Operation *FillOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::FillParam param;
    param.withMask = paramJson.value("withMask", false);
    for (auto item : paramJson.value("value", R"([])"_json).get<std::vector<float>>()) {
        param.value.push_back(item);
    }
    for (auto item : paramJson.value("outDim", R"([])"_json).get<std::vector<int64_t>>()) {
        param.outDim.push_back(item);
    }

    return OperationCreate(param, "Fill");
}

static atb::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    std::vector<uint32_t> rankIds;
    for (auto item : paramJson.value("rankIds", R"([])"_json).get<std::vector<uint32_t>>()) {
        rankIds.push_back(item);
    }

    uint32_t groupId = paramJson.value("groupId", 0);
    std::string rankTableFile = paramJson.value("rankTableFile", "");
    uint32_t bufferSize = paramJson.value("bufferSize", 0);

    if (param.rankTableFile != "") {
        // Assign commDomain by rankIds and rank
        uint32_t streamId = atb_speed::GetSingleton<atb_speed::common::DapManager>().GetStreamId();
        param.commDomain = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
            groupId, rankIds, param.rank, param.backend, bufferSize, streamId);

        // Get hcclComm (only created when hccl backend is used and inference across multi nodes)
        param.hcclComm = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(param.commDomain);
    }

    return OperationCreate(param, "AllGather");
}

static atb::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.allReduceType = paramJson.value("allReduceType", "sum");
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");
    param.quantType = paramJson.value("quantType", \
        R"("QUANT_TYPE_UNDEFINED")"_json).get<atb::infer::AllReduceParam::QuantType>();
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "AllReduce");
}

static atb::Operation *BroadcastOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::BroadcastParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "Broadcast");
}

static atb::Operation *ReduceScatterOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReduceScatterParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.backend = paramJson.value("reduceType", "sum");
    param.backend = paramJson.value("backend", "lccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "ReduceScatter");
}

static atb::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParam param;
    param.transposeA = paramJson.value("transposeA", false);
    param.transposeB = paramJson.value("transposeB", true);
    param.hasBias = paramJson.value("hasBias", true);
    param.enAccum = paramJson.value("enAccum", false);
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "Linear");
}

static atb::Operation *LinearParallelOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParallelParam param;
    param.transWeight = paramJson.value("transWeight", true);
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.hasResidual = paramJson.value("hasResidual", false);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.type = paramJson.value("type", \
        R"("LINEAR_ALL_REDUCE")"_json).get<atb::infer::LinearParallelParam::ParallelType>();
    param.keepIntermediate = paramJson.value("keepIntermediate", false);
    param.quantType = paramJson.value("quantType", \
        R"("QUANT_TYPE_UNDEFINED")"_json).get<atb::infer::LinearParallelParam::QuantType>();
    param.quantGroupSize = paramJson.value("quantGroupSize", 0);
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "LinearParallel");
}

static atb::Operation *LinearSparseOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearSparseParam param;
    param.transposeA = paramJson.value("transposeA", false);
    param.transposeB = paramJson.value("transposeB", true);
    param.tilingK = paramJson.value("tilingK", 1);
    param.tilingN = paramJson.value("tilingN", 1);

    return OperationCreate(param, "LinearSparse");
}

static atb::Operation *RopeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RopeParam param;
    param.rotaryCoeff = paramJson.value("rotaryCoeff", 4);  // 4: translated
    param.cosFormat = paramJson.value("cosFormat", 0);

    return OperationCreate(param, "Rope");
}

static atb::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SelfAttentionParam param;
    param.quantType = paramJson.value("quantType", \
        R"("TYPE_QUANT_UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::QuantType>();
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();
    param.headNum = paramJson.value("headNum", 0);
    param.kvHeadNum = paramJson.value("kvHeadNum", 0);
    param.qScale = paramJson.value("qScale", 1.0f);
    param.qkScale = paramJson.value("qkScale", 1.0f);
    param.batchRunStatusEnable = paramJson.value("batchRunStatusEnable", false);
    param.isTriuMask = paramJson.value("isTriuMask", 0);
    param.calcType = paramJson.value("calcType", \
        R"("UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::CalcType>();
    param.kernelType = paramJson.value("kernelType", \
        R"("KERNELTYPE_DEFAULT")"_json).get<atb::infer::SelfAttentionParam::KernelType>();
    param.clampType = paramJson.value("clampType", \
        R"("CLAMP_TYPE_UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::ClampType>();
    param.clampMin = paramJson.value("clampMin", 0);
    param.clampMax = paramJson.value("clampMax", 0);
    param.maskType = paramJson.value("maskType", \
        R"("MASK_TYPE_UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::MaskType>();
    param.kvcacheCfg = paramJson.value("kvcacheCfg", \
        R"("K_CACHE_V_CACHE")"_json).get<atb::infer::SelfAttentionParam::KvCacheCfg>();
    param.scaleType = paramJson.value("scaleType", \
        R"("SCALE_TYPE_TOR")"_json).get<atb::infer::SelfAttentionParam::ScaleType>();
    param.inputLayout = paramJson.value("inputLayout", \
        R"("TYPE_BSND")"_json).get<atb::infer::InputLayout>();
    param.mlaVHeadSize = paramJson.value("mlaVHeadSize", 0);

    return OperationCreate(param, "SelfAttention");
}

static atb::Operation *PagedAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::PagedAttentionParam param;
    param.headNum = paramJson.value("headNum", 0);
    param.qkScale = paramJson.value("qkScale", 1.0f);
    param.kvHeadNum = paramJson.value("kvHeadNum", 0);
    param.maskType = paramJson.value("maskType", \
        R"("MASK_TYPE_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::MaskType>();
    param.batchRunStatusEnable = paramJson.value("batchRunStatusEnable", false);
    param.quantType = paramJson.value("quantType", \
        R"("TYPE_QUANT_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::QuantType>();
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();
    param.hasQuantOffset = paramJson.value("hasQuantOffset", false);
    param.compressType = paramJson.value("compressType", \
        R"("COMPRESS_TYPE_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::CompressType>();
    param.calcType = paramJson.value("calcType", \
        R"("CALC_TYPE_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::CalcType>();
    param.scaleType = paramJson.value("scaleType", \
        R"("SCALE_TYPE_TOR")"_json).get<atb::infer::PagedAttentionParam::ScaleType>();
    param.inputLayout = paramJson.value("inputLayout", \
        R"("TYPE_BSND")"_json).get<atb::infer::InputLayout>();
    param.mlaVHeadSize = paramJson.value("mlaVHeadSize", 0);

    return OperationCreate(param, "PagedAttention");
}

static atb::Operation *TransdataOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransdataParam param;
    param.transdataType = paramJson.value("transdataType", \
        R"("UNDEFINED")"_json).get<atb::infer::TransdataParam::TransdataType>();
    for (auto item : paramJson.value("outCrops", R"([])"_json).get<std::vector<int64_t>>()) {
        param.outCrops.push_back(item);
    }

    return OperationCreate(param, "Transdata");
}

static atb::Operation *WhereOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in Where Operation");
    atb::infer::WhereParam param;

    return OperationCreate(param, "Where");
}

static atb::Operation *SetValueOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SetValueParam param;
    for (auto item : paramJson.value("starts", R"([])"_json).get<std::vector<int64_t>>()) {
        param.starts.push_back(item);
    }
    for (auto item : paramJson.value("ends", R"([])"_json).get<std::vector<int64_t>>()) {
        param.ends.push_back(item);
    }
    for (auto item : paramJson.value("strides", R"([])"_json).get<std::vector<int64_t>>()) {
        param.strides.push_back(item);
    }

    return OperationCreate(param, "SetValue");
}

static atb::Operation *TopkToppSamplingOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TopkToppSamplingParam param;
    param.topkToppSamplingType = paramJson.value("topkToppSamplingType", \
        R"("SINGLE_TOPK_SAMPLING")"_json).get<atb::infer::TopkToppSamplingParam::TopkToppSamplingType>();
    for (auto item : paramJson.value("randSeeds", R"([])"_json).get<std::vector<uint32_t>>()) {
        param.randSeeds.push_back(item);
    }
    param.randSeed = paramJson.value("randSeed", 0);
    param.topk = paramJson.value("topk", 100);

    return OperationCreate(param, "TopkToppSampling");
}

static atb::Operation *PadOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in Pad Operation");
    atb::infer::PadParam param;

    return OperationCreate(param, "Pad");
}

static atb::Operation *UnpadOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in Unpad Operation");
    atb::infer::UnpadParam param;

    return OperationCreate(param, "Unpad");
}

static atb::Operation *NonzeroOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in Nonzero Operation");
    atb::infer::NonzeroParam param;

    return OperationCreate(param, "Nonzero");
}

static atb::Operation *OnehotOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in Onehot Operation");
    atb::infer::OnehotParam param;

    return OperationCreate(param, "Onehot");
}

static atb::Operation *IndexAddOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::IndexAddParam param;
    param.indexType = paramJson.value("indexType", \
        R"("INDEX_UNDEFINED")"_json).get<atb::infer::IndexAddParam::IndexType>();
    param.axis = paramJson.value("axis", 0);

    return OperationCreate(param, "IndexAdd");
}

static atb::Operation *SendOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SendParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.destRank = paramJson.value("destRank", 1);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "Send");
}

static atb::Operation *RecvOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RecvParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.srcRank = paramJson.value("srcRank", 1);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "Recv");
}

static atb::Operation *AllToAllOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllToAllParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "AllToAll");
}

static atb::Operation *AllToAllVOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllToAllVParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    for (auto item : paramJson.value("sendCounts", R"([])"_json).get<std::vector<int64_t>>()) {
        param.sendCounts.push_back(item);
    }
    for (auto item : paramJson.value("sdispls", R"([])"_json).get<std::vector<int64_t>>()) {
        param.sdispls.push_back(item);
    }
    for (auto item : paramJson.value("recvCounts", R"([])"_json).get<std::vector<int64_t>>()) {
        param.recvCounts.push_back(item);
    }
    for (auto item : paramJson.value("rdispls", R"([])"_json).get<std::vector<int64_t>>()) {
        param.rdispls.push_back(item);
    }
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "AllToAllV");
}

static atb::Operation *W8A16OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNWeightQuantBatchMatmulParam aclnnParam;
    aclnnParam.hasBias = paramJson.value("hasBias", false);
    aclnnParam.quantGroupSize = paramJson.value("quantGroupSize", 0);
    aclnnParam.transposeB = paramJson.value("transposeB", false);
    return new atb_speed::common::W8A16Operation("W8A16LinearNode", aclnnParam);
}

static atb::Operation *W4A16OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNWeightQuantBatchMatmulParam aclnnParam;
    aclnnParam.hasBias = paramJson.value("hasBias", false);
    aclnnParam.quantGroupSize = paramJson.value("quantGroupSize", 0);
    aclnnParam.transposeB = paramJson.value("transposeB", false);
    return new atb_speed::common::W4A16Operation("W4A16LinearNode", aclnnParam);
}

static atb::Operation *W8A8OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNQuantMatmulParam aclnnParam;
    aclnnParam.transposeB = paramJson.value("transposeB", false);
    return new atb_speed::common::W8A8Operation("W8A8LinearNode", aclnnParam);
}

static atb::Operation *W4A8OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNW4A8Param aclnnParam;
    aclnnParam.hasBias = paramJson.value("hasBias", false);
    return new atb_speed::common::W4A8Operation("W4A8LinearNode", aclnnParam);
}

static atb::Operation *IndexputOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNIndexputParam aclnnParam;
    aclnnParam.accumulate = paramJson.value("accumulate", false);
    aclnnParam.unsafe = paramJson.value("unsafe", true);
    return new atb_speed::common::IndexputOperation("IndexputNode", aclnnParam);
}

static atb::Operation *IndexSelectOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::IndexSelectParam indexSelectParam;
    indexSelectParam.dim = paramJson.value("dim", 0);
    return new atb_speed::common::IndexSelectOperation("IndexSelectOperation", indexSelectParam);
}

static atb::Operation *StdOperationCreate(const nlohmann::json &paramJson)
{
    return new atb_speed::common::StdOperation("Std");
}

static atb::Operation *MatmulOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNMatmulParam param;
    param.transposeB = paramJson.value("transposeB", false);
    param.hasBias = paramJson.value("hasBias", false);
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();
    return new atb_speed::common::MatmulOperation("MatmulOperation", param);
}

static atb::Operation *GroupedMatmulOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNGroupedMatmulParam param;
    param.transposeB = paramJson.value("transposeB", false);
    // Default quantType: 0 (no quant)
    param.quantType = paramJson.value("quantType", 0);
    param.hasBias = paramJson.value("hasBias", false);
    param.outDataType = paramJson.value("outDataType", R"("ACL_FLOAT16")"_json).get<aclDataType>();
    return new atb_speed::common::GroupedMatmulOperation("GroupedMatmul", param);
}

static atb::Operation *SoftmaxOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SoftmaxParam param;
    for (auto item : paramJson.value("axes", R"([])"_json).get<std::vector<int64_t>>()) {
        param.axes.push_back(item);
    }
    return OperationCreate(param, "Softmax");
}

static atb::Operation *SortOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SortParam param;
    for (auto item : paramJson.value("num", R"([])"_json).get<std::vector<int32_t>>()) {
        param.num.push_back(item);
    }
    return OperationCreate(param, "Sort");
}

static atb::Operation *ArgsortOperationCreate(const nlohmann::json &paramJson)
{
    return new atb_speed::common::ArgSortOperation("Argsort");
}

static atb::Operation *GroupTopkOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::GroupTopkParam param;
    // Default groupNum: 1
    param.groupNum = paramJson.value("groupNum", 1);
    // Default k: 0
    param.k = paramJson.value("k", 0);
    return OperationCreate(param, "GroupTopk");
}

static atb::Operation *MoeTopkSoftmaxOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::MoeTopkSoftmaxParam param;
    // Default topkNum: 2
    param.topkNum = paramJson.value("topkNum", 2);
    return new atb_speed::common::MoeTopkSoftmaxOperation("MoeTopkSoftmax", param);
}

static atb::Operation *ReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReduceParam param;
    param.reduceType = \
        paramJson.value("reduceType", R"("REDUCE_UNDEFINED")"_json).get<atb::infer::ReduceParam::ReduceType>();
    for (auto item : paramJson.value("axis", R"([])"_json).get<std::vector<int64_t>>()) {
        param.axis.push_back(item);
    }
    return OperationCreate(param, "Reduce");
}

static atb::Operation *DynamicQuantOperationCreate(const nlohmann::json &paramJson)
{
    return new atb_speed::common::DynamicQuantOperation("DynamicQuant");
}

static atb::Operation *VectorNormOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNVectorNormParam param;
    return new atb_speed::common::VectorNormOperation("VectorNorm", param);
}

static atb::Operation *MoeInitRoutingOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::MoeInitRoutingParam param;
    // Default topkNum: 2
    param.topkNum = paramJson.value("topkNum", 2);
    // Default expertNum: 8
    param.expertNum = paramJson.value("expertNum", 8);
    return new atb_speed::common::MoeInitRoutingOperation("MoeInitRouting", param);
}

static atb::Operation *MoeComputeExpertTokensOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::MoeComputeExpertTokensParam param;
    // Default expertNum: 8
    param.expertNum = paramJson.value("expertNum", 8);
    return new atb_speed::common::MoeComputeExpertTokensOperation("MoeComputeTokens", param);
}

static atb::Operation *MoeTokenUnpermuteOperationCreate(const nlohmann::json &paramJson)
{
    return new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermute");
}

static atb::Operation *GatingOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::GatingParam param;
    // Default topkExpertNum: 0
    param.topkExpertNum = paramJson.value("topkExpertNum", 0);
    // Default cumSumNum: 0
    param.cumSumNum = paramJson.value("cumSumNum", 0);
    param.cumSumInt64 = paramJson.value("cumSumInt64", false);
    for (auto item : paramJson.value("deviceExpert", R"([])"_json).get<std::vector<int32_t>>()) {
        param.deviceExpert.push_back(item);
    }
    return OperationCreate(param, "Gating");
}

static atb::Operation *PromptFlashAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNFlashAttentionParam aclnnParam;

        aclnnParam.needMask = paramJson.value("needMask", false);
        aclnnParam.numHeads = paramJson.value("numHeads", 0);
        aclnnParam.scaleValue = paramJson.value("scaleValue", 1.0);
        aclnnParam.preTokens = paramJson.value("preTokens", 214748647);
        aclnnParam.nextTokens = paramJson.value("nextTokens", 65535);
        aclnnParam.inputLayout = paramJson.value("inputLayout", "");
        aclnnParam.numKeyValueHeads = paramJson.value("numKeyValueHeads", 0);
        aclnnParam.sparseMode = paramJson.value("sparseMode", 0);
        aclnnParam.innerPrecise = paramJson.value("innerPrecise", 1);
        return new atb_speed::common::PromptFlashAttentionOperation("PromptFlashAttentionOperation", aclnnParam);
}

static atb::Operation *RepeatOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RepeatParam param;
    for (auto item : paramJson.value("multiples", R"([])"_json).get<std::vector<int64_t>>()) {
        param.multiples.push_back(item);
    }

    return OperationCreate(param, "Repeat");
}

static atb::Operation *BlockCopyOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::BlockCopyParam param;
    return OperationCreate(param, "BlockCopy");
}

static atb::Operation *DequantRopeQuantKvcacheOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNDequantRopeQuantKvcacheParam aclnnParam;
    for (auto item : paramJson["sizeSpilts"]) {
        aclnnParam.sizeSpilts = paramJson["sizeSpilts"].get<std::vector<int64_t>>();
    }
    if (paramJson.contains("kvOutput")) {
        aclnnParam.kvOutput = paramJson["kvOutput"].get<bool>();
    }
    if (paramJson.contains("quantMode")) {
        aclnnParam.quantMode = paramJson["quantMode"].get<std::string>();
    }
    if (paramJson.contains("layout")) {
        aclnnParam.layout = paramJson["layout"].get<std::string>();
    }
    atb_speed::common::DequantRopeQuantKvcacheOperation *dequantRopeQuantKvcacheOperation = \
        new atb_speed::common::DequantRopeQuantKvcacheOperation("DequantRopeQuantKvcacheOperation", aclnnParam);
    ATB_SPEED_LOG_DEBUG("DequantRopeQuantKvcacheOperation Create");
    return dequantRopeQuantKvcacheOperation;
}

static atb::Operation *AclrtCmoAsyncOperationCreate(const nlohmann::json &paramJson)
{
    ATB_SPEED_LOG_DEBUG("paramJson " << paramJson << " is not used in Nonzero Operation");
    return new atb_speed::common::AclrtCmoAsyncOperation("AclrtCmoAsync");
}

static atb::Operation *CastOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNCastParam castParam;
    if (paramJson.contains("dtype")) {
        castParam.dtype = paramJson["dtype"].get<aclDataType>();
    } else {
        return nullptr;
    }
    if (castParam.dtype == ACL_DT_UNDEFINED) {
        return nullptr;
    }
    return new atb_speed::common::CastOperation("CastOperation", castParam);
}

static atb::Operation *ScatterOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNScatterParam scatterParam;
    if (paramJson.contains("dim")) {
        scatterParam.dim = paramJson["dim"].get<int64_t>();
    } else {
        return nullptr;
    }
    if (paramJson.contains("reduce")) {
        int64_t reduceValue = paramJson["reduce"].get<int64_t>();
        if (reduceValue == 0) {
            scatterParam.reduce = atb_speed::common::ReduceType::REPLACE;
        } else if (reduceValue == 1) {
            scatterParam.reduce = atb_speed::common::ReduceType::ADD;
        } else if (reduceValue == 2) {
            scatterParam.reduce = atb_speed::common::ReduceType::MULTIPLY;
        } else {
            ATB_SPEED_LOG_ERROR("Invalid reduce type value: " << reduceValue);
            return nullptr;
        }
    } else {
        return nullptr;
    }
    return new atb_speed::common::ScatterOperation("ScatterOperation", scatterParam, false);
}

static atb::Operation *DequantSwigluQuantOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNDequantSwigluQuantParam aclnnParam;
    if (paramJson.contains("activateLeft")) {
        aclnnParam.activateLeft = paramJson["activateLeft"].get<bool>();
    }
    if (paramJson.contains("quantMode")) {
        aclnnParam.quantMode = paramJson["quantMode"].get<std::string>();
    }
    if (paramJson.contains("inTensorsNum")) {
        aclnnParam.inTensorsNum = paramJson["inTensorsNum"].get<int>();
    }
    atb_speed::common::DequantSwigluQuantOperation *dequantSwigluQuantOperation = \
        new atb_speed::common::DequantSwigluQuantOperation("DequantSwigluQuantOperation", aclnnParam);
    ATB_SPEED_LOG_DEBUG("DequantSwigluQuantOperation Create");
    return dequantSwigluQuantOperation;
}

static atb::Operation *InplaceNanToNumOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNNanToNumParam aclnnNanToNumParam;
    if (paramJson.contains("nanValue")) {
        aclnnNanToNumParam.nanValue = paramJson["nanValue"].get<float>();
    }
    if (paramJson.contains("posInfValue")) {
        aclnnNanToNumParam.posInfValue = paramJson["posInfValue"].get<float>();
    }
    if (paramJson.contains("negInfValue")) {
        aclnnNanToNumParam.negInfValue = paramJson["negInfValue"].get<float>();
    }
    atb_speed::common::InplaceNanToNumOperation *inplaceNanToNumOperation = \
        new atb_speed::common::InplaceNanToNumOperation("InplaceNanToNumOperation", aclnnNanToNumParam);
    ATB_SPEED_LOG_DEBUG("InplaceNanToNumOperation Create");
    return inplaceNanToNumOperation;
}

REGISTER_OPERATION(Activation, ActivationOperationCreate);
REGISTER_OPERATION(Gather, GatherOperationCreate);
REGISTER_OPERATION(Split, SplitOperationCreate);
REGISTER_OPERATION(Concat, ConcatOperationCreate);
REGISTER_OPERATION(Slice, SliceOperationCreate);
REGISTER_OPERATION(Transpose, TransposeOperationCreate);
REGISTER_OPERATION(Elewise, ElewiseOperationCreate);
REGISTER_OPERATION(ReshapeAndCache, ReshapeAndCacheOperationCreate);
REGISTER_OPERATION(LayerNorm, LayerNormOperationCreate);
REGISTER_OPERATION(RmsNorm, RmsNormOperationCreate);
REGISTER_OPERATION(AllGather, AllGatherOperationCreate);
REGISTER_OPERATION(AllReduce, AllReduceOperationCreate);
REGISTER_OPERATION(Linear, LinearOperationCreate);
REGISTER_OPERATION(LinearParallel, LinearParallelOperationCreate);
REGISTER_OPERATION(LinearSparse, LinearSparseOperationCreate);
REGISTER_OPERATION(Rope, RopeOperationCreate);
REGISTER_OPERATION(SelfAttention, SelfAttentionOperationCreate);
REGISTER_OPERATION(PagedAttention, PagedAttentionOperationCreate);
REGISTER_OPERATION(Transdata, TransdataOperationCreate);
REGISTER_OPERATION(W8A16MatMul, W8A16OperationCreate);
REGISTER_OPERATION(W4A16MatMul, W4A16OperationCreate);
REGISTER_OPERATION(W8A8MatMul, W8A8OperationCreate);
REGISTER_OPERATION(W4A8MatMul, W4A8OperationCreate);
REGISTER_OPERATION(Indexput, IndexputOperationCreate);
REGISTER_OPERATION(IndexSelect, IndexSelectOperationCreate);
REGISTER_OPERATION(Std, StdOperationCreate);
REGISTER_OPERATION(Matmul, MatmulOperationCreate);
REGISTER_OPERATION(GroupedMatmul, GroupedMatmulOperationCreate);
REGISTER_OPERATION(Softmax, SoftmaxOperationCreate);
REGISTER_OPERATION(Sort, SortOperationCreate);
REGISTER_OPERATION(Argsort, ArgsortOperationCreate);
REGISTER_OPERATION(GroupTopk, GroupTopkOperationCreate);
REGISTER_OPERATION(MoeTopkSoftmax, MoeTopkSoftmaxOperationCreate);
REGISTER_OPERATION(Reduce, ReduceOperationCreate);
REGISTER_OPERATION(DynamicQuant, DynamicQuantOperationCreate);
REGISTER_OPERATION(VectorNorm, VectorNormOperationCreate);
REGISTER_OPERATION(MoeInitRouting, MoeInitRoutingOperationCreate);
REGISTER_OPERATION(MoeComputeExpertTokens, MoeComputeExpertTokensOperationCreate);
REGISTER_OPERATION(MoeTokenUnpermute, MoeTokenUnpermuteOperationCreate);
REGISTER_OPERATION(Gating, GatingOperationCreate);
REGISTER_OPERATION(PromptFlashAttention, PromptFlashAttentionOperationCreate);
REGISTER_OPERATION(AsStrided, AsStridedOperationCreate);
REGISTER_OPERATION(Cumsum, CumsumOperationCreate);
REGISTER_OPERATION(DynamicNTK, DynamicNTKOperationCreate);
REGISTER_OPERATION(Multinomial, MultinomialOperationCreate);
REGISTER_OPERATION(KvCache, KvCacheOperationCreate);
REGISTER_OPERATION(Fill, FillOperationCreate);
REGISTER_OPERATION(Broadcast, BroadcastOperationCreate);
REGISTER_OPERATION(ReduceScatter, ReduceScatterOperationCreate);
REGISTER_OPERATION(Where, WhereOperationCreate);
REGISTER_OPERATION(Repeat, RepeatOperationCreate);
REGISTER_OPERATION(SetValue, SetValueOperationCreate);
REGISTER_OPERATION(TopkToppSampling, TopkToppSamplingOperationCreate);
REGISTER_OPERATION(Pad, PadOperationCreate);
REGISTER_OPERATION(Unpad, UnpadOperationCreate);
REGISTER_OPERATION(Nonzero, NonzeroOperationCreate);
REGISTER_OPERATION(Onehot, OnehotOperationCreate);
REGISTER_OPERATION(IndexAdd, IndexAddOperationCreate);
REGISTER_OPERATION(Send, SendOperationCreate);
REGISTER_OPERATION(Recv, RecvOperationCreate);
REGISTER_OPERATION(AllToAll, AllToAllOperationCreate);
REGISTER_OPERATION(AllToAllV, AllToAllVOperationCreate);
REGISTER_OPERATION(BlockCopy, BlockCopyOperationCreate);
REGISTER_OPERATION(DequantRopeQuantKvcache, DequantRopeQuantKvcacheOperationCreate);
REGISTER_OPERATION(AclrtCmoAsync, AclrtCmoAsyncOperationCreate);
REGISTER_OPERATION(Cast, CastOperationCreate);
REGISTER_OPERATION(Scatter, ScatterOperationCreate);
REGISTER_OPERATION(DequantSwigluQuant, DequantSwigluQuantOperationCreate);
REGISTER_OPERATION(InplaceNanToNum, InplaceNanToNumOperationCreate);
} // namespace atb_torch
