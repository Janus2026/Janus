/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "operation_creator.h"

#include <functional>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_factory.h"
#include "atb_speed/utils/hccl_runner.h"
#include "../../operations/fusion/linear/linear.h"
#include "../../operations/aclnn/ops/attn_operation.h"
#include "../../operations/aclnn/ops/add_rms_norm_dynamic_quant_operation.h"
#include "../../operations/aclnn/ops/add_rms_norm_quant_operation.h"
#include "../../operations/aclnn/ops/w8a16_operation.h"
#include "../../operations/aclnn/ops/w4a16_operation.h"
#include "../../operations/aclnn/ops/w8a8_operation.h"
#include "../../operations/aclnn/ops/w4a8_operation.h"
#include "../../operations/aclnn/ops/matmul_operation.h"
#include "../../operations/aclnn/ops/grouped_matmul_operation.h"
#include "../../operations/aclnn/ops/grouped_matmul_swiglu_operation.h"
#include "../../operations/aclnn/ops/matmul_allreduce_operation.h"
#include "../../operations/aclnn/ops/dequant_rope_quant_kvcache_operation.h"
#include "../../operations/aclnn/ops/dequant_swiglu_quant_operation.h"
#include "../../operations/aclnn/ops/obfuscation_setup_operation.h"

namespace atb_speed {
using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &paramJson)>;

static atb::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("allReduceType") != paramJson.end()) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    }
    ATB_SPEED_LOG_DEBUG("AllReduceParam rank:" << param.rank);
    ATB_SPEED_LOG_DEBUG("AllReduceParam rankSize:" << param.rankSize);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    ATB_SPEED_LOG_DEBUG("AllGatherParam rank:" << param.rank);
    ATB_SPEED_LOG_DEBUG("AllGatherParam rankSize:" << param.rankSize);
    ATB_SPEED_LOG_DEBUG("AllGatherParam backend:" << param.backend);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *BroadcastOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::BroadcastParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    ATB_SPEED_LOG_DEBUG("BroadcastParam rank:" << param.rank << "rankSize:" << param.rankSize
                  << "rankRoot:" << param.rankRoot << "backend: " << param.backend);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TopkToppSamplingOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TopkToppSamplingParam param;
    if (paramJson.find("topk") != paramJson.end()) {
        param.topk = paramJson["topk"].get<int>();
    }
    if (paramJson.find("randSeed") != paramJson.end()) {
        param.randSeed = paramJson["randSeed"].get<int>();
    }
    if (paramJson.find("randSeeds") != paramJson.end()) {
        for (auto &item : paramJson["randSeeds"]) {
            param.randSeeds.push_back(item.get<uint32_t>());
        }
    }
    if (paramJson.find("topkToppSamplingType") != paramJson.end()) {
        param.topkToppSamplingType =
            paramJson["topkToppSamplingType"].get<atb::infer::TopkToppSamplingParam::TopkToppSamplingType>();
    }
    ATB_SPEED_LOG_DEBUG("TopkToppSamplingParam topk:" << param.topk << "randSeed:" << param.randSeed
                  << "randSeeds:" << param.randSeeds << "topkToppSamplingType:" << param.topkToppSamplingType);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    ATB_SPEED_LOG_DEBUG("LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", outDataType:" << param.outDataType);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SendOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SendParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("destRank") != paramJson.end()) {
        param.destRank = paramJson["destRank"].get<uint32_t>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_SPEED_LOG_DEBUG("SendParam rank:" << param.rank);
    ATB_SPEED_LOG_DEBUG("SendParam rankSize:" << param.rankSize);
    ATB_SPEED_LOG_DEBUG("SendParam rankRoot:" << param.rankRoot);
    ATB_SPEED_LOG_DEBUG("SendParam destRank:" << param.destRank);
    ATB_SPEED_LOG_DEBUG("SendParam commDomain:" << param.commDomain);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *RecvOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RecvParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("srcRank") != paramJson.end()) {
        param.srcRank = paramJson["srcRank"].get<uint32_t>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_SPEED_LOG_DEBUG("RecvParam rank:" << param.rank);
    ATB_SPEED_LOG_DEBUG("RecvParam rankSize:" << param.rankSize);
    ATB_SPEED_LOG_DEBUG("RecvParam rankRoot:" << param.rankRoot);
    ATB_SPEED_LOG_DEBUG("RecvParam srcRank:" << param.srcRank);
    ATB_SPEED_LOG_DEBUG("SendParam commDomain:" << param.commDomain);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TransdataOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransdataParam param;
    if (paramJson.contains("transdataType")) {
        ATB_SPEED_LOG_ERROR("Only support ND_TO_FRACTAL_NZ");
    }
    param.transdataType = atb::infer::TransdataParam::ND_TO_FRACTAL_NZ;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    ATB_SPEED_LOG_DEBUG("transpose(" << param.perm << ")");
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *ActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        param.activationType = atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    ATB_SPEED_LOG_DEBUG("ActivationParam activationType:" << param.activationType << ", scale:" << param.scale);
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *AclNNMatmulAllreduceOperationCreate(const nlohmann::json &paramJson)
{
    int rank = 0;
    int worldSize = 1;
    if (paramJson.contains("rank")) {
        rank = (paramJson["rank"].get<int>());
    }
    if (paramJson.contains("worldSize")) {
        worldSize = paramJson["worldSize"].get<int>();
    }
    atb_speed::HcclRunner hcclRunner(rank, worldSize, 0);
    HcclComm hcommInfo = hcclRunner.CreateHcclCommInMulitProcessByRootInfo();
    atb::Operation *op = new atb_speed::common::MatmulAllreduceOperation(
        "matmulAllReduce", hcommInfo);
    return op;
}

static atb::Operation *AclNNAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNAttnParam attnParam;
    if (paramJson.contains("headNum")) {
        attnParam.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("kvHeadNum")) {
        attnParam.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("headDim")) {
        attnParam.headDim = paramJson["headDim"].get<int>();
    }
    if (paramJson.contains("isFA")) {
        attnParam.isFA = paramJson["isFA"].get<bool>();
    }
    attnParam.isPrefill = false;
    attnParam.hasMask = true;
    attnParam.hasKVQuant = true;
    attnParam.hasQuantOffset = true;
    atb::Operation *op = new atb_speed::common::AttnOperation("AclNNAttentionNode", attnParam);
    return op;
}

void QuantOperationParamLoad(const nlohmann::json &paramJson,
                             atb_speed::common::AclNNWeightQuantBatchMatmulParam &aclnnParam)
{
    if (paramJson.contains("hasBias")) {
        aclnnParam.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("quantGroupSize")) {
        aclnnParam.quantGroupSize = paramJson["quantGroupSize"].get<int>();
    }
    if (paramJson.contains("transposeB")) {
        aclnnParam.transposeB = paramJson["transposeB"].get<bool>();
    }
}

static atb::Operation *W8A16OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNWeightQuantBatchMatmulParam aclnnParam;

    QuantOperationParamLoad(paramJson, aclnnParam);

    atb_speed::common::W8A16Operation *w8a16Operation = \
        new atb_speed::common::W8A16Operation("W8A16LinearNode", aclnnParam);
    ATB_SPEED_LOG_DEBUG("W8A16Operation Create");
    return w8a16Operation;
}

static atb::Operation *W4A16OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNWeightQuantBatchMatmulParam aclnnParam;

    QuantOperationParamLoad(paramJson, aclnnParam);

    atb_speed::common::W4A16Operation *w4a16Operation = \
        new atb_speed::common::W4A16Operation("W4A16LinearNode", aclnnParam);
    ATB_SPEED_LOG_DEBUG("W4A16Operation Create");
    return w4a16Operation;
}

static atb::Operation *W8A8OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNQuantMatmulParam aclnnParam;
    if (paramJson.contains("hasBias")) {
        aclnnParam.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        aclnnParam.transposeB = paramJson["transposeB"].get<bool>();
    }

    atb_speed::common::W8A8Operation *w8a8Operation = \
        new atb_speed::common::W8A8Operation("W8A8Operation", aclnnParam);
    
    ATB_SPEED_LOG_DEBUG("W8A8Operation Create");
    return w8a8Operation;
}

static atb::Operation *W4A8OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNW4A8Param aclnnParam;
    if (paramJson.contains("hasBias")) {
        aclnnParam.hasBias = paramJson["hasBias"].get<bool>();
    }
    atb_speed::common::W4A8Operation *w4a8Operation = \
        new atb_speed::common::W4A8Operation("W4A8Operation", aclnnParam);
    
    ATB_SPEED_LOG_DEBUG("W4A8Operation Create");
    return w4a8Operation;
}

static atb::Operation *MatmulOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNMatmulParam aclnnParam;
    if (paramJson.contains("hasBias")) {
        aclnnParam.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        aclnnParam.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("outDataType")) {
        aclnnParam.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    atb_speed::common::MatmulOperation *matmulOperation = \
        new atb_speed::common::MatmulOperation("MatmulOperation", aclnnParam);

    ATB_SPEED_LOG_DEBUG("MatmulOperation Create");
    return matmulOperation;
}

static atb::Operation *GroupedMatmulSwigluOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNGroupedSwigluMatmulParam aclnnParam;
    if (paramJson.contains("transposeB")) {
        aclnnParam.transposeB = paramJson["transposeB"].get<bool>();
    }
    atb_speed::common::GroupedMatmulSwigluOperation *groupedMatmulSwigluOperation = \
        new atb_speed::common::GroupedMatmulSwigluOperation("GroupMatMulSwigluNode", aclnnParam);
    ATB_SPEED_LOG_DEBUG("GroupedMatmulSwigluOperation Create");
    return groupedMatmulSwigluOperation;
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

static atb::Operation *GroupedMatmulOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNGroupedMatmulParam aclnnParam;
    if (paramJson.contains("transposeB")) {
        aclnnParam.transposeB = paramJson["transposeB"].get<bool>();
    }
    atb_speed::common::GroupedMatmulOperation *groupedMatmulOperation = \
        new atb_speed::common::GroupedMatmulOperation("GroupMatMulNode", aclnnParam);
    ATB_SPEED_LOG_DEBUG("GroupedMatmulOperation Create");
    return groupedMatmulOperation;
}
static atb::Operation *LinearWithLoraOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::FusionLinearParam param;
    if (paramJson.contains("isBF16")) {
        param.isBF16 = paramJson["isBF16"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("transposeType")) {
        param.transposeType = paramJson["transposeType"].get<int>();
    }
    if (paramJson.contains("supportLora")) {
        param.supportLora = paramJson["supportLora"].get<bool>();
    }
    if (paramJson.contains("loraEnableGMM")) {
        param.loraEnableGMM = paramJson["loraEnableGMM"].get<bool>();
    }
    atb::Operation *op;
    FusionLinear(param, &op);
    return op;
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

static atb::Operation *AddRmsNormDynamicQuantOperationCreate(const nlohmann::json &paramJson)
{
    double epsilon = 1e-6;
    if (paramJson.contains("epsilon")) {
        epsilon = paramJson["epsilon"].get<double>();
    }
    atb_speed::common::AddRmsNormDynamicQuantOperation *addRmsNormDynamicQuantOperation = \
        new atb_speed::common::AddRmsNormDynamicQuantOperation("AddRmsNormDynamicQuantOperation", epsilon);
    ATB_SPEED_LOG_DEBUG("AddRmsNormDynamicQuantOperation Create");
    return addRmsNormDynamicQuantOperation;
}

static atb::Operation *AddRmsNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    double epsilon = 1e-6;
    if (paramJson.contains("epsilon")) {
        epsilon = paramJson["epsilon"].get<double>();
    }
    atb_speed::common::AddRmsNormQuantOperation *addRmsNormQuantOperation = \
        new atb_speed::common::AddRmsNormQuantOperation("AddRmsNormQuantOperation", epsilon);
    ATB_SPEED_LOG_DEBUG("AddRmsNormQuantOperation Create");
    return addRmsNormQuantOperation;
}

static atb::Operation *ObfuscationSetupOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::ObfuscationSetupParam param;
    if (paramJson.contains("fdtoClose")) {
        param.fdtoClose = paramJson["fdtoClose"].get<int32_t>();
    }
    if (paramJson.contains("dataType")) {
        param.dataType = paramJson["dataType"].get<int32_t>();
    }
    if (paramJson.contains("hiddenSizePerRank")) {
        param.hiddenSizePerRank = paramJson["hiddenSizePerRank"].get<int32_t>();
    }
    if (paramJson.contains("tpRank")) {
        param.tpRank = paramJson["tpRank"].get<int32_t>();
    }
    if (paramJson.contains("cmd")) {
        param.cmd = paramJson["cmd"].get<int32_t>();
    }
    if (paramJson.contains("threadNum")) {
        param.threadNum = paramJson["threadNum"].get<int32_t>();
    }
    atb_speed::common::ObfuscationSetupOperation *obfuscationSetupOperation = \
        new atb_speed::common::ObfuscationSetupOperation("ObfuscationSetupOperation", param);
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation Create");
    return obfuscationSetupOperation;
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"TopkToppSamplingOperation", &TopkToppSamplingOperationCreate},
    {"BroadcastOperation", &BroadcastOperationCreate},
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"AllGatherOperation", &AllGatherOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"TransdataOperation", &TransdataOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
    {"AclNNAttention", &AclNNAttentionOperationCreate},
    {"W8A16Operation", &W8A16OperationCreate},
    {"W4A16Operation", &W4A16OperationCreate},
    {"W8A8Operation", &W8A8OperationCreate},
    {"W4A8Operation", &W4A8OperationCreate},
    {"MatmulOperation", MatmulOperationCreate},
    {"DequantRopeQuantKvcacheOperation", DequantRopeQuantKvcacheOperationCreate},
    {"GroupedMatmulOperationCreate", GroupedMatmulOperationCreate},
    {"GroupedMatmulSwigluOperationCreate", GroupedMatmulSwigluOperationCreate},
    {"LinearWithLoraOperationCreate", LinearWithLoraOperationCreate},
    {"AclNNMatmulAllreduce", &AclNNMatmulAllreduceOperationCreate},
    {"DequantSwigluQuantOperation", &DequantSwigluQuantOperationCreate},
    {"AddRmsNormDynamicQuantOperation", &AddRmsNormDynamicQuantOperationCreate},
    {"AddRmsNormQuantOperation", &AddRmsNormQuantOperationCreate},
    {"ObfuscationSetupOperation", &ObfuscationSetupOperationCreate},
    // Do not register operations or layers here, expect atb's operations;
    // Please register them in files `.cpp`, such as `models/baichuan2/13b/layer/flash_attention_layer.cpp`;
};

atb::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        ATB_SPEED_LOG_ERROR(opName << " parse json fail, error:" << e.what());
        return nullptr;
    }
    auto operation = atb_speed::OperationFactory::CreateOperation(opName, paramJson);
    if (operation != nullptr) {
        ATB_SPEED_LOG_DEBUG("Get Op from the OperationFactory, opName: " << opName);
        return operation;
    }

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_SPEED_LOG_ERROR("Not support opName:" << opName);
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception &e) {
        ATB_SPEED_LOG_ERROR(opName << " parse json fail, error:" << e.what());
    }
    return nullptr;
}
}