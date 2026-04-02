/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include "acl/acl.h"
#include "atb/context.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "atb/atb_infer.h"
#include "test_util.h"
#include "customize_op_params.h"
#include "custom_paged_attention_function.h"

// Test constants
const uint32_t NTOKENS = 16;
const uint32_t BATCH_SIZE = NTOKENS;
const uint32_t MAX_SEQ_LEN = 20000;
const uint32_t HEAD_NUM = 128;
const uint32_t KV_HEAD_NUM = 4;
const uint32_t HEAD_SIZE = 128;
const uint32_t BLOCK_NUM = 20;
const uint32_t BLOCK_SIZE = 128;
const uint32_t MAX_CONTEXT_LEN = MAX_SEQ_LEN;


constexpr uint32_t TILINGMIN = 128; // 512: TILINGMIN * sizeof(uint32_t)
constexpr uint32_t TILING_PARA_SIZE = 17;
constexpr uint32_t TILING_HEAD_SIZE = 44;
// Function to prepare input tensors
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, unsigned int seed = 0)
{
    srand(seed);
    // Create query tensor with random data
    std::vector<float> queryData(NTOKENS * HEAD_NUM * HEAD_SIZE);
    for (size_t i = 0; i < queryData.size(); i++) {
        queryData[i] = (rand() % 2000 - 1000) / 1000.0f;  // Random values between -1.0 and 1.0
    }
    atb::Tensor query = CreateTensorFromVector(
        contextPtr, stream, queryData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE});
    
    // Create key, value tensor with random data
    std::vector<float> kvCacheData(BLOCK_NUM * BLOCK_SIZE * KV_HEAD_NUM * HEAD_SIZE);
    for (size_t i = 0; i < kvCacheData.size(); i++) {
        kvCacheData[i] = (rand() % 2000 - 1000) / 1000.0f;  // Random values between -1.0 and 1.0
    }
    atb::Tensor kCache = CreateTensorFromVector(contextPtr,
        stream,
        kvCacheData,
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {BLOCK_NUM, BLOCK_SIZE, KV_HEAD_NUM, HEAD_SIZE});
    atb::Tensor vCache = CreateTensorFromVector(contextPtr,
        stream,
        kvCacheData,
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {BLOCK_NUM, BLOCK_SIZE, KV_HEAD_NUM, HEAD_SIZE});
    
    // Create blockTables
    uint32_t maxNumBlocksPerQuery = (MAX_CONTEXT_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int32_t> blockTablesData(NTOKENS * maxNumBlocksPerQuery, 0);
    for (size_t i = 0; i < blockTablesData.size(); i++) {
        blockTablesData.at(i) = rand() % (BLOCK_NUM - 1);
    }
    atb::Tensor blockTables = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {NTOKENS, maxNumBlocksPerQuery});
    CHECK_STATUS(aclrtMemcpy(blockTables.deviceData,
        blockTables.dataSize,
        blockTablesData.data(),
        sizeof(int32_t) * blockTablesData.size(),
        ACL_MEMCPY_HOST_TO_DEVICE));
    
    // Create contextLens, host side tensor with random data

    // contextLensData need to persist after function return
    auto contextLensDataPtr = new std::vector<int32_t>(BATCH_SIZE, 0);
    std::vector<int32_t> &contextLensData = *contextLensDataPtr;
    for (size_t i = 0; i < contextLensData.size(); i++) {
        contextLensData.at(i) = rand() % (MAX_CONTEXT_LEN ) + 1;
    }
    // std::stable_sort(contextLensData.begin(), contextLensData.end());
    
    atb::Tensor contextLens = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE});
    contextLens.hostData = contextLensData.data();
    // CHECK_STATUS(aclrtMemcpy(contextLens.deviceData, contextLens.dataSize, contextLensData.data(), sizeof(int32_t) * contextLensData.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    
    // Create norm mask, upper triangular mask with -inf values
    std::vector<float> maskData(BATCH_SIZE * MAX_SEQ_LEN, 0);
    for (uint32_t i = 0; i < BATCH_SIZE; ++i) {
        // std::cout << "contextLensData[i]: " << contextLensData[i] << std::endl;
        for (int32_t j = contextLensData[i]; j < int32_t(BATCH_SIZE); ++j) {
            maskData[i * MAX_SEQ_LEN + j] = -32768.0f;  // 32768 : -inf
        }
    }
    atb::Tensor mask = CreateTensorFromVector(
        contextPtr, stream, maskData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, MAX_SEQ_LEN});
    
    // Put all input tensors into SVector in order
    atb::SVector<atb::Tensor> inTensors = {query, kCache, vCache, blockTables, contextLens, mask};
    return inTensors;
}

// Function to create custom PagedAttention operation
atb::Operation *PrepareCustomOperation()
{
    atb::customize::CustomPagedAttentionParam paOpParam;
    paOpParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    paOpParam.headNum = HEAD_NUM;
    paOpParam.kvHeadNum = KV_HEAD_NUM;
    paOpParam.qkScale = 0.08838834764831843;
    atb::Operation *paOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(paOpParam, &paOp));
    return paOp;
}

// Function to create standard PagedAttention operation
atb::Operation *PrepareStandardOperation()
{
    atb::infer::PagedAttentionParam paOpParam;
    paOpParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    paOpParam.headNum = HEAD_NUM;
    paOpParam.kvHeadNum = KV_HEAD_NUM;
    paOpParam.qkScale = 0.08838834764831843;
    atb::Operation *paOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(paOpParam, &paOp));
    return paOp;
}
/**
 * @brief 创建一个Gelu Activation的Operation，并设置参数
 * @return atb::Operation * 返回一个Operation指针
 */
 atb::Operation *CreateActivationOperation()
 {
     atb::infer::ActivationParam opParam;
     opParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
     atb::Operation *sigmoidOp = nullptr;
     CHECK_STATUS(atb::CreateOperation(opParam, &sigmoidOp));
     return sigmoidOp;
 }

 atb::Operation *CreateElewiseOperation()
{
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_COS;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(elewiseParam, &op));
    return op;
}

void parse_host_tiling_buffer(const uint32_t *hostTilingBuffer, uint64_t tilingBufferSize) {
    std::cout << "hostTilingBuffer.tilingBuffer: " << (void*)hostTilingBuffer << std::endl;
    std::cout << "hostTilingBuffer.tilingBufferSize: " << tilingBufferSize << std::endl;
    if (hostTilingBuffer == nullptr || tilingBufferSize == 0) {
        std::cout << "Invalid host tiling buffer!" << std::endl;
        return;
    }
    
    uint32_t tilingParamSize = tilingBufferSize / sizeof(uint32_t);
    std::cout << "Total tiling param elements: " << tilingParamSize << std::endl;
    
    // Parse header fields (TILING_HEAD_SIZE = 44)
    std::cout << "\n=== Tiling Header Fields ===" << std::endl;
    std::cout << "TILING_BATCH(tiling_head[0]): " << hostTilingBuffer[0] << std::endl;
    std::cout << "TILING_NUMHEADS(tiling_head[1]): " << hostTilingBuffer[1] << std::endl;
    std::cout << "TILING_HEADDIM(tiling_head[2]): " << hostTilingBuffer[2] << std::endl;
    std::cout << "TILING_NUMBLOKS(tiling_head[3]): " << hostTilingBuffer[3] << std::endl;
    std::cout << "TILING_BLOCKSIZE(tiling_head[4]): " << hostTilingBuffer[4] << std::endl;
    std::cout << "TILING_MAXBLOCKS(tiling_head[5]): " << hostTilingBuffer[5] << std::endl;
    std::cout << "TILING_TOR(tiling_head[6]): " << hostTilingBuffer[6] << std::endl;
    std::cout << "TILING_KVHEADS(tiling_head[7]): " << hostTilingBuffer[7] << std::endl;
    std::cout << "TILING_FORMER_BATCH(tiling_head[8]): " << hostTilingBuffer[8] << std::endl;
    std::cout << "TILING_FORMER_HEAD(tiling_head[9]): " << hostTilingBuffer[9] << std::endl;
    std::cout << "TILING_TAIL_BATCH(tiling_head[10]): " << hostTilingBuffer[10] << std::endl;
    std::cout << "TILING_TAIL_HEAD(tiling_head[11]): " << hostTilingBuffer[11] << std::endl;
    std::cout << "TILING_HEADNUM_MOVE(tiling_head[12]): " << hostTilingBuffer[12] << std::endl;
    std::cout << "TILING_MASK_MAX_LEN(tiling_head[13]): " << hostTilingBuffer[13] << std::endl;
    std::cout << "TILING_BATCH_STRIDE(tiling_head[14]): " << hostTilingBuffer[14] << std::endl;
    std::cout << "TILING_HEAD_STRIDE(tiling_head[15]): " << hostTilingBuffer[15] << std::endl;
    std::cout << "TILING_KEY(tiling_head[16]): " << hostTilingBuffer[16] << std::endl;
    std::cout << "TILING_HEADSIZE(tiling_head[17]): " << hostTilingBuffer[17] << std::endl;
    std::cout << "TILING_PARASIZE(tiling_head[18]): " << hostTilingBuffer[18] << std::endl;
    std::cout << "TILING_GROUPNUM(tiling_head[19]): " << hostTilingBuffer[19] << std::endl;
    std::cout << "TILING_FORMER_GROUP_MOVE(tiling_head[20]): " << hostTilingBuffer[20] << std::endl;
    std::cout << "TILING_TAIL_GROUP_MOVE(tiling_head[21]): " << hostTilingBuffer[21] << std::endl;
    std::cout << "TILING_MAX_KVSEQLEN(tiling_head[22]): " << hostTilingBuffer[22] << std::endl;
    std::cout << "TILING_KVSPLIT(tiling_head[23]): " << hostTilingBuffer[23] << std::endl;
    std::cout << "TILING_KVCORENUM(tiling_head[24]): " << hostTilingBuffer[24] << std::endl;
    std::cout << "TILING_BLOCKSIZE_CALC(tiling_head[25]): " << hostTilingBuffer[25] << std::endl;
    std::cout << "TILING_TOTAL_BLOCK_NUM(tiling_head[26]): " << hostTilingBuffer[26] << std::endl;
    std::cout << "TILING_PREFILL_BS(tiling_head[27]): " << hostTilingBuffer[27] << std::endl;
    std::cout << "TILING_DECODER_BS(tiling_head[28]): " << hostTilingBuffer[28] << std::endl;
    std::cout << "TILING_HEADDIM_V(tiling_head[29]): " << hostTilingBuffer[29] << std::endl;
    std::cout << "TILING_MODCOEF(tiling_head[30]): " << hostTilingBuffer[30] << std::endl;
    std::cout << "TILING_DIVCOEF(tiling_head[31]): " << hostTilingBuffer[31] << std::endl;
    std::cout << "TILING_QHEADORIGINAL(tiling_head[32]): " << hostTilingBuffer[32] << std::endl;
    std::cout << "TILING_COMPRESSHEAD(tiling_head[33]): " << hostTilingBuffer[33] << std::endl;
    std::cout << "TILING_QUANTYPE(tiling_head[34]): " << hostTilingBuffer[34] << std::endl;
    std::cout << "TILING_DATA_SHAPE_TYPE(tiling_head[35]): " << hostTilingBuffer[35] << std::endl;
    std::cout << "TILING_SCALETYPE(tiling_head[36]): " << hostTilingBuffer[36] << std::endl;
    std::cout << "TILING_MASK_TYPE_ND(tiling_head[37]): " << hostTilingBuffer[37] << std::endl;
    std::cout << "TILING_HEADDIM_K_SPLIT(tiling_head[38]): " << hostTilingBuffer[38] << std::endl;
    std::cout << "TILING_HEADDIM_V_SPLIT(tiling_head[39]): " << hostTilingBuffer[39] << std::endl;
    std::cout << "TILING_HEADDIM_V_SPLIT_VECTOR_FORMER(tiling_head[40]): " << hostTilingBuffer[40] << std::endl;
    std::cout << "TILING_HEADDIM_V_SPLIT_VECTOR_TAIL(tiling_head[41]): " << hostTilingBuffer[41] << std::endl;
    std::cout << "TILING_MTP_HEAD_SPLIT_SIZE(tiling_head[42]): " << hostTilingBuffer[42] << std::endl;
    std::cout << "TILING_MTP_HEAD_SPLIT_NUM(tiling_head[43]): " << hostTilingBuffer[43] << std::endl;

    // Parse batch parameters
    if (tilingParamSize > TILING_HEAD_SIZE) {
        uint32_t batchCount = hostTilingBuffer[0];
        std::cout << "\n=== Batch Parameters ===" << std::endl;
        std::cout << "Number of batches: " << batchCount << std::endl;
        batchCount = std::min(batchCount, 20u);
        
        for (uint32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
            uint32_t offset = TILING_HEAD_SIZE + batchIdx * TILING_PARA_SIZE;
            if (offset + TILING_PARA_SIZE <= tilingParamSize) {
                std::cout << "\n--- Batch " << batchIdx << " ---" << std::endl;
                std::cout << "  qSeqLen(batch_tiling_param[0]): " << hostTilingBuffer[offset + 0] << std::endl;
                std::cout << "  kvSeqLen(batch_tiling_param[1]): " << hostTilingBuffer[offset + 1] << std::endl;
                std::cout << "  qSBlockTile(batch_tiling_param[2]): " << hostTilingBuffer[offset + 2] << std::endl;
                std::cout << "  blockSize(batch_tiling_param[3]): " << hostTilingBuffer[offset + 3] << std::endl;
                std::cout << "  addrQSeqOffset[high](batch_tiling_param[4]): " << hostTilingBuffer[offset + 4] << std::endl;
                std::cout << "  addrQSeqOffset[low](batch_tiling_param[5]): " << hostTilingBuffer[offset + 5] << std::endl;
                std::cout << "  addrOSeqOffset[high](batch_tiling_param[6]): " << hostTilingBuffer[offset + 6] << std::endl;
                std::cout << "  addrOSeqOffset[low](batch_tiling_param[7]): " << hostTilingBuffer[offset + 7] << std::endl;
                std::cout << "  seqIdx(batch_tiling_param[8]): " << hostTilingBuffer[offset + 8] << std::endl;
                std::cout << "  totalQBlkNum(batch_tiling_param[9]): " << hostTilingBuffer[offset + 9] << std::endl;
                std::cout << "  maskOffset[high](batch_tiling_param[10]): " << hostTilingBuffer[offset + 10] << std::endl;
                std::cout << "  addrLSeqOffset[high](batch_tiling_param[11]): " << hostTilingBuffer[offset + 11] << std::endl;
                std::cout << "  addrLSeqOffset[low](batch_tiling_param[12]): " << hostTilingBuffer[offset + 12] << std::endl;
                std::cout << "  maskOffset[low](batch_tiling_param[14]): " << hostTilingBuffer[offset + 14] << std::endl;
                std::cout << "  addrOFdSeqOffset[high](batch_tiling_param[15]): " << hostTilingBuffer[offset + 15] << std::endl;
                std::cout << "  addrOFdSeqOffset[low](batch_tiling_param[16]): " << hostTilingBuffer[offset + 16] << std::endl;
            }
        }
    }
    
    std::cout << "\n=== End of Tiling Buffer Parse ===" << std::endl;
}

void parse_device_tiling_buffer(::atb::customize::TilingBufferInfo tiling_buffer_info) {
    std::cout << "=== Tiling Buffer Info ===" << std::endl;
    std::cout << "tiling_buffer_info.tilingBuffer: " << (void*)tiling_buffer_info.tilingBuffer << std::endl;
    std::cout << "tiling_buffer_info.tilingBufferSize: " << tiling_buffer_info.tilingBufferSize << std::endl;
    
    if (tiling_buffer_info.tilingBuffer == nullptr || tiling_buffer_info.tilingBufferSize == 0) {
        std::cout << "Invalid tiling buffer!" << std::endl;
        return;
    }
    
    // Allocate host memory for tiling buffer
    uint32_t *hostTilingBuffer = (uint32_t*)malloc(tiling_buffer_info.tilingBufferSize);
    if (hostTilingBuffer == nullptr) {
        std::cout << "Failed to allocate host memory for tiling buffer!" << std::endl;
        return;
    }
    
    // Copy from device to host (d2h)
    aclError ret = aclrtMemcpy(hostTilingBuffer, tiling_buffer_info.tilingBufferSize,
                               tiling_buffer_info.tilingBuffer, tiling_buffer_info.tilingBufferSize,
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to copy tiling buffer from device to host! Error: " << ret << std::endl;
        free(hostTilingBuffer);
        return;
    }
    
    // Parse host tiling buffer
    parse_host_tiling_buffer(hostTilingBuffer, tiling_buffer_info.tilingBufferSize);
    
    // Clean up
    free(hostTilingBuffer);
}

// Test fixture class for shared ACL initialization
class TestCustomPagedAttention : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize ACL for all tests in this test suite
        CHECK_STATUS(aclInit(nullptr));
        int32_t deviceId = 0;
        CHECK_STATUS(aclrtSetDevice(deviceId));
    }

    static void TearDownTestSuite() {
        // Finalize ACL after all tests in this test suite
        CHECK_STATUS(aclFinalize());
    }
};

TEST_F(TestCustomPagedAttention, StandardPagedAttentionTest)
{    
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // Prepare input tensors
    atb::SVector<atb::Tensor> inTensors = PrepareInTensor(context, stream);
    
    // Create standard PagedAttention operation
    atb::Operation *standardOp = PrepareStandardOperation();
    ASSERT_NE(standardOp, nullptr);

    // Prepare output tensor for standard operation
    atb::Tensor standardOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE});
    atb::VariantPack standardVariantPack;
    standardVariantPack.inTensors = inTensors;
    standardVariantPack.outTensors.push_back(standardOut);

    uint64_t standardWorkspaceSize = 0;
    CHECK_STATUS(standardOp->Setup(standardVariantPack, standardWorkspaceSize, context));
    uint8_t *standardWorkspacePtr = nullptr;
    if (standardWorkspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&standardWorkspacePtr), standardWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    
    // Execute standard PagedAttention operation
    CHECK_STATUS(standardOp->Execute(standardVariantPack, standardWorkspacePtr, standardWorkspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream));
    
    // Copy output tensors from device to host for verification
    CHECK_STATUS(CopyDeviceToHost(standardOut));
    // CHECK_STATUS(CopyDeviceToHost(outTensor));
    // std::cout << "standardOut.hostData: " << standardOut.hostData << std::endl;
    // std::cout << "outTensor.hostData: " << outTensor.hostData << std::endl;
    // for (int i = 0; i < 10 ; i++) {
    //     std::cout << "standardOut[" << i << "]: " << ((uint16_t*)standardOut.hostData)[i]  << std::endl;
    //     std::cout << "cast to float standardOut[" << i << "]: " << HalfToFloat(((uint16_t*)standardOut.hostData)[i])  << std::endl;
    // }
    
    CHECK_STATUS(aclrtFree(standardOut.deviceData));
    if (standardOut.hostData != nullptr) {
        free(standardOut.hostData);
    }
    for (atb::Tensor &inTensor : inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }

    if (standardWorkspaceSize > 0) {
        CHECK_STATUS(aclrtFree(standardWorkspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(standardOp));
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context));
    
}

TEST_F(TestCustomPagedAttention, CustomPagedAttentionTest)
{
    // Setup contexts and streams
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(context->SetExecuteType(atb::ExecuteType::EXECUTE_NORMAL));
    // CHECK_STATUS(context->SetLaunchMode(atb::LaunchMode::GRAPH_LAUNCH_MODE));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // prepare prelaunch context (only once)
    atb::Context *context_prelaunch = nullptr;
    CHECK_STATUS(atb::customize::CreatePlanContext(&context_prelaunch));
    atb::Context *context_execute = nullptr;
    CHECK_STATUS(atb::CreateContext(&context_execute));
    
    void *stream_prelaunch = nullptr;
    void *stream_execute = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream_prelaunch));
    CHECK_STATUS(aclrtCreateStream(&stream_execute));
    context_prelaunch->SetExecuteStream(stream_prelaunch);
    context_execute->SetExecuteStream(stream_execute);

    // Create operations (only once)
    atb::Operation *standardOp = PrepareStandardOperation();
    ASSERT_NE(standardOp, nullptr);
    atb::Operation *customOp_prelaunch = PrepareCustomOperation();
    ASSERT_NE(customOp_prelaunch, nullptr);
    atb::Operation *customOp_execute = PrepareCustomOperation();
    ASSERT_NE(customOp_execute, nullptr);
    
    // Create tiling tensor (only once, reused in each iteration)
    atb::Tensor tilingTensor = CreateTensor(ACL_UINT32, ACL_FORMAT_ND, {static_cast<int64_t>(1024*256)});
    
    uint64_t customWorkspaceSize = 0;
    uint8_t *customWorkspacePtr = nullptr;

    // Loop 10 times with different random data
    for (int testIter = 0; testIter < 10; testIter++) {
        std::cout << "\n========== Test Iteration " << (testIter + 1) << " / 10 ==========" << std::endl;
        unsigned int seed = testIter + 1;  // Use different seed for each iteration
        
        // Step 1: Prepare input tensors with random data (for standard operation)
        atb::SVector<atb::Tensor> inTensors = PrepareInTensor(context, stream, seed);
        
        // Step 2: Setup standard operation
        atb::Tensor standardOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE});
        atb::VariantPack standardVariantPack;
        standardVariantPack.inTensors = inTensors;
        standardVariantPack.outTensors.push_back(standardOut);
        
        uint64_t standardWorkspaceSize = 0;
        CHECK_STATUS(standardOp->Setup(standardVariantPack, standardWorkspaceSize, context));
        
        uint8_t *standardWorkspacePtr = nullptr;
        if (standardWorkspaceSize > 0) {
            CHECK_STATUS(aclrtMalloc((void **)(&standardWorkspacePtr), standardWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        // Step 3: Execute standard operation
        CHECK_STATUS(standardOp->Execute(standardVariantPack, standardWorkspacePtr, standardWorkspaceSize, context));
        CHECK_STATUS(aclrtSynchronizeStream(stream));
        
        // Step 4: Prepare prelaunch input tensors (reuse standard input + tilingTensor)
        // Note: We need to copy the input tensors for prelaunch since they're on different contexts
        // Add tilingTensor
        auto& prelaunch_inTensors = inTensors;
        prelaunch_inTensors.push_back(tilingTensor);
        
        // Step 5: Setup customOp_prelaunch with input + tilingTensor
        atb::VariantPack prelaunch_customVariantPack;
        prelaunch_customVariantPack.inTensors = prelaunch_inTensors;
        prelaunch_customVariantPack.outTensors.push_back(standardOut);
    
        CHECK_STATUS(aclrtSynchronizeStream(stream_prelaunch));
        CHECK_STATUS(context_prelaunch->SetLaunchMode(atb::LaunchMode::GRAPH_LAUNCH_MODE));
        CHECK_STATUS(customOp_prelaunch->Setup(prelaunch_customVariantPack, customWorkspaceSize, context_prelaunch));
        
        // Step 6: Get tiling buffer and copy to tilingTensor
        atb::customize::TilingBufferInfo host_tiling_buffer_info = atb::customize::GetHostTilingBufferFromCustomPagedAttentionOperation(customOp_prelaunch);
        // parse_host_tiling_buffer((uint32_t*)host_tiling_buffer_info.tilingBuffer, host_tiling_buffer_info.tilingBufferSize);
        CHECK_STATUS(aclrtMemcpy(tilingTensor.deviceData, tilingTensor.dataSize, 
                                 host_tiling_buffer_info.tilingBuffer, host_tiling_buffer_info.tilingBufferSize, 
                                 ACL_MEMCPY_HOST_TO_DEVICE));
        
        
        // Step 7: Setup customOp_execute with input + tilingTensor
        atb::SVector<atb::Tensor> custom_inTensors = PrepareInTensor(context_execute, stream_execute, seed);
        custom_inTensors.push_back(tilingTensor);
        
        atb::Tensor customOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE});
        atb::VariantPack customVariantPack_execute;
        customVariantPack_execute.inTensors = custom_inTensors;
        customVariantPack_execute.outTensors.push_back(customOut);
        
        CHECK_STATUS(customOp_execute->Setup(customVariantPack_execute, customWorkspaceSize, context_execute));
        
        // Allocate workspace if needed
        
        if (customWorkspaceSize > 0 ) {
            CHECK_STATUS(aclrtMalloc((void **)(&customWorkspacePtr), customWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        
        // Step 8: Execute custom operation
        CHECK_STATUS(customOp_execute->Execute(customVariantPack_execute, customWorkspacePtr, customWorkspaceSize, context_execute));
        CHECK_STATUS(aclrtSynchronizeStream(stream_execute));
        
        // Step 9: Copy output tensors from device to host for verification
        CHECK_STATUS(CopyDeviceToHost(standardOut));
    CHECK_STATUS(CopyDeviceToHost(customOut));
    
        // Step 10: Compare results
    atb::Status compareResult = CompareTensorHostData(customOut, standardOut);
        EXPECT_EQ(compareResult, 0) << "Iteration " << (testIter + 1) << ": Custom vs Standard PagedAttention comparison failed!";
    if (compareResult == 0) {
            std::cout << "Iteration " << (testIter + 1) << ": Comparison passed!" << std::endl;
        } else {
            std::cout << "Iteration " << (testIter + 1) << ": Comparison failed!" << std::endl;
        }
        
        // Clean up iteration resources
        CHECK_STATUS(aclrtFree(standardOut.deviceData));
        if (standardOut.hostData != nullptr) {
            free(standardOut.hostData);
        }
    CHECK_STATUS(aclrtFree(customOut.deviceData));
    if (customOut.hostData != nullptr) {
        free(customOut.hostData);
    }
        for (atb::Tensor &inTensor : inTensors) {
            if (inTensor.deviceData == tilingTensor.deviceData) { 
                continue;
            }
            CHECK_STATUS(aclrtFree(inTensor.deviceData));
        }
        for (atb::Tensor &inTensor : custom_inTensors) {
            if (inTensor.deviceData == tilingTensor.deviceData) { 
                continue;
            }
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
        if (standardWorkspaceSize > 0 && standardWorkspacePtr != nullptr) {
        CHECK_STATUS(aclrtFree(standardWorkspacePtr));
        }
        if (customWorkspaceSize > 0 && customWorkspacePtr != nullptr) {
            CHECK_STATUS(aclrtFree(customWorkspacePtr));
        }
    }
    
    // Clean up shared resources
    CHECK_STATUS(aclrtFree(tilingTensor.deviceData));
    CHECK_STATUS(atb::DestroyOperation(standardOp));
    CHECK_STATUS(atb::DestroyOperation(customOp_prelaunch));
    CHECK_STATUS(atb::DestroyOperation(customOp_execute));
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(aclrtDestroyStream(stream_prelaunch));
    CHECK_STATUS(aclrtDestroyStream(stream_execute));
    CHECK_STATUS(DestroyContext(context));
    CHECK_STATUS(DestroyContext(context_prelaunch));
    CHECK_STATUS(DestroyContext(context_execute));
    
    std::cout << "\n========== All 10 test iterations completed successfully! ==========" << std::endl;
}

atb::Tensor PrepareQueryTensor(atb::Context *contextPtr, aclrtStream stream, const uint32_t num_tokens, unsigned int seed = 0)
{
    srand(seed);
    // Create query tensor with random data
    std::vector<float> queryData(num_tokens * HEAD_NUM * HEAD_SIZE);
    for (size_t i = 0; i < queryData.size(); i++) {
        queryData[i] = (rand() % 2000 - 1000) / 1000.0f;  // Random values between -1.0 and 1.0
    }
    atb::Tensor query = CreateTensorFromVector(
        contextPtr, stream, queryData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {num_tokens, HEAD_NUM, HEAD_SIZE});
    return query;
}

atb::Tensor PrepareKCacheTensor(atb::Context *contextPtr, aclrtStream stream, unsigned int seed = 0)
{
    srand(seed);
    // Create kCache tensor with random data
    std::vector<float> kCacheData(BLOCK_NUM * BLOCK_SIZE * KV_HEAD_NUM * HEAD_SIZE);
    for (size_t i = 0; i < kCacheData.size(); i++) {
        kCacheData[i] = (rand() % 2000 - 1000) / 1000.0f;  // Random values between -1.0 and 1.0   
    }
    atb::Tensor kCache = CreateTensorFromVector(contextPtr, stream, kCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BLOCK_NUM, BLOCK_SIZE, KV_HEAD_NUM, HEAD_SIZE});
    return kCache;
}

std::tuple<atb::Tensor,atb::Tensor,atb::Tensor> PrepareBlockTablesContextLensMaskTensor(atb::Context *contextPtr, aclrtStream stream, const uint32_t batch_size, uint32_t max_context_len, unsigned int seed = 0)
{
    srand(seed);
    // Create contextLens tensor with random data
    std::vector<int32_t>* contextLensDataPtr = new std::vector<int32_t>(batch_size, 0);
    std::vector<int32_t> &contextLensData = *contextLensDataPtr;
    for (size_t i = 0; i < contextLensData.size(); i++) {
        contextLensData[i] = rand() % (max_context_len ) + 1;
    }
    // std::cout << "contextLensData.size(): " << contextLensData.size() << " [";
    // for (size_t i = 0; i < contextLensData.size(); i++) {
    //     std::cout << contextLensData[i];
    //     if (i < contextLensData.size() - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;
    atb::Tensor contextLens = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {batch_size});
    contextLens.hostData = contextLensData.data();

    // Create blockTables
    uint32_t maxNumBlocksPerQuery = (max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int32_t> blockTablesData(batch_size * maxNumBlocksPerQuery, 0);
    for (size_t i = 0; i < blockTablesData.size(); i++) {
        blockTablesData.at(i) = rand() % (BLOCK_NUM - 1);
    }
    // std::cout << "blockTablesData.size(): " << blockTablesData.size() << std::endl;

    atb::Tensor blockTables = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {batch_size, maxNumBlocksPerQuery});
    CHECK_STATUS(aclrtMemcpy(blockTables.deviceData,
        blockTables.dataSize,
        blockTablesData.data(),
        sizeof(int32_t) * blockTablesData.size(),
        ACL_MEMCPY_HOST_TO_DEVICE));  

    // Create mask tensor 
    std::vector<float> maskData(batch_size * max_context_len, 0.0);
    for (size_t i = 0; i < batch_size; i++) {
        for (int32_t j = 0; j < contextLensData[i]; j++) {
            maskData[i * max_context_len + j] = -32768.0f;  // 32768 : -inf
        }
    }
    atb::Tensor mask = CreateTensorFromVector(contextPtr, stream, maskData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {batch_size, max_context_len});
    return {contextLens, blockTables, mask};
}

atb::Tensor PrepareTilingTensor()
{
    atb::Tensor tilingTensor = CreateTensor(ACL_UINT32, ACL_FORMAT_ND, {static_cast<int64_t>(1024*256)});
    return tilingTensor;
}
/**
 *
 * 类似pytorch_npu中的 op_plugin::npu_stride_copy_out
 * acl原生aclrtMemcpy2d不支持d2d拷贝
 */
void async_copy_2dtensor(const atb::Tensor &src_tensor, atb::Tensor &dst_tensor,  aclrtStream stream)
{
    assert(src_tensor.desc.shape.dimNum == 2);
    assert(dst_tensor.desc.shape.dimNum == 2);
    assert(src_tensor.desc.shape.dims[0] == dst_tensor.desc.shape.dims[0]);
    assert(src_tensor.desc.shape.dims[1] == dst_tensor.desc.shape.dims[1]);
    
    // 如果 src 和 dst 都是连续的，且 stride 相同，可以直接使用 aclrtMemcpyAsync
    int64_t src_stride = src_tensor.desc.shape.dims[1];
    int64_t dst_stride = dst_tensor.desc.shape.dims[1];
    
    if (src_stride == dst_stride) {
        // 连续内存，直接拷贝
        size_t copySize = src_tensor.desc.shape.dims[0] * src_tensor.desc.shape.dims[1] * 
                         (src_tensor.desc.dtype == ACL_FLOAT16 ? 2 : 4);
        CHECK_STATUS(aclrtMemcpyAsync(dst_tensor.deviceData, copySize, 
                                     src_tensor.deviceData, copySize,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        return;
    }

    size_t elementSize = (src_tensor.desc.dtype == ACL_FLOAT16 || src_tensor.desc.dtype == ACL_BF16) ? 2 : 4;
    int64_t height = src_tensor.desc.shape.dims[0];
    int64_t width = src_tensor.desc.shape.dims[1];
    size_t rowSize = width * elementSize;
    // copy row by row
    for (int64_t i = 0; i < height; i++) {
        void *srcPtr = (char*)src_tensor.deviceData + i * src_stride * elementSize;
        void *dstPtr = (char*)dst_tensor.deviceData + i * dst_stride * elementSize;
        CHECK_STATUS(aclrtMemcpyAsync(dstPtr, rowSize, srcPtr, rowSize,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
}

void async_copy_1dtensor(const atb::Tensor &src_tensor, atb::Tensor &dst_tensor, aclrtStream stream)
{
    CHECK_STATUS(aclrtMemcpyAsync(dst_tensor.deviceData, dst_tensor.dataSize, src_tensor.deviceData, src_tensor.dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
}

TEST_F(TestCustomPagedAttention, GraphReplayWithTilingTensor)
{
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);
    
    atb::Tensor tilingTensor = PrepareTilingTensor();
    atb::Tensor kCache = PrepareKCacheTensor(context, stream);
    atb::Tensor vCache = kCache;

    atb::Tensor pesistent_query = PrepareQueryTensor(context, stream, NTOKENS);
    auto [pesistent_contextLens, pesistent_blockTables, pesistent_mask] = PrepareBlockTablesContextLensMaskTensor(context, stream, NTOKENS, MAX_SEQ_LEN);
    
    atb::Operation *standardOp = PrepareStandardOperation();

    atb::Tensor standardOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE});
    
    atb::Context *context_prelaunch = nullptr;
    CHECK_STATUS(atb::customize::CreatePlanContext(&context_prelaunch));
    atb::Context *context_execute = nullptr;
    CHECK_STATUS(atb::CreateContext(&context_execute));
    
    void *stream_prelaunch = nullptr;
    void *stream_execute = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream_prelaunch));
    context_prelaunch->SetExecuteStream(stream_prelaunch);
    CHECK_STATUS(aclrtCreateStream(&stream_execute));
    context_execute->SetExecuteStream(stream_execute);

    atb::Operation *customOp_prelaunch = PrepareCustomOperation();
    ASSERT_NE(customOp_prelaunch, nullptr);
    atb::Operation *customOp_execute = PrepareCustomOperation();
    ASSERT_NE(customOp_execute, nullptr);
    
    atb::Tensor customOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE});
    
    atb::VariantPack custom_variantPack;
    custom_variantPack.inTensors = {pesistent_query, kCache, vCache, pesistent_blockTables, pesistent_contextLens, pesistent_mask, tilingTensor};
    custom_variantPack.outTensors.push_back(customOut);
    uint64_t customWorkspaceSize = 0;
    CHECK_STATUS(aclmdlRICaptureBegin(stream_execute, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    CHECK_STATUS(customOp_execute->Setup(custom_variantPack, customWorkspaceSize, context_execute));
    uint8_t *customWorkspacePtr = nullptr;
    if (customWorkspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&customWorkspacePtr), customWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_STATUS(customOp_execute->Execute(custom_variantPack, customWorkspacePtr, customWorkspaceSize, context_execute));
    // ========结束捕获任务========
    aclmdlRI modelRI;
    CHECK_STATUS(aclmdlRICaptureEnd(stream_execute, &modelRI));

    // std::vector<uint32_t> max_context_lens{2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000};
    std::vector<uint32_t> max_context_lens{20000, 18000, 16000, 14000, 12000, 10000, 8000, 6000, 4000, 2000};
    
    for (int i = 0; i < 10; i++) {
        unsigned int rand_seed = i+1;
        atb::Tensor query = PrepareQueryTensor(context, stream, NTOKENS, rand_seed);
        auto [contextLens, blockTables, mask] = PrepareBlockTablesContextLensMaskTensor(context, stream, NTOKENS, max_context_lens[i], rand_seed);
        atb::SVector<atb::Tensor> standard_inTensors = {query, kCache, vCache, blockTables, contextLens, mask};
        atb::VariantPack standard_variantPack;
        standard_variantPack.inTensors = standard_inTensors;
        standard_variantPack.outTensors.push_back(standardOut);

        uint64_t standardWorkspaceSize = 0;
        CHECK_STATUS(standardOp->Setup(standard_variantPack, standardWorkspaceSize, context));
        
        uint8_t *standardWorkspacePtr = nullptr;
        if (standardWorkspaceSize > 0) {
            CHECK_STATUS(aclrtMalloc((void **)(&standardWorkspacePtr), standardWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        CHECK_STATUS(standardOp->Execute(standard_variantPack, standardWorkspacePtr, standardWorkspaceSize, context));
        CHECK_STATUS(aclrtSynchronizeStream(stream));
        // CHECK_STATUS(CopyDeviceToHost(standardOut));

        atb::SVector<atb::Tensor> custom_inTensors = {query, kCache, vCache, pesistent_blockTables, contextLens, pesistent_mask, tilingTensor};
        atb::VariantPack custom_variantPack;
        custom_variantPack.inTensors = custom_inTensors;
        custom_variantPack.outTensors.push_back(customOut);

        // CHECK_STATUS(aclrtSynchronizeStream(stream_prelaunch));
        CHECK_STATUS(context_prelaunch->SetLaunchMode(atb::LaunchMode::GRAPH_LAUNCH_MODE));
        
        CHECK_STATUS(customOp_prelaunch->Setup(custom_variantPack, customWorkspaceSize, context_prelaunch));
        auto tiling_host_buffer = atb::customize::GetHostTilingBufferFromCustomPagedAttentionOperation(customOp_prelaunch);

        CHECK_STATUS(aclrtMemcpyAsync(tilingTensor.deviceData, tilingTensor.dataSize, 
            tiling_host_buffer.tilingBuffer, tiling_host_buffer.tilingBufferSize, 
                                ACL_MEMCPY_HOST_TO_DEVICE, stream_execute));
        
        async_copy_1dtensor(query, pesistent_query, stream_execute);
        async_copy_2dtensor(blockTables, pesistent_blockTables, stream_execute);
        async_copy_1dtensor(contextLens, pesistent_contextLens, stream_execute);
        async_copy_2dtensor(mask, pesistent_mask, stream_execute);
        CHECK_STATUS(aclmdlRIExecuteAsync(modelRI, stream_execute));
        CHECK_STATUS(aclrtSynchronizeStream(stream_execute));
        CHECK_STATUS(CopyDeviceToHost(customOut));
        CHECK_STATUS(CopyDeviceToHost(standardOut));
        atb::Status compareResult = CompareTensorHostData(customOut, standardOut);
        EXPECT_EQ(compareResult, 0) << "Iteration " << (i + 1) << ": Custom vs Standard PagedAttention comparison failed!";
    }

}