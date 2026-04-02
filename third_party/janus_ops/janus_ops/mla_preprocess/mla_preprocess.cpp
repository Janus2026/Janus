/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mal_preprocess.cpp
 * \brief
 */
#include "kernel_operator.h"

#if ORIG_DTYPE_INPUT == DT_FLOAT16
#include "mla_preprocess_fp16.h"
#elif ORIG_DTYPE_INPUT == DT_BF16
#include "mla_preprocess_bf16.h"
#include "mla_preprocess_no_quant.h"
#endif

using namespace MlaPreprocess;

extern "C" __global__ __aicore__ void
mla_preprocess(GM_ADDR hiddenStateGm, GM_ADDR gamma1Gm, GM_ADDR beta1Gm, GM_ADDR quantScale1Gm, GM_ADDR quantOffset1Gm,
               GM_ADDR wdqkvGm, GM_ADDR descale1Gm, GM_ADDR bias1Gm, GM_ADDR gamma2Gm, GM_ADDR beta2Gm,
               GM_ADDR quantScale2Gm, GM_ADDR quantOffset2Gm, GM_ADDR wuqGm, GM_ADDR descale2Gm, GM_ADDR bias2Gm,
               GM_ADDR gamma3Gm, GM_ADDR cos1Gm, GM_ADDR sin1Gm, GM_ADDR wukGm, GM_ADDR keycacheGm,
               GM_ADDR keycacheRopeGm, GM_ADDR slotMappingGm, GM_ADDR gmCtkvScale, GM_ADDR gmQnopeScale, GM_ADDR qGm,
               GM_ADDR keycacheOutGm, GM_ADDR qGm2, GM_ADDR keycacheOutGm2, GM_ADDR workspace, GM_ADDR tiling)
{
    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_CUBE__
    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
#endif
    GET_TILING_DATA(tilingData, tiling);
    auto s1Gm = AscendC::GetUserWorkspace(workspace);
    auto s2Gm = s1Gm + tilingData.maxWorkspaceSize;
    auto s3Gm = s2Gm + tilingData.maxWorkspaceSize;

#if ORIG_DTYPE_INPUT == DT_FLOAT16
    if (TILING_KEY_IS(0)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::ND, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(4)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::ND, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(8)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::ND, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(12)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::ND, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(16)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(20)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(24)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(28)) {
        MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(32)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::ND, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(36)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::ND, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(40)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::ND, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(44)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::ND, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(48)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(52)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(56)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(60)) {
        MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(64)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::ND, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(68)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::ND, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(72)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::ND, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(76)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::ND, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(80)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::NZ, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(84)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::NZ, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(88)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(92)) {
        MLAOperation<CACHE_MODE_INT8_NZCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(96)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::ND, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(100)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::ND, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(104)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::ND, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(108)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::ND, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(112)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::NZ, DataFormat::ND, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(116)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::NZ, DataFormat::ND, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(120)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(124)) {
        MLAOperation<CACHE_MODE_NZCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    }

#elif ORIG_DTYPE_INPUT == DT_BF16
    PRELOAD(2);
    auto s4Gm = s3Gm + tilingData.maxWorkspaceSize;
    auto s5Gm = s4Gm + tilingData.maxWorkspaceSize;
    // InDtype[1] cacheMode[2] weightFormat1[1] weightFormat2[1] weightFormat3[1] quantMode[2]
    if (TILING_KEY_IS(1)) { // fp16_cm0_nd_nd_nd_qm1
        MLAOperation<half, 0, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(5)) { // fp16_cm0_nd_nd_nz_qm1
        MLAOperation<half, 0, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(9)) { // fp16_cm0_nd_nz_nd_qm1
        MLAOperation<half, 0, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(13)) { // fp16_cm0_nd_nz_nz_qm1
        MLAOperation<half, 0, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(17)) { // fp16_cm0_nz_nd_nd_qm1
        MLAOperation<half, 0, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(21)) { // fp16_cm0_nz_nd_nz_qm1
        MLAOperation<half, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(25)) { // fp16_cm0_nz_nz_nd_qm1
        MLAOperation<half, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(29)) { // fp16_cm0_nz_nz_nz_qm1
        MLAOperation<half, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(33)) { // fp16_cm1_nd_nd_nd_qm1
        MLAOperation<half, 1, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(37)) { // fp16_cm1_nd_nd_nz_qm1
        MLAOperation<half, 1, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(41)) { // fp16_cm1_nd_nz_nd_qm1
        MLAOperation<half, 1, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(45)) { // fp16_cm1_nd_nz_nz_qm1
        MLAOperation<half, 1, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(49)) { // fp16_cm1_nz_nd_nd_qm1
        MLAOperation<half, 1, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(53)) { // fp16_cm1_nz_nd_nz_qm1
        MLAOperation<half, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(57)) { // fp16_cm1_nz_nz_nd_qm1
        MLAOperation<half, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(61)) { // fp16_cm1_nz_nz_nz_qm1
        MLAOperation<half, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(65)) { // fp16_cm2_nd_nd_nd_qm1
        MLAOperation<half, 2, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(69)) { // fp16_cm2_nd_nd_nz_qm1
        MLAOperation<half, 2, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(73)) { // fp16_cm2_nd_nz_nd_qm1
        MLAOperation<half, 2, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(77)) { // fp16_cm2_nd_nz_nz_qm1
        MLAOperation<half, 2, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(81)) { // fp16_cm2_nz_nd_nd_qm1
        MLAOperation<half, 2, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(85)) { // fp16_cm2_nz_nd_nz_qm1
        MLAOperation<half, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(89)) { // fp16_cm2_nz_nz_nd_qm1
        MLAOperation<half, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(93)) { // fp16_cm2_nz_nz_nz_qm1
        MLAOperation<half, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(97)) { // fp16_cm3_nd_nd_nd_qm1
        MLAOperation<half, 3, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(101)) { // fp16_cm3_nd_nd_nz_qm1
        MLAOperation<half, 3, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(105)) { // fp16_cm3_nd_nz_nd_qm1
        MLAOperation<half, 3, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(109)) { // fp16_cm3_nd_nz_nz_qm1
        MLAOperation<half, 3, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(113)) { // fp16_cm3_nz_nd_nd_qm1
        MLAOperation<half, 3, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(117)) { // fp16_cm3_nz_nd_nz_qm1
        MLAOperation<half, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(121)) { // fp16_cm3_nz_nz_nd_qm1
        MLAOperation<half, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(125)) { // fp16_cm3_nz_nz_nz_qm1
        MLAOperation<half, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(128)) { // bf16_cm0_nd_nd_nd_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(129)) { // bf16_cm0_nd_nd_nd_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(132)) { // bf16_cm0_nd_nd_nz_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(133)) { // bf16_cm0_nd_nd_nz_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(136)) { // bf16_cm0_nd_nz_nd_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(137)) { // bf16_cm0_nd_nz_nd_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(140)) { // bf16_cm0_nd_nz_nz_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(141)) { // bf16_cm0_nd_nz_nz_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(144)) { // bf16_cm0_nz_nd_nd_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(145)) { // bf16_cm0_nz_nd_nd_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(148)) { // bf16_cm0_nz_nd_nz_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(149)) { // bf16_cm0_nz_nd_nz_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(152)) { // bf16_cm0_nz_nz_nd_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(153)) { // bf16_cm0_nz_nz_nd_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(156)) { // bf16_cm0_nz_nz_nz_qm0
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(157)) { // bf16_cm0_nz_nz_nz_qm1
        MLAOperation<bfloat16_t, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(160)) { // bf16_cm1_nd_nd_nd_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(161)) { // bf16_cm1_nd_nd_nd_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(164)) { // bf16_cm1_nd_nd_nz_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(165)) { // bf16_cm1_nd_nd_nz_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(168)) { // bf16_cm1_nd_nz_nd_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(169)) { // bf16_cm1_nd_nz_nd_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(172)) { // bf16_cm1_nd_nz_nz_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(173)) { // bf16_cm1_nd_nz_nz_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(176)) { // bf16_cm1_nz_nd_nd_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(177)) { // bf16_cm1_nz_nd_nd_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(180)) { // bf16_cm1_nz_nd_nz_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(181)) { // bf16_cm1_nz_nd_nz_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(184)) { // bf16_cm1_nz_nz_nd_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(185)) { // bf16_cm1_nz_nz_nd_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(188)) { // bf16_cm1_nz_nz_nz_qm0
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(189)) { // bf16_cm1_nz_nz_nz_qm1
        MLAOperation<bfloat16_t, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(192)) { // bf16_cm2_nd_nd_nd_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(193)) { // bf16_cm2_nd_nd_nd_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(196)) { // bf16_cm2_nd_nd_nz_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(197)) { // bf16_cm2_nd_nd_nz_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(200)) { // bf16_cm2_nd_nz_nd_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(201)) { // bf16_cm2_nd_nz_nd_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(204)) { // bf16_cm2_nd_nz_nz_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(205)) { // bf16_cm2_nd_nz_nz_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(208)) { // bf16_cm2_nz_nd_nd_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(209)) { // bf16_cm2_nz_nd_nd_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(212)) { // bf16_cm2_nz_nd_nz_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(213)) { // bf16_cm2_nz_nd_nz_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(216)) { // bf16_cm2_nz_nz_nd_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(217)) { // bf16_cm2_nz_nz_nd_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(220)) { // bf16_cm2_nz_nz_nz_qm0
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(221)) { // bf16_cm2_nz_nz_nz_qm1
        MLAOperation<bfloat16_t, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(224)) { // bf16_cm3_nd_nd_nd_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(225)) { // bf16_cm3_nd_nd_nd_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(228)) { // bf16_cm3_nd_nd_nz_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(229)) { // bf16_cm3_nd_nd_nz_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(232)) { // bf16_cm3_nd_nz_nd_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(233)) { // bf16_cm3_nd_nz_nd_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(236)) { // bf16_cm3_nd_nz_nz_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(237)) { // bf16_cm3_nd_nz_nz_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::ND, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(240)) { // bf16_cm3_nz_nd_nd_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(241)) { // bf16_cm3_nz_nd_nd_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::ND, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(244)) { // bf16_cm3_nz_nd_nz_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::ND, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(245)) { // bf16_cm3_nz_nd_nz_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(248)) { // bf16_cm3_nz_nz_nd_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(249)) { // bf16_cm3_nz_nz_nd_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(252)) { // bf16_cm3_nz_nz_nz_qm0
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TENSOR_ASYMM_QUANT>
            op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(253)) { // bf16_cm3_nz_nz_nz_qm1
        MLAOperation<bfloat16_t, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ, QuantMode::PER_TOKEN_SYMM_QUANT> op(
            tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, s4Gm, s5Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(1184)) {
        MLAOperation_no_quant<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(1152)) {
        MLAOperation_no_quant<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(1188)) {
        MLAOperation_no_quant<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    } else if (TILING_KEY_IS(1156)) {
        MLAOperation_no_quant<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::NZ> op(tilingData);
        op.Init(hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm, bias1Gm, gamma2Gm, beta2Gm,
                quantScale2Gm, quantOffset2Gm, gamma3Gm, sin1Gm, cos1Gm, sin1Gm, cos1Gm, keycacheGm, slotMappingGm,
                wuqGm, bias2Gm, wukGm, descale1Gm, descale2Gm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2,
                keycacheOutGm2, s1Gm, s2Gm, s3Gm, nullptr);
        op.ProcessCube();
        op.ProcessVector();
    }
#endif
}