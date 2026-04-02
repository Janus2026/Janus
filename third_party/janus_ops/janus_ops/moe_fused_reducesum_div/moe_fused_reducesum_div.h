/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_reducesum_div.h
 * \brief
 */
#include "kernel_operator.h"

constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t SINGLE_CONST_TWO = 2;
using namespace AscendC;

namespace kernels {
    template <typename INPUT_TP>
    class MoeFusedReducesumDiv {
    public:
        __aicore__ MoeFusedReducesumDiv() {};
    
        __aicore__ inline void init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const MoeFusedReducesumDivTilingData &tiling) {
            this->tiling = tiling;
            int32_t core_id = GetBlockIdx();

            n = tiling.n;
            n_ailgn = tiling.nAlign;
            big_core_num = tiling.bigCoreNum;
            little_core_num = tiling.littleCoreNum;
            avg_core_calc_rows = tiling.avgCoreCalcRows;
            single_core_calc_rows = tiling.singleCoreCalcRows;

            single_core_calc_rows /= BUFFER_NUM;

            if (big_core_num == 0) {
                core_offset = core_id * avg_core_calc_rows;
                core_calc_rows = avg_core_calc_rows;
            } else if (core_id < big_core_num && big_core_num != 0) {
                core_offset = core_id * (avg_core_calc_rows + 1);
                core_calc_rows = avg_core_calc_rows + 1;
            } else {
                core_offset = big_core_num * (avg_core_calc_rows + 1) + (core_id - big_core_num) * avg_core_calc_rows;
                core_calc_rows = avg_core_calc_rows;
            }
            tail = core_calc_rows % single_core_calc_rows;
            loop = core_calc_rows / single_core_calc_rows;

            

            inputGM.SetGlobalBuffer((__gm__ INPUT_TP*)input);
            outputGM.SetGlobalBuffer((__gm__ INPUT_TP*)output);

            pipe.InitBuffer(inputQUE, BUFFER_NUM, single_core_calc_rows * n_ailgn * sizeof(float));
            pipe.InitBuffer(outputQUE, BUFFER_NUM, single_core_calc_rows * n_ailgn * sizeof(float));
            pipe.InitBuffer(sumBUF, single_core_calc_rows / BUFFER_NUM * sizeof(float));
        }
    
        __aicore__ inline void process() {
            int32_t core_id = GetBlockIdx();
            loop *= BUFFER_NUM;
            for (int32_t block_id = 0; block_id < loop; block_id++){
                processOne(block_id, single_core_calc_rows);
            }
            if (tail != 0){
                processOne(loop, tail);
            }
        }

        __aicore__ inline void processOne(int32_t block_id, int32_t calc_rows) {
	        LocalTensor<float> inputLocal = inputQUE.AllocTensor<float>();
            // load
            DataCopyExtParams dataCopyParams;
            DataCopyPadExtParams<INPUT_TP> padParams;
            dataCopyParams.blockCount = calc_rows;
            dataCopyParams.blockLen = n * sizeof(INPUT_TP);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = n_ailgn - n;
            SetPadValue(0);

            DataCopyPad(
                inputLocal.ReinterpretCast<INPUT_TP>()[(sizeof(INPUT_TP) == SINGLE_CONST_TWO) * calc_rows * n_ailgn], inputGM[(core_offset + block_id * single_core_calc_rows) * n], dataCopyParams,
                padParams);

            inputQUE.EnQue(inputLocal);
            inputLocal = inputQUE.DeQue<float>();

            // cast
            if (sizeof(INPUT_TP) == SINGLE_CONST_TWO) {
                Cast(inputLocal, inputLocal.ReinterpretCast<INPUT_TP>()[calc_rows * n_ailgn], RoundMode::CAST_NONE, calc_rows * n_ailgn);
            }
            
            inputQUE.EnQue(inputLocal);
            // compute
            inputLocal = inputQUE.DeQue<float>();
            LocalTensor<float> sumLocal = sumBUF.Get<float>();
            LocalTensor<float> outputLocal = outputQUE.AllocTensor<float>();

            for(int32_t i=0; i<calc_rows; i++){
                ReduceSum(sumLocal[i], inputLocal[i * n_ailgn], outputLocal[i * n_ailgn], n);
                Muls<float>(outputLocal[i * n_ailgn], inputLocal[i * n_ailgn], (float)1.0/sumLocal.GetValue(i), n);
            }

            outputQUE.EnQue(outputLocal);
            outputLocal = outputQUE.DeQue<float>();
            if (sizeof(INPUT_TP) == SINGLE_CONST_TWO) {
                Cast(outputLocal.ReinterpretCast<INPUT_TP>()[0], outputLocal, RoundMode::CAST_RINT, calc_rows * n_ailgn);
                PipeBarrier<PIPE_V>();
            }

            DataCopyExtParams dataCopyParamsOUT;
            dataCopyParamsOUT.blockCount = calc_rows;
            dataCopyParamsOUT.blockLen = n * sizeof(INPUT_TP);
            dataCopyParamsOUT.srcStride = n_ailgn - n;
            dataCopyParamsOUT.dstStride = 0;
            DataCopyPad(
                outputGM[(core_offset + block_id * single_core_calc_rows) * n], outputLocal.ReinterpretCast<INPUT_TP>()[0], dataCopyParams);
            inputQUE.FreeTensor(inputLocal);
            outputQUE.FreeTensor(outputLocal);
        }
    
    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inputQUE;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outputQUE;
        TBuf<TPosition::VECCALC> sumBUF;
        
        GlobalTensor<INPUT_TP> inputGM;
        GlobalTensor<INPUT_TP> outputGM;
    
        MoeFusedReducesumDivTilingData tiling;

        int32_t n;
        int32_t n_ailgn;
        int32_t big_core_num;
        int32_t little_core_num;
        int32_t avg_core_calc_rows;
        int32_t single_core_calc_rows;

        int32_t core_offset;
        int32_t core_calc_rows;
        int32_t tail;
        int32_t loop; 
    };
}
