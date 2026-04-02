/* Copyright 2025 The Janus Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#undef __MIX_CORE_MACRO__

constexpr auto L1_size = 512 * 1024; // 512KB
constexpr auto L1_size_half = 256 * 1024;
constexpr auto L1_size_half_pp = 128 * 1024;
constexpr auto L0A_size = 64 * 1024; // 64KB
constexpr auto L0A_size_pp = 32 * 1024;
constexpr auto L0B_size = 64 * 1024; // 64KB
constexpr auto L0B_size_pp = 32 * 1024;
constexpr auto L0C_size = 128 * 1024; // 128KB
constexpr auto L0C_size_pp = 64 * 1024;
constexpr auto CONST_16 = 16;
constexpr auto CONST_256 = 16 * 16;

constexpr auto k0 = 256;
constexpr auto n0 = 256;
constexpr auto k_part = 64;

using namespace AscendC;
using bf16 = bfloat16_t;

__aicore__ inline uint32_t
CalculateGlobalBOffset(uint32_t row_idx, uint32_t col_idx, uint32_t k) {
  return row_idx * k0 + col_idx * n0 * k;
}

class KernelPPMatmulOpt {
public:
  __aicore__ inline KernelPPMatmulOpt() {}

  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                              GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(_tiling_data, tiling);

    _m = _tiling_data.m;
    _k = _tiling_data.k;
    _n = _tiling_data.n;
    _rows = _tiling_data.rows;
    _core_idx = AscendC::GetBlockIdx();

    _aGm.SetGlobalBuffer((__gm__ bf16 *)a, _m * _k);
    _bGm.SetGlobalBuffer((__gm__ bf16 *)b, _k * _n);
    _cGm.SetGlobalBuffer((__gm__ float *)workspace, _m * _n);

    _start_idx = _core_idx < 20 ? (_core_idx * 42) : (_core_idx * 40 + 40);
    _end_idx = _core_idx < 20 ? _start_idx + 42 : _start_idx + 40;
    uint32_t row_idx_start = _start_idx % _rows;
    uint32_t col_idx_start = _start_idx / _rows;

    // Copy initial data to L1 as early as possible
    _pipe.InitBuffer(_inQueueA1, 1, L1_size_half);
    _pipe.InitBuffer(_inQueueB1, 2, L1_size_half_pp);
    _dataCopyB1Params.ndNum = 1;
    _dataCopyB1Params.nValue = n0;
    _dataCopyB1Params.dValue = k0;
    _dataCopyB1Params.srcNdMatrixStride = 0;
    _dataCopyB1Params.srcDValue = _k;
    _dataCopyB1Params.dstNzC0Stride = n0;
    _dataCopyB1Params.dstNzNStride = 1;
    _dataCopyB1Params.dstNzMatrixStride = 0;

    // Prefetch first B block
    auto b1Local = _inQueueB1.AllocTensor<bf16>();
    AscendC::DataCopy(
        b1Local, _bGm[CalculateGlobalBOffset(row_idx_start, col_idx_start, _k)],
        _dataCopyB1Params);
    _inQueueB1.EnQue(b1Local);

    // Initialize other queues
    _pipe.InitBuffer(_inQueueA2, 1, L0A_size);
    _pipe.InitBuffer(_inQueueB2, 2, L0B_size_pp);
    _pipe.InitBuffer(_outQueueCO1, 2, L0C_size_pp);

    // Initialize parameters
    _mmad_offset_max = _tiling_data.mmadOffsetMax;

    // Initialize data copy parameters for A
    _dataCopyA1Params.ndNum = 1;
    _dataCopyA1Params.nValue = _m;
    _dataCopyA1Params.dValue = k0;
    _dataCopyA1Params.srcNdMatrixStride = 0;
    _dataCopyA1Params.srcDValue = _k;
    _dataCopyA1Params.dstNzC0Stride = 16; // pad to 16
    _dataCopyA1Params.dstNzNStride = 1;
    _dataCopyA1Params.dstNzMatrixStride = 0;

    // Initialize load parameters
    _loadL0AParams.repeatTimes = k0 / 16;
    _loadL0AParams.srcStride = 1;
    _loadL0AParams.ifTranspose = false;

    _loadL0BParams.repeatTimes = n0 * k_part / CONST_256;
    _loadL0BParams.srcStride = 1;
    _loadL0BParams.ifTranspose = false;

    // Initialize mmad parameters
    _mmadParams.m = 16; // baseM
    _mmadParams.n = n0;
    _mmadParams.k = k_part;

    // Initialize fixpipe parameters
    _fixpipeParams.nSize = n0;
    _fixpipeParams.mSize = _m;
    _fixpipeParams.srcStride = 16; // baseM
    _fixpipeParams.dstStride = _n;
    _fixpipeParams.ndNum = 1;
    _fixpipeParams.srcNdStride = 0;
    _fixpipeParams.dstNdStride = 0;
  }

  __aicore__ inline void Process() {
    AscendC::LocalTensor<float> c1Local;

    uint32_t next_i;
    for (uint32_t i = _start_idx;; i = next_i) {
      uint32_t row_idx = i % _rows;
      uint32_t col_idx = i / _rows;
      if (i == _start_idx || row_idx == 0) {
        _mmadParams.cmatrixInitVal = true;
        c1Local = _outQueueCO1.AllocTensor<float>();
      }

      // Load A from GM to L1 to L0A
      auto a1Local = _inQueueA1.AllocTensor<bf16>();
      auto a2Local = _inQueueA2.AllocTensor<bf16>();
      AscendC::DataCopy(a1Local, _aGm[row_idx * k0], _dataCopyA1Params);
      _inQueueA1.EnQue(a1Local);
      auto a1LocalOut = _inQueueA1.DeQue<bf16>();
      AscendC::LoadData(a2Local, a1LocalOut, _loadL0AParams);
      _inQueueA1.FreeTensor(a1LocalOut);
      _inQueueA2.EnQue(a2Local);
      auto a2LocalOut = _inQueueA2.DeQue<bf16>();

      // Prefetch next B block from GM to L1
      next_i = i + 1;
      if (next_i != _end_idx) {
        uint32_t next_row_idx = next_i % _rows;
        uint32_t next_col_idx = next_i / _rows;
        uint32_t global_b_offset =
            CalculateGlobalBOffset(next_row_idx, next_col_idx, _k);
        auto b1Local = _inQueueB1.AllocTensor<bf16>();
        AscendC::DataCopy(b1Local, _bGm[global_b_offset], _dataCopyB1Params);
        _inQueueB1.EnQue(b1Local);
      }

      // Get current B block in L1
      auto b1LocalOut = _inQueueB1.DeQue<bf16>();

      for (uint32_t mmad_offset = 0; mmad_offset < _mmad_offset_max;
           mmad_offset += k_part * 16) {
        // Load B from L1 to L0B
        auto b2Local = _inQueueB2.AllocTensor<bf16>();
        AscendC::LoadData(b2Local, b1LocalOut[mmad_offset * 16],
                          _loadL0BParams);
        _inQueueB2.EnQue(b2Local);
        auto b2LocalOut = _inQueueB2.DeQue<bf16>();

        // Perform MMAD operation
        AscendC::Mmad(c1Local, a2LocalOut[mmad_offset], b2LocalOut,
                      _mmadParams);
        _mmadParams.cmatrixInitVal = false;
        _inQueueB2.FreeTensor(b2LocalOut);
      }

      _inQueueA2.FreeTensor(a2LocalOut);
      _inQueueB1.FreeTensor(b1LocalOut);

      if (next_i == _end_idx || row_idx == _rows - 1) {
        // Fixpipe operation
        _outQueueCO1.EnQue<float>(c1Local);
        auto c1Local_out = _outQueueCO1.DeQue<float>();
        AscendC::SetAtomicAdd<float>();
        AscendC::Fixpipe(_cGm[col_idx * n0], c1Local_out, _fixpipeParams);
        AscendC::SetAtomicNone();
        if (next_i == _end_idx) {
          // The computing is now over
          AscendC::SyncAll<false>();
          return;
        }
        _outQueueCO1.FreeTensor(c1Local_out);
      }
    }
  }

private:
  // Tiling and core info
  uint32_t _m, _k, _n, _rows;
  uint32_t _start_idx;
  uint32_t _end_idx;
  int64_t _core_idx;

  // Looping parameters
  uint32_t _mmad_offset_max;

  // Tensors and queues
  AscendC::TPipe _pipe;
  AscendC::GlobalTensor<bf16> _aGm;
  AscendC::GlobalTensor<bf16> _bGm;
  AscendC::GlobalTensor<float> _cGm;
  AscendC::TQue<AscendC::TPosition::A1, 1> _inQueueA1;
  AscendC::TQue<AscendC::TPosition::A2, 1> _inQueueA2;
  AscendC::TQue<AscendC::TPosition::B1, 2> _inQueueB1;
  AscendC::TQue<AscendC::TPosition::B2, 2> _inQueueB2;
  AscendC::TQue<AscendC::TPosition::CO1, 2> _outQueueCO1;

  // Algorithm parameters
  AscendC::Nd2NzParams _dataCopyA1Params;
  AscendC::Nd2NzParams _dataCopyB1Params;
  AscendC::LoadData2dParams _loadL0AParams;
  AscendC::LoadData2dParams _loadL0BParams;
  AscendC::MmadParams _mmadParams;
  AscendC::FixpipeParamsV220 _fixpipeParams;
};

extern "C" __global__ __aicore__ void pp_matmul_opt(GM_ADDR a, GM_ADDR b,
                                                    GM_ADDR c,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
  if ASCEND_IS_AIC {
    KernelPPMatmulOpt op;
    op.Init(a, b, c, usrWorkspace, tiling);
    // Wait for AIV cores to initialize usrWorkspace
    AscendC::SyncAll<false>();
    op.Process();
  }
  if ASCEND_IS_AIV {
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t m = tiling_data.m;
    uint32_t n = tiling_data.n;
    uint32_t core_idx = AscendC::GetBlockIdx();

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> srcGlobal;
    AscendC::GlobalTensor<bf16> dstGlobal;
    srcGlobal.SetGlobalBuffer((__gm__ float *)usrWorkspace + core_idx * n);
    dstGlobal.SetGlobalBuffer((__gm__ bf16 *)c + core_idx * n);

    if (core_idx < m) {
      AscendC::InitGlobalMemory(srcGlobal, n, (float)(0.0));
    }
    // The initialization of usrWorkspace is now over
    AscendC::SyncAll<false>();

    pipe.InitBuffer(inQueueSrc, 1, n * sizeof(float));
    pipe.InitBuffer(outQueueDst, 1, n * sizeof(bf16));

    // Wait for all ai cores finish computing
    AscendC::SyncAll<false>();
    if (core_idx < m) {
      auto srcLocal = inQueueSrc.AllocTensor<float>();
      AscendC::DataCopy(srcLocal, srcGlobal, n);
      inQueueSrc.EnQue(srcLocal);
      srcLocal = inQueueSrc.DeQue<float>();

      auto dstLocal = outQueueDst.AllocTensor<bf16>();
      AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_RINT, n);
      outQueueDst.EnQue<bf16>(dstLocal);
      inQueueSrc.FreeTensor(srcLocal);
      dstLocal = outQueueDst.DeQue<bf16>();
      AscendC::DataCopy(dstGlobal, dstLocal, n);
      outQueueDst.FreeTensor(dstLocal);
    }
  }
}