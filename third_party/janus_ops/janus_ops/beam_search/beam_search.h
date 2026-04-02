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
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
template <typename TokenIdType, typename LogProbType> class BeamSearch {
public:
  __aicore__ inline BeamSearch() {}
  __aicore__ inline void
  Init(GM_ADDR log_probs, GM_ADDR top_tokens,
       GM_ADDR top_probs, GM_ADDR out_token_ids, GM_ADDR out_token_index,
       GM_ADDR out_log_probs, int32_t num_sequences, int32_t sequence_length,
       int32_t beam_width, int32_t top_k, int32_t request_num, int32_t core_num,
       int32_t min_size, int32_t step_size, TopkTiling &topkTilingData,
       TopkTiling &topKTilingData1);
  __aicore__ inline void Process();

  __aicore__ inline int32_t AlignUp(int32_t value, int32_t alignment);
  __aicore__ inline void AlignUpDataCopy(AscendC::LocalTensor<TokenIdType> dst,
                                         AscendC::GlobalTensor<TokenIdType> src,
                                         int32_t length);
  __aicore__ inline void
  AlignUpDataCopyGm(AscendC::GlobalTensor<TokenIdType> dst,
                    AscendC::LocalTensor<TokenIdType> src, int32_t length);
  __aicore__ inline void
  AlignUpDataCopyGmLog(AscendC::GlobalTensor<LogProbType> dst,
                       AscendC::LocalTensor<LogProbType> src, int32_t length);

  __aicore__ inline void
  AlignUpDataCopyProb(AscendC::LocalTensor<LogProbType> dst,
                      AscendC::GlobalTensor<LogProbType> src, int32_t length);
  __aicore__ inline void
  AlignUpDataCopyProbSlice(AscendC::LocalTensor<LogProbType> dst,
                           AscendC::GlobalTensor<LogProbType> src,
                           int32_t length, int32_t slice_length);
  __aicore__ inline void AlignUpCopyInt(AscendC::LocalTensor<int32_t> dst,
                                        AscendC::LocalTensor<int32_t> src,
                                        int32_t length);
  __aicore__ inline void AlignUpCopyProb(AscendC::LocalTensor<LogProbType> dst,
                                         AscendC::LocalTensor<LogProbType> src,
                                         int32_t length);
  __aicore__ inline void
  Psum(int32_t request_idx, AscendC::LocalTensor<LogProbType> &prefix_top_probs,
       AscendC::LocalTensor<int32_t> &prefix_top_index);
  __aicore__ inline void
  TopKWithSorted(int32_t request_idx,
                 AscendC::LocalTensor<LogProbType> &top_probs_local,
                 AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                 AscendC::LocalTensor<int32_t> &prefix_top_index,
                 int32_t step_size, int32_t round_idx);
  __aicore__ inline void
  StackWithOutput(int32_t request_idx,
                  AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                  AscendC::LocalTensor<int32_t> &prefix_top_index);

private:
  AscendC::GlobalTensor<TokenIdType> token_ids_gm;
  AscendC::GlobalTensor<LogProbType> log_probs_gm;
  AscendC::GlobalTensor<TokenIdType> top_tokens_gm;
  AscendC::GlobalTensor<LogProbType> top_probs_gm;
  AscendC::GlobalTensor<TokenIdType> out_token_ids_gm;
  AscendC::GlobalTensor<TokenIdType> out_token_index_gm;
  AscendC::GlobalTensor<LogProbType> out_log_probs_gm;
  int32_t num_sequences;
  int32_t sequence_length;
  int32_t beam_width;
  int32_t top_k;
  int32_t request_num;
  int32_t core_num;
  int32_t core_idx;
  int32_t align_beam_width;
  int32_t align_top_k;
  int32_t min_size;
  int32_t step_size;
  TopkTiling topkTilingData;
  TopkTiling topKTilingData1;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> token_ids_in_que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> log_probs_in_que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> top_tokens_in_que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> top_probs_in_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_token_ids_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_token_index_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_log_probs_out_que;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_prob_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prefix_probs_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prefix_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_second_res_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> merge_probs_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> merge_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_second_res_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> align_float_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> remainder_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> align_top_k_vector_buf;
};
