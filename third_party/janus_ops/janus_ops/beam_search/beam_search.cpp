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

#include "./beam_search.h"

#include "kernel_operator.h"
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::Init(
    GM_ADDR log_probs, GM_ADDR top_tokens, GM_ADDR top_probs,
    GM_ADDR out_token_ids, GM_ADDR out_token_index, GM_ADDR out_log_probs, 
    int32_t num_sequences, int32_t sequence_length,
    int32_t beam_width, int32_t top_k, int32_t request_num, int32_t core_num,
    int32_t min_size, int32_t step_size, TopkTiling &topkTilingData, TopkTiling &topKTilingData1) {
  log_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)log_probs);
  top_tokens_gm.SetGlobalBuffer((__gm__ TokenIdType *)top_tokens);
  top_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)top_probs);
  out_token_ids_gm.SetGlobalBuffer((__gm__ TokenIdType *)out_token_ids);
  out_token_index_gm.SetGlobalBuffer((__gm__ int32_t *)out_token_index);
  out_log_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)out_log_probs);
  this->num_sequences = num_sequences;
  this->sequence_length = sequence_length;
  this->beam_width = beam_width;
  this->top_k = top_k;
  this->request_num = request_num;
  this->core_num = core_num;
  this->topkTilingData = topkTilingData;
  this->topKTilingData1 = topKTilingData1;
  this->align_beam_width = AlignUp(this->beam_width, 32);
  this->align_top_k = AlignUp(this->top_k, 32 / sizeof(LogProbType));
  this->min_size = min_size;
  this->step_size = step_size;
  pipe.InitBuffer(log_probs_in_que, 1,
                  this->align_beam_width * sizeof(LogProbType)); // 1k
  pipe.InitBuffer(top_tokens_in_que, 1, this->step_size * this->align_beam_width*sizeof(LogProbType)); // this->step_size is logic number
  pipe.InitBuffer(out_token_ids_out_que,1,this->align_beam_width * sizeof(TokenIdType));
  pipe.InitBuffer(out_token_index_out_que,1,this->align_beam_width * sizeof(TokenIdType));
  pipe.InitBuffer(out_log_probs_out_que,1,this->align_beam_width * sizeof(LogProbType));
  pipe.InitBuffer(top_k_result_prob_buf,
                  this->align_beam_width * sizeof(LogProbType)); // 2k
  pipe.InitBuffer(top_k_result_index_buf,
                  this->align_beam_width * sizeof(int32_t)); // 2k
  pipe.InitBuffer(prefix_probs_buf, this->align_beam_width * sizeof(LogProbType));    // 1k
  pipe.InitBuffer(prefix_index_buf, this->align_beam_width * sizeof(int32_t)); // 2k
  pipe.InitBuffer(top_k_second_res_buf, this->align_beam_width * sizeof(LogProbType));   // 1k
  pipe.InitBuffer(merge_probs_buf,
                  this->align_beam_width * sizeof(LogProbType) * 2);       // 2k
  pipe.InitBuffer(merge_index_buf, this->align_beam_width * sizeof(int32_t) * 2); // 4k
  pipe.InitBuffer(top_k_second_res_index_buf, this->align_beam_width * sizeof(int32_t));     // 2k                                // 2k
  pipe.InitBuffer(top_k_tmp_buf, this->min_size * sizeof(int32_t));
  pipe.InitBuffer(align_float_buf,this->align_beam_width*sizeof(float));
  pipe.InitBuffer(remainder_buf,this->align_beam_width*sizeof(int32_t));
  /*
  // old version
  int32_t align_sequence_length = AlignUp(this->sequence_length, 32);
  pipe.InitBuffer(out_token_ids_out_que, 1,
                  align_sequence_length * sizeof(TokenIdType)); // 8k or 16k
  pipe.InitBuffer(token_ids_in_que, 1,
                  align_sequence_length * sizeof(TokenIdType)); // 1k
  */
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline int32_t
BeamSearch<TokenIdType, LogProbType>::AlignUp(int32_t value,
                                              int32_t alignment) {
  if (value % alignment != 0) {
    value = (value / alignment + 1) * alignment;
  }
  return value;
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::Process() {
  for (int32_t request_idx = 0; request_idx < this->request_num; request_idx++) {
    if(request_idx % 24 == block_idx){
      LocalTensor<LogProbType> prefix_top_probs = prefix_probs_buf.Get<LogProbType>();
      AscendC::Duplicate<LogProbType>(prefix_top_probs, static_cast<LogProbType>(-1.0f / 0.0f),this->align_beam_width);
      LocalTensor<int32_t> prefix_top_index = prefix_index_buf.GetWithOffset<int32_t>(this->align_beam_width, 0);
      AscendC::Duplicate<int32_t>(prefix_top_index, static_cast<int32_t>(0),this->align_beam_width);
      Psum(request_idx, prefix_top_probs, prefix_top_index);
      StackWithOutput(request_idx, prefix_top_probs, prefix_top_index);
    }
  }
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopy(
    AscendC::LocalTensor<TokenIdType> dst,
    AscendC::GlobalTensor<TokenIdType> src, int32_t length) {
  int32_t block_size = AlignUp(length,32 / sizeof(TokenIdType));
  DataCopyPadExtParams<TokenIdType> params;
  params.isPad = true;
  params.paddingValue = 0;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(TokenIdType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyGm(
    AscendC::GlobalTensor<TokenIdType> dst,
    AscendC::LocalTensor<TokenIdType> src, int32_t length) {
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(TokenIdType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(dst, src, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyGmLog(
    AscendC::GlobalTensor<LogProbType> dst,
    AscendC::LocalTensor<LogProbType> src, int32_t length) {
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(dst, src, copyParams);
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyProb(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src, int32_t length) {
  int32_t block_size = AlignUp(length,32 / sizeof(LogProbType));
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = 0;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyProbSlice(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src, int32_t length,
    int32_t slice_length) {
  int32_t block_size = AlignUp(length,32 / sizeof(LogProbType));
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = -1.0f / 0.0f;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = slice_length;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
  AscendC::PipeBarrier<PIPE_MTE2>();
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpCopyInt(
    LocalTensor<int32_t> dst, AscendC::LocalTensor<int32_t> src,
    int32_t length) {
  int32_t mask = 64;
  AscendC::CopyRepeatParams copyParams;
  copyParams.dstStride = 1;
  copyParams.srcStride = 1;
  copyParams.dstRepeatSize = 8;
  copyParams.srcRepeatSize = 8;
  AscendC::Copy<int32_t, true>(dst, src, mask, length / 64, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpCopyProb(
    LocalTensor<LogProbType> dst, AscendC::LocalTensor<LogProbType> src,
    int32_t length) {
  int32_t mask = 256 / sizeof(LogProbType);
  AscendC::CopyRepeatParams copyParams;
  copyParams.dstStride = 1;
  copyParams.srcStride = 1;
  copyParams.dstRepeatSize = 8;
  copyParams.srcRepeatSize = 8;
  AscendC::Copy<LogProbType, true>(dst, src, mask, length / mask, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearch<TokenIdType, LogProbType>::Psum(int32_t request_idx,AscendC::LocalTensor<LogProbType> &prefix_top_probs,
  AscendC::LocalTensor<int32_t> &prefix_top_index) {
  LocalTensor<LogProbType> log_probs_local = log_probs_in_que.AllocTensor<LogProbType>();
  AlignUpDataCopyProb(log_probs_local,log_probs_gm[request_idx * this->beam_width],this->beam_width);
  log_probs_in_que.EnQue(log_probs_local);
  int32_t beam_width_round = (this->beam_width + this->step_size - 1) / this->step_size;
  LocalTensor<LogProbType> log_probs_per = log_probs_in_que.DeQue<LogProbType>();
  int32_t tail_number = this->beam_width % this->step_size;
  for (int32_t i = 0; i < beam_width_round; i++) {
    LocalTensor<LogProbType> top_probs_local = top_tokens_in_que.AllocTensor<LogProbType>();

    // We need to make sure the TopK input are initialized
    // TopK block_size is aligned to 32
    bool is_last_round_with_tail = (i == beam_width_round - 1 && tail_number != 0);
    int32_t round_size = is_last_round_with_tail ? tail_number : this->step_size;
    int32_t full_round_top_k_block_size = AlignUp(this->step_size * this->align_top_k, 32);
    if (is_last_round_with_tail) {
      // Note: This is an issue of AscendC:
      // The TopK op uses both topKInfo and topkTilingData as the input config
      // TopK will use the block_size from topkTilingData, which leads to data reuse from previous round
      // Here we simply reinit [top_probs_local, end)
      AscendC::Duplicate<LogProbType>(top_probs_local, static_cast<LogProbType>(-1.0f / 0.0f), full_round_top_k_block_size);
    } else {
      LocalTensor top_probs_local_last_topk_block = top_probs_local[full_round_top_k_block_size - 32];
      AscendC::Duplicate<LogProbType>(top_probs_local_last_topk_block, static_cast<LogProbType>(-1.0f / 0.0f), 32);
    }
    int32_t eventIDVToMTE2 = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));	
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);	
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);

    AlignUpDataCopyProbSlice(
      top_probs_local,
      top_probs_gm[request_idx * this->beam_width * this->top_k + i * this->step_size * this->top_k], this->top_k,
      round_size);
    top_tokens_in_que.EnQue(top_probs_local);

    LocalTensor<LogProbType> top_probs_local_tmp2 = top_tokens_in_que.DeQue<LogProbType>();
    for (int beam_width_offset = 0; beam_width_offset < round_size;
         beam_width_offset++) {
      int32_t log_probs_beam_offset = i*this->step_size+beam_width_offset;
      LogProbType log_probs_value = log_probs_per.GetValue(log_probs_beam_offset);
      AscendC::Adds<LogProbType>(
          top_probs_local_tmp2[beam_width_offset * this->align_top_k],
          top_probs_local_tmp2[beam_width_offset * this->align_top_k],
          log_probs_value, this->top_k);
    }
    TopKWithSorted(request_idx, top_probs_local_tmp2, prefix_top_probs,
                   prefix_top_index, round_size, i);
    top_tokens_in_que.FreeTensor(top_probs_local_tmp2);
  }
  
  log_probs_in_que.FreeTensor(log_probs_per);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::TopKWithSorted(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &top_probs_local,
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index, int32_t round_length_top_k,int32_t round_idx) {
  int32_t offset = 0;
  LocalTensor<LogProbType> dst_local_value =
      top_k_result_prob_buf.Get<LogProbType>();
  LocalTensor<int32_t> dst_local_index = top_k_result_index_buf.Get<int32_t>();
  LocalTensor<int32_t> src_local_index;
  LocalTensor<bool> src_local_finish;
  int32_t block_size = AlignUp(round_length_top_k * this->align_top_k, 32);
  TopKInfo topKInfo;
  topKInfo.outter = 1;
  topKInfo.inner = block_size;
  topKInfo.n = round_length_top_k * this->align_top_k;
  bool isLargest = true;
  LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();

  AscendC::TopK<LogProbType, false, false, false,AscendC::TopKMode::TOPK_NORMAL>(
      dst_local_value, dst_local_index, top_probs_local, src_local_index,
      src_local_finish, tmp_local, this->align_top_k, this->topkTilingData, topKInfo,
      isLargest);
  AscendC::Adds<int32_t>(dst_local_index, dst_local_index,
                         static_cast<int32_t>((round_idx * this->step_size)*this->align_top_k),
                         this->align_top_k);
  /*
  // old version
  AscendC::Adds<int32_t>(dst_local_index, dst_local_index,
  static_cast<int32_t>((request_idx * this->beam_width+round_idx * this->step_size)*this->align_top_k),
  this->beam_width);
  */ 
  LocalTensor<LogProbType> merge_top_probs = merge_probs_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(merge_top_probs, static_cast<LogProbType>(-1.0f / 0.0f),2*this->align_beam_width);
  AscendC::DataCopy(merge_top_probs, dst_local_value, this->align_top_k);
  AscendC::DataCopy(merge_top_probs[this->align_top_k], prefix_top_probs, this->align_top_k);
  LocalTensor<int32_t> merge_index = merge_index_buf.Get<int32_t>();
  AscendC::DataCopy(merge_index,dst_local_index,this->align_top_k);
  AscendC::DataCopy(merge_index[this->align_top_k], prefix_top_index, this->align_top_k);
  LocalTensor<LogProbType> dst_merge_probs = top_k_second_res_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(dst_merge_probs, static_cast<LogProbType>(-1.0f / 0.0f),
                              this->align_beam_width);
  LocalTensor<int32_t> dst_merge_index = top_k_second_res_index_buf.Get<int32_t>();
  AscendC::Duplicate<int32_t>(dst_merge_index, static_cast<int32_t>(0),
                              this->align_beam_width);
  int32_t block_size_merge = AlignUp(this->beam_width * 2, 32);
  TopKInfo topKInfo2;
  topKInfo2.outter = 1;
  topKInfo2.inner = block_size_merge;
  topKInfo2.n = this->beam_width * 2;
  AscendC::TopK<LogProbType, true, false, false,AscendC::TopKMode::TOPK_NORMAL>(
          dst_merge_probs,
          dst_merge_index,
          merge_top_probs,
          merge_index,
          src_local_finish,
          tmp_local,
          this->align_top_k,
          this->topKTilingData1,
          topKInfo2,
          isLargest);
  AscendC::DataCopy(prefix_top_probs, dst_merge_probs, this->align_top_k);

  AscendC::DataCopy(prefix_top_index, dst_merge_index, this->align_top_k);
}

// New version: Output topk token ids and log_probs
// out_token_ids_gm shape is [request_num * beam_width]
// out_token_index_gm shape is [request_num * beam_width]
// out_log_probs_gm shape is [request_num * beam_width]

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::StackWithOutput(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index) {
  int32_t out_token_ids_offset = 0;
  // reuse the merge_probs_buf  buffer 
  LocalTensor align_top_k_vector_float = merge_probs_buf.GetWithOffset<float>(this->align_beam_width,0);
  AscendC::Duplicate<float>(align_top_k_vector_float, static_cast<float>(this->align_top_k),
  this->align_beam_width);
  // resue the tmp buffer used by the topk
  LocalTensor align_top_k_vector_cp = merge_index_buf.GetWithOffset<float>(this->align_beam_width,0);
  AscendC::Duplicate<float>(align_top_k_vector_cp, static_cast<float>(0),
  this->align_beam_width);
  AscendC::Cast(align_top_k_vector_cp,prefix_top_index,AscendC::RoundMode::CAST_RINT,this->align_beam_width);
  align_top_k_vector_float =align_top_k_vector_cp / align_top_k_vector_float;

  // reuse the index buffer used by topK for the second time
  LocalTensor align_top_k_vector = top_k_second_res_index_buf.GetWithOffset<int32_t>(this->align_beam_width,0);
  AscendC::Duplicate<int32_t>(align_top_k_vector, static_cast<int32_t>(0),
  this->align_beam_width);
  AscendC::Cast(align_top_k_vector,align_top_k_vector_float,AscendC::RoundMode::CAST_FLOOR,this->align_top_k);
  // reuse the top_k_result_index_buf
  LocalTensor align_remainder_vector = top_k_result_index_buf.GetWithOffset<int32_t>(this->align_beam_width,0);
  AscendC::Duplicate<int32_t>(align_remainder_vector, static_cast<int32_t>(this->align_top_k),
  this->align_beam_width);
  align_remainder_vector = align_top_k_vector * align_remainder_vector;
  align_remainder_vector = prefix_top_index - align_remainder_vector;
  
  // get token index result
  LocalTensor out_token_index_local = out_token_index_out_que.AllocTensor<TokenIdType>();
  AscendC::Duplicate<int32_t>(out_token_index_local,static_cast<int32_t>(0),this->align_beam_width);
  AscendC::Adds(out_token_index_local,align_top_k_vector,static_cast<int32_t>(request_idx * this->beam_width),this->align_beam_width);
  out_token_index_out_que.EnQue(out_token_index_local);
  LocalTensor out_token_index_local_tmp = out_token_index_out_que.DeQue<int32_t>();
  AlignUpDataCopyGm(out_token_index_gm[request_idx * this->beam_width], out_token_index_local_tmp, this->beam_width);
  out_token_index_out_que.FreeTensor(out_token_index_local_tmp);

  // get log probs result
  LocalTensor out_log_probs_local = out_log_probs_out_que.AllocTensor<LogProbType>();
  AscendC::DataCopy(out_log_probs_local,prefix_top_probs,this->align_beam_width);
  out_log_probs_out_que.EnQue(out_log_probs_local);
  LocalTensor out_log_probs_local_tmp = out_log_probs_out_que.DeQue<LogProbType>();
  AlignUpDataCopyGmLog(out_log_probs_gm[request_idx * this->beam_width],out_log_probs_local_tmp,this->beam_width);
  out_log_probs_out_que.FreeTensor(out_log_probs_local_tmp);

  // get token ids result
  LocalTensor out_token_ids_local = out_token_ids_out_que.AllocTensor<TokenIdType>();
  AscendC::Muls(align_top_k_vector,align_top_k_vector,static_cast<int32_t>(this->top_k),this->align_beam_width);
  prefix_top_index = align_top_k_vector + align_remainder_vector;
  AscendC::Adds(prefix_top_index,prefix_top_index,static_cast<TokenIdType>(request_idx * this->beam_width * this->top_k),this->align_beam_width);
  for(int32_t i=0; i<this->beam_width ;i++){
    int32_t value = prefix_top_index.GetValue(i);
    out_token_ids_local.SetValue(i,top_tokens_gm.GetValue(value));
  }
  AlignUpDataCopyGm(out_token_ids_gm[request_idx* this->beam_width] ,out_token_ids_local,this->beam_width);
  out_token_ids_out_que.FreeTensor(out_token_ids_local);
}

/*
// old version: output complete token ids
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::StackWithOutput(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index) {
  int32_t out_token_ids_offset = 0;
  for (int32_t i = 0; i < this->beam_width; i++) {
    LocalTensor<TokenIdType> token_ids_local =
        token_ids_in_que.AllocTensor<TokenIdType>();
    int32_t prefix_top_index_value = prefix_top_index.GetValue(i);
    int32_t true_top_index = prefix_top_index_value-(request_idx * this->beam_width*this->align_top_k);
    int32_t q = true_top_index / this->align_top_k;
    int32_t c = true_top_index % this->align_top_k;
    true_top_index = q * this->top_k + c;
   
    int32_t token_idx = request_idx * this->beam_width + q;

    int32_t top_tokens_index = request_idx * this->beam_width * this->top_k + true_top_index;
    int32_t top_tokens_gm_value = top_tokens_gm.GetValue(top_tokens_index);
    
    out_token_ids_offset = (request_idx * this->beam_width + i) * (this->sequence_length + 1);
    int32_t length_token = AlignUp(this->sequence_length, 32);
    uint32_t offset = token_idx * (this->sequence_length); 
    AlignUpDataCopy(token_ids_local,
                    token_ids_gm[offset],
                    this->sequence_length);
    token_ids_in_que.EnQue(token_ids_local);
    LocalTensor<TokenIdType> token_ids_local_tmp =
        token_ids_in_que.DeQue<TokenIdType>();
    LocalTensor<TokenIdType> out_token_ids_local_tmp =
        out_token_ids_out_que.AllocTensor<TokenIdType>();
    int32_t length_align = AlignUp(length_token, 64 / sizeof(int32_t));
    AscendC::DataCopy(out_token_ids_local_tmp, token_ids_local_tmp, length_token);
    out_token_ids_out_que.EnQue(out_token_ids_local_tmp);
    LocalTensor<TokenIdType> out_token_ids_local_tmp_tmp =
        out_token_ids_out_que.DeQue<TokenIdType>();
    AlignUpDataCopyGm(out_token_ids_gm[out_token_ids_offset],
                      out_token_ids_local_tmp_tmp, this->sequence_length);
    out_token_ids_gm.SetValue(
      (request_idx * beam_width + i) * (this->sequence_length + 1) + this->sequence_length,
      top_tokens_gm_value);
    token_ids_in_que.FreeTensor(token_ids_local_tmp);
    out_token_ids_out_que.FreeTensor(out_token_ids_local_tmp_tmp);
  }
}
*/
extern "C" __global__ __aicore__ void
beam_search(GM_ADDR log_probs, GM_ADDR top_tokens,
            GM_ADDR top_probs, GM_ADDR out_token_ids, GM_ADDR out_token_index, GM_ADDR out_log_probs, GM_ADDR workspace,
            GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  BeamSearch<int32_t, float> op;
  op.Init(log_probs, top_tokens, top_probs, out_token_ids, out_token_index, out_log_probs,
          tiling_data.num_sequences, tiling_data.sequence_length,
          tiling_data.beam_width, tiling_data.top_k, tiling_data.request_num,
          tiling_data.core_num, tiling_data.min_size, tiling_data.step_size,
          tiling_data.topkTilingData, tiling_data.topKTilingData1);
  op.Process();
}