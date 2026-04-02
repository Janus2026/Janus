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

#include "acl/acl.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "x_attention_tl.h"
using namespace Catlass;
template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M, uint32_t BLOCK_M_UNSHARED>
__aicore__ inline void XAttentionTlKernel<HEAD_SIZE, BLOCK_N, BLOCK_M, BLOCK_M_UNSHARED>::Init(
    GM_ADDR Q_handle,
    GM_ADDR K_handle,
    GM_ADDR V_handle,
    GM_ADDR q_unshared_handle,
    GM_ADDR unshared_key_handle,
    GM_ADDR unshared_value_handle,
    GM_ADDR output_shared_handle,
    GM_ADDR output_unshared_handle,
    GM_ADDR shared_exp_handle,
    GM_ADDR unshared_exp_handle,
    GM_ADDR shared_max_handle,
    GM_ADDR unshared_max_handle,
    GM_ADDR Output_handle,
    GM_ADDR workspace,
    int64_t batch_size,
    int64_t beam_size,
    int64_t shared_k_length,
    int64_t unshared_k_length,
    int64_t q_length,
    int64_t head_size,
    int64_t head_num
  ) {
  // 缓存 tiling 参数
  batch_ = batch_size;
  q_len_ = beam_size;
  kv_len_ = shared_k_length;
  unshared_k_len_ = unshared_k_length;
  unshared_q_len_ = q_length;
  head_size_ = head_size;
  head_num_ = head_num;
  beam_size_ = beam_size;
  block_m = 64;
  block_n = 64;
  block_m_unshared = 1;
  tiles_q_ = q_len_ / block_m;
  tatal_shared_tasks_ = (beam_size / block_m) * batch_size * head_num;
  tatal_unshared_tasks_ = beam_size * batch_size * head_num;
  shared_core_total_num_ = 16;
  unshared_core_total_num_ = 8;
  single_shared_task_num_ = tatal_shared_tasks_ / shared_core_total_num_;
  single_unshared_task_num_ = tatal_unshared_tasks_ / unshared_core_total_num_;
  workspace_1_size_ = batch_size * head_num * tiles_q_ * block_m * block_n * sizeof(float);
  workspace_2_size_ = batch_size * head_num * tiles_q_ * block_m * block_n * sizeof(half);
  workspace_3_size_ = batch_size * head_num * tiles_q_ * block_m * head_size_ * sizeof(float);
  workspace_1_unshared_size_ = batch_size * head_num * beam_size_ * block_m_unshared * block_n * sizeof(float);
  workspace_2_unshared_size_ = batch_size * head_num * beam_size_ * block_m_unshared * block_n * sizeof(half);
  workspace_3_unshared_size_ = batch_size * head_num * beam_size_ * block_m_unshared * head_size_ * sizeof(float);
  // 绑定全局内存
  Q_.SetGlobalBuffer((__gm__ half*)Q_handle);
  K_.SetGlobalBuffer((__gm__ half*)K_handle);
  V_.SetGlobalBuffer((__gm__ half*)V_handle);
  Q_unshared_.SetGlobalBuffer((__gm__ half*)q_unshared_handle);
  K_unshared_.SetGlobalBuffer((__gm__ half*)unshared_key_handle);
  V_unshared_.SetGlobalBuffer((__gm__ half*)unshared_value_handle);
  Output_shared_.SetGlobalBuffer((__gm__ half*)output_shared_handle);
  Output_unshared_.SetGlobalBuffer((__gm__ half*)output_unshared_handle);
  Output_.SetGlobalBuffer((__gm__ half*)Output_handle);
  workspace_1_.SetGlobalBuffer((__gm__ float*)workspace);
  workspace_2_.SetGlobalBuffer((__gm__ half*)(workspace + workspace_1_size_));
  workspace_3_.SetGlobalBuffer((__gm__ float*)(workspace + workspace_1_size_ + workspace_2_size_));
  workspace_1_unshared_.SetGlobalBuffer((__gm__ float*)(workspace + workspace_1_size_ + workspace_2_size_ + workspace_3_size_));
  workspace_2_unshared_.SetGlobalBuffer((__gm__ half*)(workspace + workspace_1_size_ + workspace_2_size_ + workspace_3_size_ + workspace_1_unshared_size_));
  workspace_3_unshared_.SetGlobalBuffer((__gm__ float*)(workspace + workspace_1_size_ + workspace_2_size_ + workspace_3_size_ + workspace_1_unshared_size_ + workspace_2_unshared_size_));
  Shared_exp_.SetGlobalBuffer((__gm__ float*)shared_exp_handle);
  Unshared_exp_.SetGlobalBuffer((__gm__ float*)unshared_exp_handle);
  Shared_max_.SetGlobalBuffer((__gm__ float*)shared_max_handle);
  Unshared_max_.SetGlobalBuffer((__gm__ float*)unshared_max_handle);
  // 初始化片上缓冲
  pipe.InitBuffer(ascend_l0a_, 65536);
  pipe.InitBuffer(ascend_l0b_, 131072);
  pipe.InitBuffer(ascend_l1_, 524032);
  pipe.InitBuffer(ascend_l0c_, 131072);
  pipe.InitBuffer(ascend_ub_, 196352);
}

template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M, uint32_t BLOCK_M_UNSHARED>
__aicore__ inline void XAttentionTlKernel<HEAD_SIZE, BLOCK_N, BLOCK_M, BLOCK_M_UNSHARED>::run() {
    cid_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
      cid_ = cid_ / 2;
    }
    vid_ = AscendC::GetSubBlockIdx();
    // printf("cid_: %d, vid_: %d\n", cid_, vid_);
    if ASCEND_IS_AIC {
      if(cid_ < shared_core_total_num_){
        for (int32_t i = 0; i < single_shared_task_num_; ++i) {
          int32_t task_id = cid_ * single_shared_task_num_ + i;
          runAIC(task_id);
        }
      }
      else{
        for (int32_t i = 0; i < single_unshared_task_num_; ++i) {
          int32_t task_id = (cid_-shared_core_total_num_) * single_unshared_task_num_ + i;
          runAicUnshared(task_id);
        }
      }
    }
    if ASCEND_IS_AIV {
      if(cid_ < shared_core_total_num_){
        for (int32_t i = 0; i < single_shared_task_num_; ++i) {
          int32_t task_id = cid_ * single_shared_task_num_ + i;
          runAIV(task_id);
        }
      }
      else{
          for (int32_t i = 0; i < single_unshared_task_num_; ++i) {
              int32_t task_id = (cid_-shared_core_total_num_) * single_unshared_task_num_ + i;
              runAivUnshared(task_id);
          }
        }
      }
  }
//TODO: change flashattention to page attention
  template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M, uint32_t BLOCK_M_UNSHARED>
  __aicore__ inline void XAttentionTlKernel<HEAD_SIZE, BLOCK_N, BLOCK_M, BLOCK_M_UNSHARED>::runAIC(int32_t task_id) {
    auto q_ub = ascend_ub_.GetWithOffset<half>(block_m * head_size_, 0);
    auto q_l1 = ascend_l1_.GetWithOffset<half>(block_m * head_size_, 0);
    auto k_ub = ascend_ub_.GetWithOffset<half>(block_n * head_size_, block_m * head_size_ *sizeof(half));
    auto k_l1 = ascend_l1_.GetWithOffset<half>(block_n * head_size_, block_m * head_size_ *sizeof(half));
    auto acc_s_l0c = ascend_l0c_.GetWithOffset<float>(block_m * block_n, 0);
    auto acc_s_l1 = ascend_l1_.GetWithOffset<half>(block_m * block_n, block_m * head_size_ *sizeof(half));
    auto v_ub = ascend_ub_.GetWithOffset<half>(block_n * head_size_, (block_m * head_size_ + block_n * head_size_) * sizeof(half));
    auto v_l1 = ascend_l1_.GetWithOffset<half>(block_n * head_size_, (block_m * head_size_ + block_m * block_n) * sizeof(half));
    auto acc_o_l0c = ascend_l0c_.GetWithOffset<float>(block_m * head_size_, 0);
    // TODO: Q shape: batch ,beam_size,heads,head_size
    // TODO: Q shape: batch ,heads ,beam_size, head_size
      tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, BLOCK_M>(
        q_ub[0],
          Q_[((tileOffsetQ(task_id)) + (batchBeamBase(task_id)))],
        HEAD_SIZE);
    tl::ascend::copy_ub_to_ub<half, half, BLOCK_M * HEAD_SIZE>(q_l1[0], q_ub[0]);
    for (int32_t k = 0; k < ((kv_len_ + (block_n - 1)) / block_n); ++k) {
      tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, BLOCK_N>(
          k_ub[0],
          K_[((k * block_n * head_size_) + (batchKvBase(task_id)))],
          head_size_);
      tl::ascend::copy_ub_to_ub<half, half, BLOCK_N * HEAD_SIZE>(k_l1[0], k_ub[0]);
      tl::ascend::gemm_v0<half, float, BLOCK_M, BLOCK_N, HEAD_SIZE, false, true>(
          q_l1[0], k_l1[0], acc_s_l0c[0], ascend_l0a_, ascend_l0b_, (bool)1);
      // printf("workspaceSOffset: %d\n", workspaceSOffset(task_id));
      tl::ascend::
          copy_l0c_to_gm<float, float, layout::RowMajor, BLOCK_M, BLOCK_N, BLOCK_M, BLOCK_N>(
              workspace_1_[(workspaceSOffset(task_id))],
              acc_s_l0c[0],
              0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
      AscendC::CrossCoreWaitFlag(1);
      tl::ascend::copy_gm_to_l1<half, BLOCK_M, BLOCK_N, BLOCK_M, BLOCK_N>(
          acc_s_l1[0],
          workspace_2_[(workspaceSOffset(task_id))]);
      tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, BLOCK_N>(
          v_ub[0],
          V_[((k * block_n * head_size_) + (batchKvBase(task_id)))],
          head_size_);
      tl::ascend::copy_ub_to_ub<half, half, BLOCK_N * HEAD_SIZE>(v_l1[0], v_ub[0]);
      tl::ascend::gemm_v0<half, float, BLOCK_M, HEAD_SIZE, BLOCK_N, false, false>(
          acc_s_l1[0], v_l1[0], acc_o_l0c[0], ascend_l0a_, ascend_l0b_, (bool)1);
      tl::ascend::
          copy_l0c_to_gm<float, float, layout::RowMajor, BLOCK_M, HEAD_SIZE, BLOCK_N, HEAD_SIZE>(
              workspace_3_[(workspaceOOffset(task_id))],
              acc_o_l0c[0],
              0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(2);
      AscendC::CrossCoreWaitFlag(3);
    }
  }
  template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M, uint32_t BLOCK_M_UNSHARED>
  __aicore__ inline void XAttentionTlKernel<HEAD_SIZE, BLOCK_N, BLOCK_M, BLOCK_M_UNSHARED>::runAIV(int32_t task_id) {
    auto acc_o = ascend_ub_.GetWithOffset<float>(block_m * head_size_ / 2, 0);
    auto sumexp = ascend_ub_.GetWithOffset<float>(block_m / 2, 65536);
    auto m_i = ascend_ub_.GetWithOffset<float>(block_m / 2, 65664);
    auto acc_s_ub = ascend_ub_.GetWithOffset<float>(block_m / 2 * block_n, 66048);
    auto m_i_prev = ascend_ub_.GetWithOffset<float>(block_m / 2, 74240);
    auto acc_s_ub_ = ascend_ub_.GetWithOffset<float>(block_m / 2 * block_n, 74368);
    auto tmp_ub = ascend_ub_.GetWithOffset<uint8_t>(block_m * block_n * 6, 74368);
    auto sumexp_i_ub = ascend_ub_.GetWithOffset<float>(block_m / 2, 98944);
    auto acc_s_half = ascend_ub_.GetWithOffset<half>(block_m / 2 * block_n, 98944);
    auto acc_o_ub = ascend_ub_.GetWithOffset<float>(block_m / 2 * head_size_, 98944);
    auto acc_o_half = ascend_ub_.GetWithOffset<half>(block_m / 2 * head_size_, 98944);
    AscendC::Duplicate<float>(acc_o[0], 0.000000e+00f, block_m * head_size_ / 2);
    AscendC::Duplicate<float>(sumexp[0], 0.000000e+00f, block_m / 2);
    AscendC::Duplicate<float>(m_i[0], -1073741824, block_m / 2);
    for (int32_t _k = 0; _k < ((kv_len_ + (block_n - 1)) / block_n); ++_k) {
      AscendC::Duplicate<float>(acc_s_ub[0], 0.000000e+00f, block_m / 2 * block_n);
      tl::ascend::copy_ub_to_ub<float, float, BLOCK_M / 2>(m_i_prev[0], m_i[0]);
      AscendC::CrossCoreWaitFlag(0);
      // printf("workspaceSOffset: %d\n", workspaceSOffset(task_id));
      tl::ascend::copy_gm_to_ub<float, BLOCK_N, BLOCK_N, BLOCK_M / 2>(
          acc_s_ub_[0],
          workspace_1_[((workspaceSOffset(cid_)) + (vid_ * block_m / 2 * block_n))],
          block_n);
      AscendC::Add(acc_s_ub[0], acc_s_ub[0], acc_s_ub_[0], block_m / 2 * block_n);
      AscendC::Muls(acc_s_ub[0], acc_s_ub[0], 8.838835e-02f, block_m / 2 * block_n);
      tl::ascend::reduce_max<float, BLOCK_M / 2, BLOCK_N, AscendC::Pattern::Reduce::AR>(
          m_i[0], acc_s_ub[0], tmp_ub[0]);
      AscendC::Max(m_i[0], m_i[0], m_i_prev[0], block_m / 2);
      AscendC::Sub(m_i_prev[0], m_i_prev[0], m_i[0], block_m / 2);
      AscendC::Exp(m_i_prev[0], m_i_prev[0], block_m / 2);
      for (int32_t h_i = 0; h_i < block_m / 2; ++h_i) {
        auto m_i_scalar = m_i.GetValue(h_i);
        AscendC::Adds(
            acc_s_ub[(h_i * block_n)], acc_s_ub[(h_i * block_n)], -m_i_scalar, block_n);
      }
      AscendC::Exp(acc_s_ub[0], acc_s_ub[0], block_m / 2 * block_n);
      tl::ascend::reduce_sum<float, BLOCK_M / 2, BLOCK_N, AscendC::Pattern::Reduce::AR>(
          sumexp_i_ub[0], acc_s_ub[0], tmp_ub[0]);
      AscendC::Mul(sumexp[0], sumexp[0], m_i_prev[0], block_m / 2);
      AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ub[0], block_m / 2);
      for (int32_t h_i_1 = 0; h_i_1 < block_m / 2; ++h_i_1) {
        auto m_i_prev_scalar = m_i_prev.GetValue(h_i_1);
        AscendC::Muls(
            acc_o[(h_i_1 * head_size_)], acc_o[(h_i_1 * head_size_)], m_i_prev_scalar, head_size_);
      }
      tl::ascend::copy_ub_to_ub<half, float, BLOCK_M / 2 * BLOCK_N>(acc_s_half[0], acc_s_ub[0]);
      tl::ascend::copy_ub_to_gm<half, BLOCK_N, BLOCK_N, BLOCK_M / 2>(
          workspace_2_[((workspaceSOffset(cid_)) + (vid_ * block_m / 2 * block_n))],
          acc_s_half[0]);
      AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
      AscendC::CrossCoreWaitFlag(2);
      tl::ascend::copy_gm_to_ub<float, HEAD_SIZE, HEAD_SIZE, BLOCK_M / 2>(
          acc_o_ub[0],
          workspace_3_[((workspaceOOffset(cid_)) + (vid_ * block_m / 2 * head_size_))],
          head_size_);
      AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], block_m / 2 * head_size_);
      AscendC::CrossCoreSetFlag<0x2, PIPE_V>(3);
    }
    tl::ascend::copy_ub_to_gm<float, BLOCK_M / 2, BLOCK_M / 2, 1>(
      Shared_exp_[(expOffsetShared(task_id) + (vid_ * block_m / 2))],
      sumexp[0]);
    tl::ascend::copy_ub_to_gm<float, BLOCK_M / 2, BLOCK_M / 2, 1>(
      Shared_max_[(expOffsetShared(task_id) + (vid_ * block_m / 2))],
      m_i[0]);

    for (int32_t h_i_2 = 0; h_i_2 < block_m / 2; ++h_i_2) {
      auto sumexp_scalar = 1.0f / sumexp.GetValue(h_i_2);
      AscendC::Muls(
          acc_o[(h_i_2 * head_size_)], acc_o[(h_i_2 * head_size_)], sumexp_scalar, head_size_);
    }
    tl::ascend::copy_ub_to_ub<half, float, BLOCK_M / 2 * HEAD_SIZE>(acc_o_half[0], acc_o[0]);
    tl::ascend::copy_ub_to_gm<half, HEAD_SIZE, HEAD_SIZE, BLOCK_M / 2>(
        Output_shared_[(
            ((tileOffsetQ(task_id)) + (vid_ * block_m / 2 * head_size_)) +
            (batchBeamBase(task_id)))],
        acc_o_half[0]);
  }
  template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M, uint32_t BLOCK_M_UNSHARED>
  __aicore__ inline void XAttentionTlKernel<HEAD_SIZE, BLOCK_N, BLOCK_M, BLOCK_M_UNSHARED>::runAicUnshared(int32_t task_id) {
    auto q_ub = ascend_ub_.GetWithOffset<half>(block_m_unshared * head_size_, 0);
    auto q_l1 = ascend_l1_.GetWithOffset<half>(block_m_unshared * head_size_, 0);
    auto k_ub = ascend_ub_.GetWithOffset<half>(block_n * head_size_, block_m_unshared * head_size_ *sizeof(half));
    auto k_l1 = ascend_l1_.GetWithOffset<half>(block_n * head_size_, block_m_unshared * head_size_ *sizeof(half));
    auto acc_s_l0c = ascend_l0c_.GetWithOffset<float>(block_m_unshared * block_n, 0);
    auto acc_s_l1 = ascend_l1_.GetWithOffset<half>(block_m_unshared * block_n, block_m_unshared * head_size_ *sizeof(half));
    auto v_ub = ascend_ub_.GetWithOffset<half>(block_n * head_size_, (block_m_unshared * head_size_ + block_n * head_size_) * sizeof(half));
    auto v_l1 = ascend_l1_.GetWithOffset<half>(block_n * head_size_, (block_m_unshared * head_size_ + block_m_unshared * block_n) * sizeof(half));
    auto acc_o_l0c = ascend_l0c_.GetWithOffset<float>(block_m_unshared * head_size_, 0);
      // tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, BLOCK_M_UNSHARED>(
      //   q_ub[0],
      //   Q_unshared_[((tileOffsetQUnshared(task_id)) + (batchBeamBaseUnshared(task_id)))],
      //   head_size_);
      tl::ascend::copy_gm_to_l1<half, BLOCK_M_UNSHARED, HEAD_SIZE, BLOCK_M_UNSHARED, HEAD_SIZE>(
          q_l1[0],
          Q_unshared_[((tileOffsetQUnshared(task_id)) + (batchBeamBaseUnshared(task_id)))]);
      // AscendC::DataCopy(q_l1[0], Q_unshared_[((tileOffsetQUnshared(task_id)) + (batchBeamBaseUnshared(task_id)))], head_size_);
      
    // tl::ascend::copy_ub_to_ub<half, half, BLOCK_M_UNSHARED * HEAD_SIZE>(q_l1[0], q_ub[0]);
    for (int32_t k = 0; k < ((unshared_k_len_ + (block_n - 1)) / block_n); ++k) {
      tl::ascend::copy_gm_to_l1<half, BLOCK_N, HEAD_SIZE, BLOCK_N, HEAD_SIZE>(
          k_l1[0],
          K_unshared_[((k * block_n * head_size_) + (batchKvBaseUnshared(task_id)))]);
      // tl::ascend::copy_ub_to_ub<half, half, BLOCK_N * HEAD_SIZE>(k_l1[0], k_ub[0]);
      tl::ascend::gemm_v0<half, float, BLOCK_M_UNSHARED, BLOCK_N, HEAD_SIZE, false, true>(
          q_l1[0], k_l1[0], acc_s_l0c[0], ascend_l0a_, ascend_l0b_, (bool)1);
      tl::ascend::
          copy_l0c_to_gm<float, float, layout::RowMajor, BLOCK_M_UNSHARED, BLOCK_N, BLOCK_M_UNSHARED, BLOCK_N>(
              workspace_1_unshared_[(workspaceSOffsetUnshared(task_id))],
              acc_s_l0c[0],
              0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
      AscendC::CrossCoreWaitFlag(1);
      tl::ascend::copy_gm_to_l1<half, BLOCK_M_UNSHARED, BLOCK_N, BLOCK_M_UNSHARED, BLOCK_N>(
          acc_s_l1[0],
          workspace_2_unshared_[(workspaceSOffsetUnshared(task_id))]);
      tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, BLOCK_N>(
          v_l1[0],
          V_unshared_[((k * block_n * head_size_) + (batchKvBaseUnshared(task_id)))],
          head_size_);
      // tl::ascend::copy_ub_to_ub<half, half, BLOCK_N * HEAD_SIZE>(v_l1[0], v_ub[0]);
      tl::ascend::gemm_v0<half, float, BLOCK_M_UNSHARED, HEAD_SIZE, BLOCK_N, false, false>(
          acc_s_l1[0], v_l1[0], acc_o_l0c[0], ascend_l0a_, ascend_l0b_, (bool)1);
      tl::ascend::
          copy_l0c_to_gm<float, float, layout::RowMajor, BLOCK_M_UNSHARED, HEAD_SIZE, BLOCK_N, HEAD_SIZE>(
              workspace_3_unshared_[(workspaceOOffsetUnshared(task_id))],
              acc_o_l0c[0],
              0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(2);
      AscendC::CrossCoreWaitFlag(3);
    }
  }

  template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M, uint32_t BLOCK_M_UNSHARED>
  __aicore__ inline void XAttentionTlKernel<HEAD_SIZE, BLOCK_N, BLOCK_M, BLOCK_M_UNSHARED>::runAivUnshared(int32_t task_id) {
    auto acc_o = ascend_ub_.GetWithOffset<float>(block_m_unshared * head_size_, 0);
    auto sumexp = ascend_ub_.GetWithOffset<float>(block_m_unshared, 65536);
    auto m_i = ascend_ub_.GetWithOffset<float>(block_m_unshared, 65664);
    auto acc_s_ub = ascend_ub_.GetWithOffset<float>(block_m_unshared * block_n, 66048);
    auto m_i_prev = ascend_ub_.GetWithOffset<float>(block_m_unshared, 74240);
    auto acc_s_ub_ = ascend_ub_.GetWithOffset<float>(block_m_unshared * block_n, 74368);
    auto tmp_ub = ascend_ub_.GetWithOffset<uint8_t>(block_m_unshared * block_n * 6, 74368);
    auto sumexp_i_ub = ascend_ub_.GetWithOffset<float>(block_m_unshared, 98944);
    auto acc_s_half = ascend_ub_.GetWithOffset<half>(block_m_unshared * block_n, 98944);
    auto acc_o_ub = ascend_ub_.GetWithOffset<float>(block_m_unshared * head_size_, 98944);
    auto acc_o_half = ascend_ub_.GetWithOffset<half>(block_m_unshared * head_size_, 98944);
    AscendC::Duplicate<float>(acc_o[0], 0.000000e+00f, block_m_unshared * head_size_);
    AscendC::Duplicate<float>(sumexp[0], 0.000000e+00f, block_m_unshared);
    AscendC::Duplicate<float>(m_i[0], -1073741824, block_m_unshared);
    for (int32_t _k = 0; _k < ((unshared_k_len_ + (block_n - 1)) / block_n); ++_k) {
      AscendC::Duplicate<float>(acc_s_ub[0], 0.000000e+00f, block_m_unshared * block_n);
      m_i_prev = m_i;
      // tl::ascend::copy_ub_to_ub<float, float, BLOCK_M_UNSHARED>(m_i_prev[0], m_i[0]);
      AscendC::CrossCoreWaitFlag(0);
      tl::ascend::copy_gm_to_ub<float, BLOCK_N, BLOCK_N, BLOCK_M_UNSHARED>(
          acc_s_ub_[0],
          workspace_1_unshared_[(workspaceSOffsetUnshared(task_id))],
          block_n);
      AscendC::Add(acc_s_ub[0], acc_s_ub[0], acc_s_ub_[0], block_m_unshared * block_n);
      AscendC::Muls(acc_s_ub[0], acc_s_ub[0], 8.838835e-02f, block_m_unshared * block_n);
      tl::ascend::reduce_max<float, BLOCK_M_UNSHARED, BLOCK_N, AscendC::Pattern::Reduce::AR>(
          m_i[0], acc_s_ub[0], tmp_ub[0]);
      AscendC::Max(m_i[0], m_i[0], m_i_prev[0], block_m_unshared);
      AscendC::Sub(m_i_prev[0], m_i_prev[0], m_i[0], block_m_unshared);
      AscendC::Exp(m_i_prev[0], m_i_prev[0], block_m_unshared);
      for (int32_t h_i = 0; h_i < block_m_unshared; ++h_i) {
        auto m_i_scalar = m_i.GetValue(h_i);
        AscendC::Adds(
            acc_s_ub[(h_i * block_n)], acc_s_ub[(h_i * block_n)], -m_i_scalar, block_n);
      }
      AscendC::Exp(acc_s_ub[0], acc_s_ub[0], block_m_unshared * block_n);
      tl::ascend::reduce_sum<float, BLOCK_M_UNSHARED, BLOCK_N, AscendC::Pattern::Reduce::AR>(
          sumexp_i_ub[0], acc_s_ub[0], tmp_ub[0]);
      AscendC::Mul(sumexp[0], sumexp[0], m_i_prev[0], block_m_unshared);
      AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ub[0], block_m_unshared);
      for (int32_t h_i_1 = 0; h_i_1 < block_m_unshared; ++h_i_1) {
        auto m_i_prev_scalar = m_i_prev.GetValue(h_i_1);
        AscendC::Muls(
            acc_o[(h_i_1 * head_size_)], acc_o[(h_i_1 * head_size_)], m_i_prev_scalar, head_size_);
      }
      tl::ascend::copy_ub_to_ub<half, float, BLOCK_M_UNSHARED * BLOCK_N>(acc_s_half[0], acc_s_ub[0]);
      tl::ascend::copy_ub_to_gm<half, BLOCK_N, BLOCK_N, BLOCK_M_UNSHARED>(
          workspace_2_unshared_[(workspaceSOffsetUnshared(task_id))],
          acc_s_half[0]);
      AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
      AscendC::CrossCoreWaitFlag(2);
      tl::ascend::copy_gm_to_ub<float, HEAD_SIZE, HEAD_SIZE, BLOCK_M_UNSHARED>(
          acc_o_ub[0],
          workspace_3_unshared_[(workspaceOOffsetUnshared(task_id))],
          head_size_);
      AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], block_m_unshared * head_size_);
      AscendC::CrossCoreSetFlag<0x2, PIPE_V>(3);
    }
    tl::ascend::copy_ub_to_gm<float, BLOCK_M_UNSHARED, BLOCK_M_UNSHARED, 1>(
        Unshared_exp_[(expOffsetShared(task_id))],
        sumexp[0]);
    tl::ascend::copy_ub_to_gm<float, BLOCK_M_UNSHARED, BLOCK_M_UNSHARED, 1>(
        Unshared_max_[(expOffsetShared(task_id))],
        m_i[0]);
    for (int32_t h_i_2 = 0; h_i_2 < block_m_unshared; ++h_i_2) {
      auto sumexp_scalar = 1.0f / sumexp.GetValue(h_i_2);
      AscendC::Muls(
          acc_o[(h_i_2 * head_size_)], acc_o[(h_i_2 * head_size_)], sumexp_scalar, head_size_);
    }
    tl::ascend::copy_ub_to_ub<half, float, BLOCK_M_UNSHARED * HEAD_SIZE>(acc_o_half[0], acc_o[0]);
    tl::ascend::copy_ub_to_gm<half, HEAD_SIZE, HEAD_SIZE, BLOCK_M_UNSHARED>(
        Output_[(
            ((tileOffsetQ(cid_)) + (block_m_unshared * head_size_)) +
            (batchBeamBase(cid_)))],
        acc_o_half[0]);
  }
  template<uint32_t HEAD_SIZE, uint32_t HEAD_NUM_COMBINE>
  __aicore__ inline void XAttentionTlCombine<HEAD_SIZE, HEAD_NUM_COMBINE>::Init(
    GM_ADDR output_shared_handle,
    GM_ADDR output_unshared_handle,
    GM_ADDR Output_handle,
    GM_ADDR shared_exp_handle,
    GM_ADDR unshared_exp_handle,
    GM_ADDR shared_max_handle,
    GM_ADDR unshared_max_handle,
    int64_t batch_size,
    int64_t q_len,
    int64_t kv_len,
    int64_t unshared_k_len,
    int64_t unshared_q_len,
    int64_t tiles_q,
    int64_t head_size,
    int64_t head_num,
    int64_t beam_size,
    int64_t core_num
  ){
    //unshared shape: batch,beam_size,heads
    //shared shape: batch,heads,tiles_q,block_m
    //shared shape: batch,heads,beam_size
    // reshape shared shape: batch,beam_size,heads
    // output_shared shape: batch,heads,beam_size,head_size
    // output_unshared shape: batch,beam_size,heads,1,head_size
    Output_shared_.SetGlobalBuffer((__gm__ half*)output_shared_handle);
    Output_unshared_.SetGlobalBuffer((__gm__ half*)output_unshared_handle);
    Shared_exp_.SetGlobalBuffer((__gm__ float*)shared_exp_handle);
    Unshared_exp_.SetGlobalBuffer((__gm__ float*)unshared_exp_handle);
    Shared_max_.SetGlobalBuffer((__gm__ float*)shared_max_handle);
    Unshared_max_.SetGlobalBuffer((__gm__ float*)unshared_max_handle);
    Output_.SetGlobalBuffer((__gm__ half*)Output_handle);
    batch_ = batch_size;
    q_len_ = q_len;
    kv_len_ = kv_len;
    unshared_k_len_ = unshared_k_len;
    unshared_q_len_ = unshared_q_len;
    tiles_q_ = tiles_q;
    head_size_ = head_size;
    head_num_ = head_num;
    beam_size_ = beam_size;
    core_num_ = core_num;
    task_num_ = batch_ * beam_size_;
    single_task_num_ = task_num_ / core_num_;
    pipe.InitBuffer(ascend_ub_, 196352);

  }
  template<uint32_t HEAD_SIZE, uint32_t HEAD_NUM_COMBINE>
  __aicore__ inline void XAttentionTlCombine<HEAD_SIZE, HEAD_NUM_COMBINE>::run(){
    cid_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
      for (int32_t i = 0; i < single_task_num_; ++i) {
        int32_t task_id = cid_ * single_task_num_ + i;
        runAIV(task_id);
      }
    }
  }
  template<uint32_t HEAD_SIZE, uint32_t HEAD_NUM_COMBINE>
  __aicore__ inline void XAttentionTlCombine<HEAD_SIZE, HEAD_NUM_COMBINE>::runAIV(int32_t task_id) {
    //unshared shape: batch,beam_size,heads
    //shared shape: batch,heads,tiles_q,block_m
    //shared shape: batch,heads,beam_size
    // reshape shared shape: batch,beam_size,heads
    // output_shared shape: batch,heads,beam_size,head_size
    // output_unshared shape: batch,beam_size,heads,1,head_size
    auto shared_ub = ascend_ub_.GetWithOffset<float>(head_size_, 0);
    auto unshared_ub = ascend_ub_.GetWithOffset<float>(head_size_, head_size_);
    auto shared_max = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 2);
    auto unshared_max = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 3);
    auto shared_exp = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 4);
    auto unshared_exp = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 5);
    auto max_final = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 6);
    auto max_l1 = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 7);
    auto max_l2 = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 8);
    auto final_exp = ascend_ub_.GetWithOffset<float>(head_num_, head_size_ * 9);
    auto shared_half = ascend_ub_.GetWithOffset<half>(HEAD_SIZE, head_size_ * 10);
    auto unshared_half = ascend_ub_.GetWithOffset<half>(HEAD_SIZE, head_size_ * 10 + HEAD_SIZE * sizeof(half));
    tl::ascend::copy_gm_to_ub<float, HEAD_NUM_COMBINE, HEAD_NUM_COMBINE, 1 >(
      shared_max[0],
      Shared_max_[(task_id * head_num_)],
      head_num_);
    tl::ascend::copy_gm_to_ub<float, HEAD_NUM_COMBINE, HEAD_NUM_COMBINE, 1 >(
      unshared_max[0],
      Unshared_max_[(task_id * head_num_)],
      head_num_);
    tl::ascend::copy_gm_to_ub<float, HEAD_NUM_COMBINE, HEAD_NUM_COMBINE, 1 >(
      shared_exp[0],
      Shared_exp_[(task_id * head_num_)],
      head_num_);
    tl::ascend::copy_gm_to_ub<float, HEAD_NUM_COMBINE, HEAD_NUM_COMBINE, 1 >(
      unshared_exp[0],
      Unshared_exp_[(task_id * head_num_)],
      head_num_);
    
    AscendC::Max(max_final[0], shared_max[0], unshared_max[0], head_num_);
    AscendC::Sub(max_l1[0], shared_max[0], max_final[0], head_num_);
    AscendC::Sub(max_l2[0], unshared_max[0], max_final[0], head_num_);
    AscendC::Exp(max_l1[0], max_l1[0], head_num_);
    AscendC::Exp(max_l2[0], max_l2[0], head_num_);
    AscendC::Mul(shared_exp[0], shared_exp[0], max_l1[0], head_num_);
    AscendC::Mul(unshared_exp[0], unshared_exp[0], max_l2[0], head_num_);
    AscendC::Add(final_exp[0], shared_exp[0], unshared_exp[0], head_num_);
    for (int32_t h_i = 0; h_i < head_num_; ++h_i) {
      auto final_exp_scalar = final_exp.GetValue(h_i);
      auto m_l1_scalar = max_l1.GetValue(h_i);
      auto m_l2_scalar = max_l2.GetValue(h_i);
      tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, 1 >(
        shared_half[0],
        Output_shared_[(task_id *head_num_ * head_size_ + h_i * head_size_)],
        head_size_);
      tl::ascend::copy_gm_to_ub<half, HEAD_SIZE, HEAD_SIZE, 1 >(
        unshared_half[0],
        Output_unshared_[(task_id * head_num_ * head_size_ + h_i * head_size_)],
        head_size_);
      tl::ascend::copy_ub_to_ub<float, half, HEAD_SIZE>(shared_ub[0], shared_half[0]);
      tl::ascend::copy_ub_to_ub<float, half, HEAD_SIZE>(unshared_ub[0], unshared_half[0]);
      AscendC::Muls(shared_ub[0], shared_ub[0], 1.0f / final_exp_scalar, head_size_);
      AscendC::Muls(unshared_ub[0], unshared_ub[0], 1.0f / final_exp_scalar, head_size_);
      AscendC::Muls(shared_ub[0], shared_ub[0], m_l1_scalar, head_size_);
      AscendC::Muls(unshared_ub[0], unshared_ub[0], m_l2_scalar, head_size_);
      AscendC::Add(shared_ub[0], shared_ub[0], unshared_ub[0], head_size_);
      tl::ascend::copy_ub_to_ub<half, float, HEAD_SIZE>(shared_half[0], shared_ub[0]);
      tl::ascend::copy_ub_to_gm<half, HEAD_SIZE, HEAD_SIZE, 1>(
        Output_[(task_id * head_num_ * head_size_ + h_i * head_size_)],
        shared_half[0]);
    }
  }
extern "C" __global__ __aicore__ void x_attention_tl(
    GM_ADDR Q_handle,
    GM_ADDR K_handle,
    GM_ADDR V_handle,
    GM_ADDR q_unshared_handle,
    GM_ADDR unshared_key_handle,
    GM_ADDR unshared_value_handle,
    GM_ADDR output_shared_handle,
    GM_ADDR output_unshared_handle,
    GM_ADDR Output_handle,
    GM_ADDR shared_exp_handle,
    GM_ADDR unshared_exp_handle,
    GM_ADDR shared_max_handle,
    GM_ADDR unshared_max_handle,
    GM_ADDR workspace,
    GM_ADDR tiling
    ) {
  GET_TILING_DATA(tiling_data, tiling);
  switch (tiling_data.head_size) {
    case 128: {
      XAttentionTlKernel<128, 64, 64, 1> op;
      op.Init(
      Q_handle,
      K_handle,
      V_handle,
      q_unshared_handle,
      unshared_key_handle,
      unshared_value_handle,
      output_shared_handle,
      output_unshared_handle,
      shared_exp_handle,
      unshared_exp_handle,
      shared_max_handle,
      unshared_max_handle,
      Output_handle,
      workspace,
      tiling_data.batch_size,
      tiling_data.beam_size,
      tiling_data.shared_k_length,
      tiling_data.unshared_k_length,
      tiling_data.q_length,
      tiling_data.head_size,
      tiling_data.num_heads
    );
      op.run();
      // SyncAll();
      // // 简化：固定 head_num=8 的合并核，避免读取不存在的 head_num/tiles_q 字段
      // {
      //     XAttentionTlCombine<128, 8> op_combine;
      //     op_combine.Init(output_shared_handle,
      //       output_unshared_handle,
      //       Output_handle,
      //       shared_exp_handle,
      //       unshared_exp_handle,
      //       shared_max_handle,
      //       unshared_max_handle,
      //       tiling_data.batch_size,
      //       tiling_data.q_length,
      //       tiling_data.shared_k_length,
      //       tiling_data.unshared_k_length,
      //       tiling_data.unshared_k_length,
      //       0,
      //       tiling_data.head_size,
      //       tiling_data.num_heads,
      //       tiling_data.beam_size,
      //       tiling_data.core_num);
      //     op_combine.run();
      // }
      break;
    }
    default: {
      XAttentionTlKernel<128, 64, 64, 1> op;
      op.Init(
      Q_handle,
      K_handle,
      V_handle,
      q_unshared_handle,
      unshared_key_handle,
      unshared_value_handle,
      output_shared_handle,
      output_unshared_handle,
      shared_exp_handle,
      unshared_exp_handle,
      shared_max_handle,
      unshared_max_handle,
      Output_handle,
      workspace,
      tiling_data.batch_size,
      tiling_data.beam_size,
      tiling_data.shared_k_length,
      tiling_data.unshared_k_length,
      tiling_data.q_length,
      tiling_data.head_size,
      tiling_data.num_heads
    );
      op.run();
      // SyncAll();
      // XAttentionTlCombine<128, 8> op_combine;
      // op_combine.Init(output_shared_handle,
      //   output_unshared_handle,
      //   Output_handle,
      //   shared_exp_handle,
      //   unshared_exp_handle,
      //   shared_max_handle,
      //   unshared_max_handle,
      //   tiling_data.batch_size,
      //   tiling_data.q_length,
      //   tiling_data.shared_k_length,
      //   tiling_data.unshared_k_length,
      //   tiling_data.unshared_k_length,
      //   0,
      //   tiling_data.head_size,
      //   tiling_data.num_heads,
      //   tiling_data.beam_size,
      //   tiling_data.core_num);
      // op_combine.run();
      break;
    }
  }
}