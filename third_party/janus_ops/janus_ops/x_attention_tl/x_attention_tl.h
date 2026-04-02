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
#include "acl/acl.h"
#include "common.h"

using namespace AscendC;


// static constexpr int HEAD_SIZE = 128; 
// static constexpr int BLOCK_N = 64;
// static constexpr int BLOCK_M = 64;
// static constexpr int SUBBLOCK_M = BLOCK_M / 2; // 32

// // 常用派生尺寸（元素数）
// static constexpr int TILE_Q_ELEMS = BLOCK_M * BLOCK_K;        // 64*128 = 8192
// static constexpr int TILE_K_ELEMS = BLOCK_N * BLOCK_K;        // 128*128 = 16384
// static constexpr int ACC_S_ELEMS  = BLOCK_M * BLOCK_M;        // 64*64 = 4096
// static constexpr int ACC_O_ELEMS  = BLOCK_M * BLOCK_N;        // 64*128 = 8192
// static constexpr int ACC_O_SUB_ELEMS = SUBBLOCK_M * BLOCK_K;  // 32*128 = 4096
// static constexpr int ACC_S_SUB_ELEMS = SUBBLOCK_M * BLOCK_M;  // 32*64  = 2048

// // L1 缓冲区偏移（以元素计）
// static constexpr int L1_OFFSET_K = TILE_K_ELEMS;                // 16384
// static constexpr int L1_OFFSET_V = TILE_K_ELEMS + TILE_Q_ELEMS; // 24576

// 注意：本实现参考 attention_test 的类风格，使用 inline 方法并不持有句柄

// 将 XAttentionKernel 的完整声明移动到头文件，具体实现见 x_attention.cpp
template<uint32_t HEAD_SIZE, uint32_t BLOCK_N, uint32_t BLOCK_M,uint32_t BLOCK_M_UNSHARED>
class XAttentionTlKernel {
 public:
  __aicore__ inline XAttentionTlKernel() {}
  __aicore__ inline void Init(
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
      int64_t head_num);

  __aicore__ inline void run();

 private:
  // 私有计算核路径（无参，使用成员）
  __aicore__ inline void runAIC(int32_t task_id);
  __aicore__ inline void runAIV(int32_t task_id);
  __aicore__ inline void runAicUnshared(int32_t task_id);
  __aicore__ inline void runAivUnshared(int32_t task_id);
  // 缓存 tiling 中常用的标量，便于在成员函数使用
  int64_t batch_;
  int64_t q_len_;
  int64_t kv_len_;
  int64_t unshared_k_len_;
  int64_t unshared_q_len_;
  int64_t tiles_q_;
  int64_t head_size_;
  int64_t head_num_;
  int64_t beam_size_;
  int64_t block_m;
  int64_t block_n;
  int64_t block_m_unshared;
  int64_t tatal_shared_tasks_;
  int64_t tatal_unshared_tasks_;
  int64_t shared_core_total_num_;
  int64_t unshared_core_total_num_;
  int64_t single_shared_task_num_;
  int64_t single_unshared_task_num_;
  int64_t workspace_1_size_;
  int64_t workspace_2_size_;
  int64_t workspace_3_size_;
  int64_t workspace_1_unshared_size_;
  int64_t workspace_2_unshared_size_;
  int64_t workspace_3_unshared_size_;
  // 小工具：统一计算各类索引，减少重复表达式
  // cid = (beam_size/block_m) * batch_size * head_num
  // kv cache shape: bs,head_num,kv_length,head_size
  __aicore__ inline int64_t tileOffsetQ(int64_t cid) const {
    const int64_t tile_index = (cid % tiles_q_);
    return (tile_index * block_m * head_size_);
  }
  // unshared stage: 每个任务处理 1 行（block_m_unshared=1），无 tile 概念
  __aicore__ inline int64_t tileOffsetQUnshared(int64_t /*cid*/) const {
    return 0;
  }
  // unshared stage:
  // cid = beam_size * batch_size * head_num
  __aicore__ inline int64_t batchBeamBase(int64_t cid) const {
    const int64_t tile_group  = (cid / tiles_q_);
    const int64_t batch_index = ((tile_group / head_num_) % batch_);
    const int64_t head_index  = (tile_group % head_num_);
    const int64_t group_index = (batch_index * head_num_) + head_index;
    return (group_index * q_len_ * head_size_);
  }

  __aicore__ inline int64_t batchBeamBaseUnshared(int64_t cid) const {
    const int64_t head_index  = (cid % head_num_);
    const int64_t tmp         = (cid / head_num_);
    const int64_t beam_index  = (tmp % beam_size_);
    const int64_t batch_index = ((tmp / beam_size_) % batch_);
    const int64_t group_index = (batch_index * beam_size_ * head_num_) + (beam_index * head_num_) + head_index;
    return (group_index * unshared_q_len_ * head_size_);
  }

  __aicore__ inline int64_t batchKvBase(int64_t cid) const {
    const int64_t tile_group  = (cid / tiles_q_);
    const int64_t batch_index = ((tile_group / head_num_) % batch_);
    const int64_t head_index  = (tile_group % head_num_);
    const int64_t group_index = (batch_index * head_num_) + head_index;
    return (group_index * kv_len_ * head_size_);
  }
  // unshared shape:  bs,beam_size,head_num,kv_length,head_size
  __aicore__ inline int64_t batchKvBaseUnshared(int64_t cid) const{
    const int64_t head_index  = (cid % head_num_);
    const int64_t tmp         = (cid / head_num_);
    const int64_t beam_index  = (tmp % beam_size_);
    const int64_t batch_index = ((tmp / beam_size_) % batch_);
    const int64_t group_index = (batch_index * beam_size_ * head_num_) + (beam_index * head_num_) + head_index;
    return (group_index * unshared_k_len_ * head_size_);
  }
  // shared shape:  bs,head_num,q_length,head_size
  // shared shape: batch, heads, tiles_q, block_M, block_N
  __aicore__ inline int64_t workspaceSOffset(int64_t cid) const {
    const int64_t tile_group  = (cid / tiles_q_);
    const int64_t tile_index  = (cid % tiles_q_);
    const int64_t batch_index = ((tile_group / head_num_) % batch_);
    const int64_t head_index  = (tile_group % head_num_);
    const int64_t group_index = (batch_index * head_num_) + head_index;
    const int64_t base_offset = (group_index * tiles_q_) * block_m * block_n;
    const int64_t tile_offset = tile_index * block_m * block_n;
    return (base_offset + tile_offset);
  }
  //unshared shape: batch, beam_size, heads, block_m_unshared, block_N
  __aicore__ inline int64_t workspaceSOffsetUnshared(int64_t cid) const {
    const int64_t head_index  = (cid % head_num_);
    const int64_t tmp         = (cid / head_num_);
    const int64_t beam_index  = (tmp % beam_size_);
    const int64_t batch_index = ((tmp / beam_size_) % batch_);
    const int64_t group_index = (batch_index * beam_size_ * head_num_) + (beam_index * head_num_) + head_index;
    return (group_index * block_m_unshared * block_n);
  }
  __aicore__ inline int64_t workspaceOOffset(int64_t cid) const {
    const int64_t tile_group  = (cid / tiles_q_);
    const int64_t tile_index  = (cid % tiles_q_);
    const int64_t batch_index = ((tile_group / head_num_) % batch_);
    const int64_t head_index  = (tile_group % head_num_);
    const int64_t group_index = (batch_index * head_num_) + head_index;
    const int64_t base_offset = (group_index * tiles_q_) * block_m * head_size_;
    const int64_t tile_offset = tile_index * block_m * head_size_;
    return (base_offset + tile_offset);
  }
  //unshared shape: batch, beam_size, heads, block_m_unshared, head_size
  __aicore__ inline int64_t workspaceOOffsetUnshared(int64_t cid) const {
    const int64_t head_index  = (cid % head_num_);
    const int64_t tmp         = (cid / head_num_);
    const int64_t beam_index  = (tmp % beam_size_);
    const int64_t batch_index = ((tmp / beam_size_) % batch_);
    const int64_t group_index = (batch_index * beam_size_ * head_num_) + (beam_index * head_num_) + head_index;
    return (group_index * block_m_unshared * head_size_);
  }

  __aicore__ inline int64_t expOffsetShared(int64_t cid) const {
    const int64_t tile_group  = (cid / tiles_q_);
    const int64_t tile_index  = (cid % tiles_q_);
    const int64_t batch_index = ((tile_group / head_num_) % batch_);
    const int64_t head_index  = (tile_group % head_num_);
    const int64_t group_index = (batch_index * head_num_) + head_index;
    return (group_index * block_m);
  }

  __aicore__ inline int64_t expOffsetUnshared(int64_t cid) const {
    const int64_t head_index  = (cid % head_num_);
    const int64_t tmp         = (cid / head_num_);
    const int64_t beam_index  = (tmp % beam_size_);
    const int64_t batch_index = ((tmp / beam_size_) % batch_);
    const int64_t group_index = (batch_index * beam_size_ * head_num_) + (beam_index * head_num_) + head_index;
    return (group_index * block_m_unshared);
  }

  // 运行期索引标识
  decltype(AscendC::GetBlockIdx()) cid_;
  decltype(AscendC::GetSubBlockIdx()) vid_;

  // 全局张量成员
  AscendC::GlobalTensor<half> Q_;
  AscendC::GlobalTensor<half> K_;
  AscendC::GlobalTensor<half> V_;
  AscendC::GlobalTensor<half> Q_unshared_;
  AscendC::GlobalTensor<half> K_unshared_;
  AscendC::GlobalTensor<half> V_unshared_;
  AscendC::GlobalTensor<float> Shared_exp_;
  AscendC::GlobalTensor<float> Unshared_exp_;
  AscendC::GlobalTensor<float> Shared_max_;
  AscendC::GlobalTensor<float> Unshared_max_;
  AscendC::GlobalTensor<half> Output_shared_;
  AscendC::GlobalTensor<half> Output_unshared_;
  AscendC::GlobalTensor<half> Output_;
  AscendC::GlobalTensor<float> workspace_1_;
  AscendC::GlobalTensor<half> workspace_2_;
  AscendC::GlobalTensor<float> workspace_3_;
  AscendC::GlobalTensor<float> workspace_1_unshared_;
  AscendC::GlobalTensor<half> workspace_2_unshared_;
  AscendC::GlobalTensor<float> workspace_3_unshared_;
  

  // 片上缓冲成员
  TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a_;
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b_;
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1_;
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub_;
};

template<uint32_t HEAD_SIZE, uint32_t HEAD_NUM_COMBINE>
class XAttentionTlCombine {
    public:
     __aicore__ inline XAttentionTlCombine() {}
     __aicore__ inline void Init(
      GM_ADDR output_shared_handle,
      GM_ADDR output_unshared_handle,
      GM_ADDR Output_handle,
      GM_ADDR shared_exp_handle,
      GM_ADDR unshared_exp_handle,
      GM_ADDR shared_max_handle,
      GM_ADDR unshared_max_handle,
      int64_t batch,
      int64_t q_len,
      int64_t kv_len,
      int64_t unshared_k_len,
      int64_t unshared_q_len,
      int64_t tiles_q,
      int64_t head_size,
      int64_t head_num,
      int64_t beam_size,
      int64_t core_num
    );
     __aicore__ inline void run();
     __aicore__ inline void runAIV(int32_t task_id);
    private:
    decltype(AscendC::GetBlockIdx()) cid_;
    int64_t task_num_;
    int64_t batch_;
    int64_t q_len_;
    int64_t kv_len_;
    int64_t unshared_k_len_;
    int64_t unshared_q_len_;
    int64_t tiles_q_;
    int64_t head_size_;
    int64_t head_num_;
    int64_t beam_size_;
    int64_t single_task_num_;
    int64_t core_num_;
    AscendC::GlobalTensor<half> Output_shared_;
    AscendC::GlobalTensor<half> Output_unshared_;
    AscendC::GlobalTensor<float> Shared_exp_;
    AscendC::GlobalTensor<float> Unshared_exp_;
    AscendC::GlobalTensor<float> Shared_max_;
    AscendC::GlobalTensor<float> Unshared_max_;
    AscendC::GlobalTensor<half> Output_;
    TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub_;
};