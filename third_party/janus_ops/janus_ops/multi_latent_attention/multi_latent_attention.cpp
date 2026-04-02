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
#include "multi_latent_attention.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void multi_latent_attention(GM_ADDR query, GM_ADDR queryRope, GM_ADDR kvCache,
                                                             GM_ADDR kvCacheRope, GM_ADDR block_tables,
                                                             GM_ADDR contextLens, GM_ADDR mask, GM_ADDR qSeqlen,
                                                             GM_ADDR qkDescale, GM_ADDR pvDescale, GM_ADDR attenOut,
                                                             GM_ADDR lseOut, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    __gm__ uint64_t *workspaceParam = reinterpret_cast<__gm__ uint64_t *>(tiling);
    GM_ADDR s_gm = usrWorkspace;
    GM_ADDR s_rope_out_gm = s_gm + workspaceParam[0];
    GM_ADDR p_gm = s_rope_out_gm + workspaceParam[1];
    GM_ADDR o_tmp_gm = p_gm + workspaceParam[2];
    GM_ADDR go_gm = o_tmp_gm + workspaceParam[3];
    GM_ADDR tmp_gm = go_gm + workspaceParam[4];
    
    GM_ADDR q_gm = query;
    GM_ADDR q_rope_gm = queryRope;
    GM_ADDR ctkv_gm = kvCache;
    GM_ADDR ctkv_rope_gm = kvCacheRope;
    GM_ADDR block_tables_gm = block_tables;
    GM_ADDR mask_gm = mask;
    GM_ADDR deq_qk_gm = qkDescale;
    GM_ADDR deq_pv_gm = pvDescale;
    GM_ADDR o_gm = attenOut;
    GM_ADDR lse_gm = lseOut;
    GM_ADDR tiling_para_gm = tiling + sizeof(uint64_t) * 6;
    
    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_VEC__
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
#elif __DAV_C220_CUBE__
    SetPadding<uint64_t>(0);
    SetNdpara(1, 0, 0);
#endif
    if(TILING_KEY_IS(0)){ // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::ND_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(1)) { // bf16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::ND_FORMAT> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(16)) { // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::NZ_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(17)) { // bf16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::NZ_FORMAT> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.Run();
#endif
    } else if(TILING_KEY_IS(4)){ // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::ND_FORMAT, true> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, false, BlockStack::FOUR_FLOW, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(5)) { // bf16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::ND_FORMAT, true> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, false, BlockStack::FOUR_FLOW, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(20)) { // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::NZ_FORMAT, true> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, false, BlockStack::FOUR_FLOW, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(21)) { // bf16
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::NZ_FORMAT, true> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, false, BlockStack::FOUR_FLOW, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.RunTP1();
#endif
} else  if (TILING_KEY_IS(32)) { // fp16 + nd + preload_depth 1 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::ND_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(33)) { // bf16 + nd + preload_depth 1 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::ND_FORMAT> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(48)) { // fp16 + nz + preload_depth 1 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::NZ_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(49)) { // bf16 + nz + preload_depth 1 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::NZ_FORMAT> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, true> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(36)) { // fp16 + nd + preload_depth 2 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::ND_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, true, BlockStack::FOUR_FLOW> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(37)) { // bf16 + nd + preload_depth 2 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::ND_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, true, BlockStack::FOUR_FLOW> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(52)) { // fp16 + nz + preload_depth 2 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_HALF_DATA, half, half, half, half, InputFormat::NZ_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, true, BlockStack::FOUR_FLOW> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(53)) { // bf16 + nz + preload_depth 2 + ring 1
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16, __bf16, InputFormat::NZ_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, true, BlockStack::FOUR_FLOW> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.SetArgs2(lse_gm);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(18)) { // int8_t(IN) fp16(OUT)
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_INT8_DATA, int8_t, half, half, int8_t, InputFormat::NZ_FORMAT> pa_aic_fp16 {};
        pa_aic_fp16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_INT8_DATA, int8_t, half> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(19)) { // int8_t(IN) bf16(OUT)
#ifdef __DAV_C220_CUBE__
        MLAttentionDecoderAic<TilingKeyType::TILING_INT8_DATA, int8_t, __bf16, __bf16, int8_t, InputFormat::NZ_FORMAT> pa_aic_bf16 {};
        pa_aic_bf16.SetArgs(q_gm, q_rope_gm, ctkv_gm, ctkv_rope_gm, block_tables_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, tiling_para_gm);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        MLADecoderAiv<TilingKeyType::TILING_INT8_DATA, int8_t, __bf16> pa_aiv {};
        pa_aiv.SetArgs(block_tables_gm, deq_qk_gm, deq_pv_gm, o_gm, s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tmp_gm, tiling_para_gm, mask_gm);
        pa_aiv.Run();
#endif
    }
    
}
