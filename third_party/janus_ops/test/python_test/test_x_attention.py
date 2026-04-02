#!/usr/bin/env python3
# Copyright 2025 The Janus Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import random
import copy
import pytest
import torch

from dataclasses import dataclass
from ml_dtypes import bfloat16


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1)


def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for _ in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(1, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    return q_seqlen_list, kv_seqlen_list


class TestFlashAttentionInfer:
    @dataclass
    class AttentionInputs:
        # [bs, 1, headnum, headdim]
        query: torch.Tensor
        # shared_kv_type=1: [num_blocks, block_size, kv_head, headdim]
        # shared_kv_type=0: [num_shared_kv, kv_head, headdim]
        key_cache: torch.Tensor
        value_cache: torch.Tensor
        # unshared_kv_type=0: [bs (request_num * beam_size), kv_head, max_decode_step, headdim]
        # unshared_kv_type=1: [max_request_num, beam_size, kv_head, max_decode_step, headdim]
        unshared_k: torch.Tensor
        unshared_v: torch.Tensor
        # shared_kv_type=1: [request_num, max_blocks_per_batch] max_blocks_per_batch = ceil(max_shared_kvlen / block_size)
        # shared_kv_type=0: None
        block_tables: list
        # unshared_kv_type=1: [request_num] list of block indices for each request
        # unshared_kv_type=0: None
        unshared_block_tables: list
        # [request_num] (1, 1, 1)
        q_seqlen_list: list
        # [request_num] (share_len1, share_len2)
        k_seqlen_list: list
        global_mask: any
        mask_type: int
        shape_param: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        beam_size: int
        unshared_kvlen: int
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        shared_kv_type: int
        unshared_kv_type: int

    @classmethod
    def group_matmul(cls, head, kv_head, left, right, right_row=None, right_col=None):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            if right_row is None:
                current_right = right[i : (i + 1), :, :]
            else:
                current_right = right[i : (i + 1), :right_row, :right_col]
            left_group = left[i * group_num : (i + 1) * group_num, :, :]
            group_score = torch.matmul(left_group.to(torch.float32), current_right.to(torch.float32))
            score = group_score if score is None else torch.cat((score, group_score), dim=0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = torch.max(sim, dim=-1, keepdim=True).values
        sim_sub = sim - row_max
        sim_sub = torch.exp(sim_sub)
        row_sum = torch.sum(sim_sub, dim=-1, keepdim=True)
        soft_res = sim_sub  # no div rowsum
        return soft_res, row_max, row_sum

    def ref_masked_attention(self,
        query,  # (q_seqlen, num_heads, head_size)
        key,    # (k_seqlen, kv_heads, head_size)
        value,
        scale: float,
        mask    # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = query.permute(1, 0, 2)
        key = key.permute(1, 2, 0)
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        # softmax
        p_high, gm, gl = self.softmax_numpy(sim_high)
        p = p_high.to(query.dtype)
        p_high = p_high.to(torch.float32)
        value = value.permute(1, 0, 2)
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = out_high.permute(1, 0, 2)
        out = out.permute(1, 0, 2)
        out = out.to(query.dtype)
        return out, out_high, gm, gl

    def ref_single_query_unshared_kv_attention(self,
        attention_inputs: "TestFlashAttentionInfer.AttentionInputs",
        output: torch.Tensor,
        true_out: torch.Tensor,
        unshared_gl: torch.Tensor,
        unshared_gm: torch.Tensor,
    ) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size = attention_inputs.shape_param.head_size
        beam_size = attention_inputs.shape_param.beam_size
        request_num = len(attention_inputs.q_seqlen_list)
        batch = beam_size * request_num
        decode_step = attention_inputs.shape_param.unshared_kvlen
        unshared_kv_type = attention_inputs.shape_param.unshared_kv_type
        max_decode_step = attention_inputs.unshared_k.shape[-2] if len(attention_inputs.unshared_k.shape) == 5 else attention_inputs.unshared_k.shape[2]

        scale = 1.0 / (head_size ** 0.5)

        if unshared_kv_type == 0:
            # Continuous format: [batch, kv_heads, max_decode_step, head_size]
            assert attention_inputs.query.shape == (batch, num_heads, head_size)
            assert attention_inputs.unshared_k.shape == (batch, kv_heads, max_decode_step, head_size)
            assert attention_inputs.unshared_v.shape == (batch, kv_heads, max_decode_step, head_size)

            for i in range(batch):
                q = attention_inputs.query[i : i + 1, :, :]
                k = attention_inputs.unshared_k[i, :, :, :]
                v = attention_inputs.unshared_v[i, :, :, :]
                # Transpose for group_matmul
                q_t = q.permute(1, 0, 2)
                k_t = k.permute(0, 2, 1)

                sim = self.group_matmul(num_heads, kv_heads, q_t, k_t, head_size, decode_step)  # [num_heads, 1, unshared_kvlen]
                sim = sim * scale

                # Softmax with stats
                p, gm, gl = self.softmax_numpy(sim)
                gm = gm.permute(1, 0, 2)  # (q_seqlen, num_heads, 1)
                gl = gl.permute(1, 0, 2)
                p_high = p.to(torch.float32)
                out_high = self.group_matmul(num_heads, kv_heads, p_high, v, decode_step, head_size)
                out_high = out_high.permute(1, 0, 2)
                p_low = p.to(attention_inputs.query.dtype)
                out_low = self.group_matmul(num_heads, kv_heads, p_low, v, decode_step, head_size)
                out_low = out_low.permute(1, 0, 2)
                out_low = out_low.to(attention_inputs.query.dtype)

                # Write outputs
                output[i : i + 1, :, :] = out_low
                true_out[i : i + 1, :, :] = out_high

                unshared_gm[i, :, :] = gm[:, :, :]
                unshared_gl[i, :, :] = gl[:, :, :]
        else:
            # Paged format: [max_request_num, beam_size, kv_heads, max_decode_step, head_size]
            assert attention_inputs.query.shape == (batch, num_heads, head_size)
            assert len(attention_inputs.unshared_k.shape) == 5
            assert attention_inputs.unshared_k.shape == attention_inputs.unshared_v.shape
            max_request_num = attention_inputs.unshared_k.shape[0]
            
            # Use unshared_block_tables to map request to cache index
            for req_idx in range(request_num):
                # Get the cache index for this request from unshared_block_tables
                if attention_inputs.unshared_block_tables is not None and len(attention_inputs.unshared_block_tables) > req_idx:
                    cache_idx = attention_inputs.unshared_block_tables[req_idx][0]  # First block index for this request
                else:
                    # Fallback: use request index directly
                    cache_idx = req_idx
                
                for beam_idx in range(beam_size):
                    i = req_idx * beam_size + beam_idx
                    q = attention_inputs.query[i : i + 1, :, :]
                    
                    # Get unshared_k and unshared_v from paged format using cache_idx
                    k = attention_inputs.unshared_k[cache_idx, beam_idx, :, :, :]  # [kv_heads, max_decode_step, head_size]
                    v = attention_inputs.unshared_v[cache_idx, beam_idx, :, :, :]  # [kv_heads, max_decode_step, head_size]
                    
                    # Transpose for group_matmul
                    q_t = q.permute(1, 0, 2)
                    k_t = k.permute(0, 2, 1)  # [kv_heads, head_size, max_decode_step]

                    sim = self.group_matmul(num_heads, kv_heads, q_t, k_t, head_size, decode_step)  # [num_heads, 1, unshared_kvlen]
                    sim = sim * scale

                    # Softmax with stats
                    p, gm, gl = self.softmax_numpy(sim)
                    gm = gm.permute(1, 0, 2)  # (q_seqlen, num_heads, 1)
                    gl = gl.permute(1, 0, 2)
                    p_high = p.to(torch.float32)
                    out_high = self.group_matmul(num_heads, kv_heads, p_high, v, decode_step, head_size)
                    out_high = out_high.permute(1, 0, 2)
                    p_low = p.to(attention_inputs.query.dtype)
                    out_low = self.group_matmul(num_heads, kv_heads, p_low, v, decode_step, head_size)
                    out_low = out_low.permute(1, 0, 2)
                    out_low = out_low.to(attention_inputs.query.dtype)

                    # Write outputs
                    output[i : i + 1, :, :] = out_low
                    true_out[i : i + 1, :, :] = out_high

                    unshared_gm[i, :, :] = gm[:, :, :]
                    unshared_gl[i, :, :] = gl[:, :, :]

    def ref_single_query_shared_kv_attention(
        self,
        attention_inputs: "TestFlashAttentionInfer.AttentionInputs",
        output,
        true_out,
        shared_gl,
        shared_gm,
    ) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size
        beam_size = attention_inputs.shape_param.beam_size
        request_num = len(attention_inputs.shape_param.q_seqlen_list)
        shared_kv_type = attention_inputs.shape_param.shared_kv_type
        cu_seqlen = 0
        kv_seqlen_now = 0
        layout = "TND"

        for i in range(request_num):
            q_seqlen = int(beam_size)
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            if layout == "TND":
                q = attention_inputs.query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
            elif layout == "BSND":
                q = attention_inputs.query[i, :, :, :]
            keys = []
            values = []
            if shared_kv_type == 1:
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
            else:
                for j in range(k_seqlen):
                    k = attention_inputs.key_cache[kv_seqlen_now + j, :, :]
                    # k shape is already (kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[kv_seqlen_now + j, :, :]
                    # v shape is already (kv_heads, head_size_vo)
                    values.append(v)

            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = 1.0 / (head_size_qk ** 0.5)
            mask = None
            out, out_high, gm, gl = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            gm = gm.permute(1, 0, 2)  # (q_seqlen, num_heads, 1)
            gl = gl.permute(1, 0, 2)
            output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out_high
            shared_gl[cu_seqlen : cu_seqlen + q_seqlen, :, :] = gl
            shared_gm[cu_seqlen : cu_seqlen + q_seqlen, :, :] = gm
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def call_device_op(self, attention_inputs: "TestFlashAttentionInfer.AttentionInputs",
                       q, k, v, unshared_k, unshared_v, 
                       block_tables, unshared_block_tables, 
                       actual_shared_kvlen, decode_step):        

        shared_kv_type = attention_inputs.shape_param.shared_kv_type
        unshared_kv_type = attention_inputs.shape_param.unshared_kv_type

        q = q.npu()
        k = k.npu()
        v = v.npu()
        unshared_k = unshared_k.npu()
        unshared_v = unshared_v.npu()
        
        if shared_kv_type == 1:
            block_tables = torch.tensor(copy.deepcopy(block_tables), dtype=torch.int32).npu()
        else:
            block_tables = None
        
        if unshared_kv_type == 1:
            unshared_block_tables = torch.tensor(copy.deepcopy(unshared_block_tables), dtype=torch.int32).npu()
        else:
            unshared_block_tables = None
            
        actual_shared_kvlen = torch.tensor(actual_shared_kvlen, dtype=torch.int32).npu()
        decode_step_tensor = torch.tensor([decode_step], dtype=torch.int32).npu()
        
        # ========== DEBUG ==========
        print("="*80)
        print(f"  q shape: {q.shape}")
        print(f"  k shape: {k.shape if k is not None else None}")
        print(f"  v shape: {v.shape if v is not None else None}")
        print(f"  unshared_k shape: {unshared_k.shape}")
        print(f"  unshared_v shape: {unshared_v.shape}")
        print(f"  block_tables shape: {block_tables.shape if block_tables is not None else None}")
        print(f"  unshared_block_tables shape: {unshared_block_tables.shape if unshared_block_tables is not None else None}")
        print(f"  actual_shared_kvlen shape: {actual_shared_kvlen.shape}")
        print(f"  decode_step_tensor shape: {decode_step_tensor.shape}")
        print("="*80 + "\n")
        
        attn_out = custom_ops.x_attention_npu(
            q, k, v, unshared_k, unshared_v, actual_shared_kvlen, decode_step_tensor,
            block_tables, unshared_block_tables
        )
        return attn_out

    def calc_data(self, gen_data_params: "TestFlashAttentionInfer.GenDataParams"):
        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        beam_size = gen_data_params.beam_size
        request_num = len(gen_data_params.k_seqlen_list)
        decode_step = gen_data_params.unshared_kvlen
        max_decode_step = 3

        num_tokens = sum(gen_data_params.q_seqlen_list) * beam_size
        num_shared_kv = sum(gen_data_params.k_seqlen_list)

        batch_size = request_num * beam_size
        torch_dtype = gen_data_params.dtype
        query = (torch.empty((num_tokens, gen_data_params.num_heads, head_size_qk), dtype=torch_dtype)
                 .uniform_(q_min_range, q_max_range))
        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        block_tables = []  # (request_num, max_num_blocks_per_seq)
        key_cache = None
        value_cache = None
        
        # Generate shared KV cache based on shared_kv_type
        if gen_data_params.shared_kv_type == 1:
            # Paged format: [num_blocks, block_size, kv_heads, head_dim]
            key_cache = (torch.empty(
                (gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_qk),
                dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
            value_cache = (torch.empty(
                (gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_vo),
                dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
            max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            for i in range(request_num):
                block_table = [max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)]
                block_tables.append(block_table)
        else:
            # Continuous format: [num_shared_kv, kv_heads, head_dim]
            key_cache = (torch.empty(
                (num_shared_kv, gen_data_params.kv_heads, head_size_qk),
                dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))

            value_cache = (torch.empty(
                (num_shared_kv, gen_data_params.kv_heads, head_size_vo),
                dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
            block_tables = None

        # Generate unshared KV cache based on unshared_kv_type
        unshared_key = None
        unshared_value = None
        unshared_block_tables = None
        
        if gen_data_params.unshared_kv_type == 0:
            # Continuous format: [request_num * beam_size, kv_heads, max_decode_step, head_dim]
            unshared_key = (torch.empty(
                (batch_size, gen_data_params.kv_heads, max_decode_step, head_size_qk), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
            unshared_value = (torch.empty(
                (batch_size, gen_data_params.kv_heads, max_decode_step, head_size_vo), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
        else:
            # Paged format: [max_request_num, beam_size, kv_heads, max_decode_step, head_dim]
            # Use request_num as max_request_num for simplicity (can be larger in real scenarios)
            max_request_num = request_num
            unshared_key = (torch.empty(
                (max_request_num, beam_size, gen_data_params.kv_heads, max_decode_step, head_size_qk), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
            unshared_value = (torch.empty(
                (max_request_num, beam_size, gen_data_params.kv_heads, max_decode_step, head_size_vo), dtype=torch_dtype
            ).uniform_(kv_min_range, kv_max_range))
            
            # Generate unshared_block_tables
            # Each request has a block table mapping to its unshared KV cache
            # For simplicity, each request maps to its own index in the paged cache
            unshared_block_tables = []
            for i in range(request_num):
                # Each request maps to its own index (i) in the paged cache
                unshared_block_tables.append([request_num - 1 - i])

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        sum_max_shape_out = (num_tokens, gen_data_params.num_heads, 1)
        shared_ref_out = torch.zeros(shape_out, dtype=torch_dtype)
        shared_true_out = torch.zeros(shape_out, dtype=torch.float32)
        shared_gl = torch.zeros(sum_max_shape_out, dtype=torch.float32)
        shared_gm = torch.zeros(sum_max_shape_out, dtype=torch.float32)

        unshared_ref_out = torch.zeros(shape_out, dtype=torch_dtype)
        unshared_true_out = torch.zeros(shape_out, dtype=torch.float32)
        unshared_gl = torch.zeros(sum_max_shape_out, dtype=torch.float32)
        unshared_gm = torch.zeros(sum_max_shape_out, dtype=torch.float32)

        attention_inputs = self.AttentionInputs(
            query,
            key_cache,
            value_cache,
            unshared_key,
            unshared_value,
            block_tables,
            unshared_block_tables,
            gen_data_params.q_seqlen_list,
            gen_data_params.k_seqlen_list,
            None,
            gen_data_params.mask_type,
            gen_data_params,
        )

        self.ref_single_query_shared_kv_attention(
            attention_inputs, shared_ref_out, shared_true_out, shared_gl, shared_gm
        )

        self.ref_single_query_unshared_kv_attention(
            attention_inputs, unshared_ref_out, unshared_true_out, unshared_gl, unshared_gm
        )

        gm = torch.maximum(shared_gm, unshared_gm)
        update_shared_expgm = torch.exp(shared_gm - gm)
        update_unshared_expgm = torch.exp(unshared_gm - gm)
        gl = shared_gl * update_shared_expgm + unshared_gl * update_unshared_expgm
        tmp_shared_true = shared_true_out * update_shared_expgm
        tmp_unshared_true = unshared_true_out * update_unshared_expgm
        tmp_add = tmp_shared_true + tmp_unshared_true
        final_true_out = tmp_add / gl

        # Prepare actual_shared_kvlen for device op
        actual_shared_kvlen = gen_data_params.k_seqlen_list
        
        npu_res = self.call_device_op(
            attention_inputs, query, key_cache, value_cache, unshared_key, unshared_value, 
            block_tables, unshared_block_tables, actual_shared_kvlen, decode_step
        )
        golden_res = final_true_out
        npu_res = npu_res.cpu().float()
        assert torch.allclose(npu_res, golden_res, atol=0.001, rtol=0.001)

@pytest.mark.parametrize("dtype,request_num,beam_size,q_seqlen,kv_seqlen,unshared_seqlen,num_head,kv_heads,embedding_size,block_size,is_varied_len,mask_type,shared_kv_type,unshared_kv_type", [
    (torch.bfloat16, 1, 128, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 1, 256, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 1, 512, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 1, 1024, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 1, 2048, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 1, 4096, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 128, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 256, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 512, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 1024, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 2048, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 4096, 1, 1024, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 128, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 256, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 512, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 1024, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 2048, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    (torch.bfloat16, 2, 4096, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 8, 128, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 8, 256, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 8, 512, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 8, 1024, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 8, 2048, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 8, 4096, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 16, 128, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 16, 256, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 16, 512, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 16, 1024, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 16, 2048, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 16, 4096, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 32, 128, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 32, 256, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 32, 512, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 32, 1024, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 32, 2048, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
    # (torch.bfloat16, 32, 4096, 1, 2048, 2, 32, 8, 128, 128, 0, 0, 0, 1),
])
def test_x_attention_npu(dtype,request_num,beam_size,q_seqlen,kv_seqlen,unshared_seqlen,num_head,kv_heads,embedding_size,block_size,is_varied_len,mask_type,shared_kv_type,unshared_kv_type):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    # request = 5
    # beam_size = 512 # must >= 128
    # q_seqlen = 1  # must be 1
    # kv_seqlen = 4090  # shared_kv_len
    # unshared_seqlen = 2
    # num_head = 8
    # kv_heads = 8
    # embedding_size = 128
    # block_size = 128
    # is_varied_len = 0
    # mask_type = 0
    # shared_kv_type = 1
    # unshared_kv_type = 0

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, request_num)
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = request_num * ((max_kv_seqlen + block_size - 1) // block_size)

    test_obj = TestFlashAttentionInfer()
    gen_data_params = test_obj.GenDataParams(
        q_seqlen_list,
        kv_seqlen_list,
        beam_size,
        unshared_seqlen,
        num_head,
        kv_heads,
        embedding_size,
        num_blocks,
        block_size,
        mask_type,
        dtype,
        shared_kv_type,
        unshared_kv_type
    )
    test_obj.calc_data(gen_data_params)
