#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass
np.random.seed(1)


WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for i in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(1, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    print(q_seqlen_list)
    print(kv_seqlen_list)
    return q_seqlen_list, kv_seqlen_list

class TestFlashAttentionInfer():

    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any
        shape_param: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int):
        if q_seqlen > kv_seqlen:
            logging("[ERROR] q_seqlen cannot exceed kv_seqlen.")
            sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                    right[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask    # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = query
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        post_mask_factor = -3e38
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32) * post_mask_factor
        
        # softmax
        p_high = self.softmax_numpy(sim_high)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(query.dtype)
        return out, out_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, true_out) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size

        batch = len(attention_inputs.shape_param.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.shape_param.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen): # j 每个k token拼接
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.shape_param.kv_dtype == 0:
                keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                # import pdb;pdb.set_trace()
            scale = 1.0 / (head_size_qk ** 0.5)
            if attention_inputs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            else:
                mask = None
            out, out_high = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        num_kv_tokens = np.array(gen_data_params.k_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)
        query = np.random.uniform(q_min_range, q_max_range,
            size=(num_tokens, gen_data_params.num_heads, head_size_qk)).astype(gen_data_params.dtype)
        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        block_tables = []   # (num_tokens, max_num_blocks_per_seq）
        layout = 'TND'
        key_cache = None
        value_cache = None
        if gen_data_params.kv_dtype == 1:
            key_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size,
                gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)

            value_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size,
                gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
            max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            for i in range(batch_size):
                block_table = [
                    max_num_blocks_per_seq * i + j
                    for j in range(max_num_blocks_per_seq)
                ]
                block_tables.append(block_table)
        elif gen_data_params.kv_dtype == 0:
            if layout == 'TND':
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
            elif layout == 'BSND':
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
        
        if gen_data_params.mask_type == 1:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(gen_data_params.dtype)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                kseqlen = gen_data_params.k_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1).astype(gen_data_params.dtype)
                mask[pre_qseqlen : (pre_qseqlen + qseqlen), kseqlen - qseqlen : kseqlen] = tri
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
        elif gen_data_params.mask_type == 0:
            mask = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = np.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = np.zeros(shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(query, key_cache, value_cache, block_tables,
            gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, mask, gen_data_params.mask_type, gen_data_params)
        
        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            ref_output,
            true_out,
        )

        num_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
        num_kv_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens.bin"))
        query.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
        key_cache.tofile(os.path.join(WORKSPACE, "data", "k.bin"))
        value_cache.tofile(os.path.join(WORKSPACE, "data", "v.bin"))
        np.array(block_tables).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "block_table.bin"))
        np.array(gen_data_params.q_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "q_seqlen.bin"))
        np.array(gen_data_params.k_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "kv_seqlen.bin"))
        if mask is not None:
            actual_input_mask_triu = np.triu(np.ones((1024, 1024)), 1).astype(gen_data_params.dtype)
            actual_input_mask_triu.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        ref_output.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden.bin"))


if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    embedding_size = int(sys.argv[6])
    block_size = 128
    is_varied_len = int(sys.argv[7])
    mask_type = int(sys.argv[8])

    str_dtype = str(sys.argv[9])
    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()

    kv_dtype = int(sys.argv[10])

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, batch)
    
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    
    testObj = TestFlashAttentionInfer()
    gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                            kv_heads, embedding_size,
                                            num_blocks, block_size, mask_type, dtype, kv_dtype)
    testObj.calc_data(gen_data_params)

