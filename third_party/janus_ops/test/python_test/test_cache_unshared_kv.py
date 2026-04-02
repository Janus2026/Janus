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

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cache_unshared_kv_npu(dtype):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(1234)

    batch = 10
    max_decode_step = 3
    beam_size = 512
    head_num = 8
    head_dim = 128
    block_num = 30

    x_key_block_npu = torch.zeros(
        [block_num, beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    ).npu()
    x_value_block_npu = torch.zeros(
        [block_num, beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    ).npu()
    x_key_block = torch.zeros(
        [block_num, beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    )
    x_value_block = torch.zeros(
        [block_num, beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    )

    block_table = torch.randperm(block_num)[:batch].to(torch.int32)
    atol_div = 1e-6

    for decode_step in range(1, max_decode_step + 1):
        curr_key = torch.rand([batch * beam_size, head_num, head_dim], dtype=dtype)
        curr_value = torch.rand([batch * beam_size, head_num, head_dim], dtype=dtype)
        real_index = decode_step - 1

        # Fill key/value for this step
        for b_idx in range(batch):
            block_index = block_table[b_idx]
            x_key_block[block_index, :, :, real_index, :] = curr_key[b_idx * beam_size : (b_idx * beam_size + beam_size)]
            x_value_block[block_index, :, :, real_index, :] = curr_value[b_idx * beam_size : (b_idx * beam_size + beam_size)]

        decode_step_tensor = torch.tensor([decode_step], dtype=torch.int32)

        # Run NPU kernel (in-place on x_key_block_npu/x_value_block_npu)
        custom_ops.cache_unshared_kv_npu(
            x_key_block_npu,
            x_value_block_npu,
            curr_key.npu(),
            curr_value.npu(),
            block_table.npu(),
            decode_step_tensor.npu(),
        )

        npu_key_block = x_key_block_npu.cpu()
        npu_value_block = x_value_block_npu.cpu()

        assert torch.allclose(npu_key_block, x_key_block, atol=atol_div, rtol=0)
        assert torch.allclose(npu_value_block, x_value_block, atol=atol_div, rtol=0)
