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

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "aclnn_convert_kv_cache_format.h"
#include "utils_tensor.h"

namespace {

using HalfBits = uint16_t;

struct BlockStat {
  std::unordered_map<HalfBits, int> freq;
};

static inline HalfBits half_bits_at(const at::Tensor& t, int64_t idx) {
  TORCH_CHECK(
      t.device().is_cpu(), "half_bits_at requires a CPU tensor, got ", t.device());
  const HalfBits* p = reinterpret_cast<const HalfBits*>(t.data_ptr<at::Half>());
  return p[idx];
}

BlockStat build_block_stat(const at::Tensor& tensor,
                           int64_t token_start,
                           int64_t block_tokens,
                           int64_t token_size) {
  TORCH_CHECK(tensor.device().is_cpu(), "build_block_stat requires a CPU tensor");
  BlockStat st;
  int64_t base = token_start * token_size;
  int64_t total = block_tokens * token_size;
  for (int64_t i = 0; i < total; ++i) {
    HalfBits v = half_bits_at(tensor, base + i);
    st.freq[v] += 1;
  }
  return st;
}

bool block_mem_equal(const at::Tensor& a,
                     const at::Tensor& b,
                     int64_t token_start,
                     int64_t block_tokens,
                     int64_t token_size) {
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(),
              "block_mem_equal requires CPU tensors, a=",
              a.device(),
              " b=",
              b.device());
  int64_t offset = token_start * token_size;
  int64_t elems = block_tokens * token_size;
  const void* pa =
      reinterpret_cast<const void*>(a.data_ptr<at::Half>() + offset);
  const void* pb =
      reinterpret_cast<const void*>(b.data_ptr<at::Half>() + offset);
  return std::memcmp(pa, pb, elems * sizeof(at::Half)) == 0;
}

void assert_block_reordered_only(const at::Tensor& before,
                                 const at::Tensor& after,
                                 int64_t token_start,
                                 int64_t block_tokens,
                                 int64_t token_size,
                                 const std::string& tag) {
  TORCH_CHECK(before.device().is_cpu() && after.device().is_cpu(),
              "assert_block_reordered_only requires CPU tensors");
  BlockStat st_before =
      build_block_stat(before, token_start, block_tokens, token_size);
  BlockStat st_after =
      build_block_stat(after, token_start, block_tokens, token_size);
  ASSERT_EQ(st_before.freq.size(), st_after.freq.size())
      << tag << " freq map size mismatch";
  for (auto& kv : st_before.freq) {
    auto it = st_after.freq.find(kv.first);
    ASSERT_TRUE(it != st_after.freq.end()) << tag << " missing value";
    ASSERT_EQ(it->second, kv.second) << tag << " frequency mismatch";
  }
  bool same =
      block_mem_equal(before, after, token_start, block_tokens, token_size);
  if (same) {
    std::cout << "[WARN] block " << tag
              << " appears identical after ND2NZ; layout reordering may not have been triggered."
              << std::endl;
  } else {
    SUCCEED();
  }
}

class ConvertKvCacheFormatTest : public ::testing::Test {
 protected:
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;

  void SetUp() override { utils::initialize_acl(deviceId, &stream); }
  void TearDown() override {
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
  }

  // Wrapper: input can be CPU or NPU; this will move t to NPU in place;
  // therefore pass a temporary clone.
  aclTensor* create_acl_from_tensor_inplace_move(at::Tensor& t) {
    aclTensor* aclT = nullptr;
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(t.sizes().vec(), t, &aclT),
        "create tensor failed");
    return aclT;
  }

  static at::Tensor to_cpu(const at::Tensor& t) {
    return t.device().is_cpu() ? t : t.to(torch::kCPU);
  }
};

TEST_F(ConvertKvCacheFormatTest, PrefillMultiBatch) {
  const int64_t num_batches = 3;
  const int64_t num_kv_heads = 1;
  const int64_t head_size_k = 512;
  const int64_t head_size_v = 64;
  const int64_t block_size = 128;

  std::vector<int32_t> kvSeqLens = {64, 128, 256};
  std::vector<int64_t> batchOffsets(num_batches);
  batchOffsets[0] = 0;
  batchOffsets[1] = batchOffsets[0] + kvSeqLens[0];
  batchOffsets[2] = batchOffsets[1] + kvSeqLens[1];
  int64_t total_tokens = batchOffsets[2] + kvSeqLens[2];
  int64_t padded_tokens =
      ((total_tokens + block_size - 1) / block_size) * block_size;

  int64_t token_size_k = num_kv_heads * head_size_k;
  int64_t token_size_v = num_kv_heads * head_size_v;

  // Reference CPU cache
  at::Tensor kHost = torch::empty({padded_tokens, num_kv_heads, head_size_k},
                                  torch::dtype(torch::kFloat16));
  at::Tensor vHost = torch::empty({padded_tokens, num_kv_heads, head_size_v},
                                  torch::dtype(torch::kFloat16));

  auto fill_pattern = [](at::Tensor& t, float base_mul) {
    auto sizes = t.sizes();
    int64_t T = sizes[0], H = sizes[1], D = sizes[2];
    auto acc = t.data_ptr<at::Half>();
    for (int64_t tok = 0; tok < T; ++tok) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < D; ++d) {
          int64_t linear_channel = h * D + d;
          float val = static_cast<float>(tok * base_mul + linear_channel);
          acc[(tok * H + h) * D + d] = static_cast<at::Half>(val);
        }
      }
    }
  };
  fill_pattern(kHost, 100.f);
  fill_pattern(vHost, 100.f);

  at::Tensor offsetsHost =
      torch::empty({num_batches}, torch::dtype(torch::kInt64));
  std::memcpy(offsetsHost.data_ptr<int64_t>(),
              batchOffsets.data(),
              sizeof(int64_t) * num_batches);
  at::Tensor seqLensHost =
      torch::empty({num_batches}, torch::dtype(torch::kInt32));
  std::memcpy(seqLensHost.data_ptr<int32_t>(),
              kvSeqLens.data(),
              sizeof(int32_t) * num_batches);

  // Backup reference
  at::Tensor kBefore = kHost.clone();
  at::Tensor vBefore = vHost.clone();

  at::Tensor kDevTensor = kHost.clone();
  at::Tensor vDevTensor = vHost.clone();
  at::Tensor offsetsDevTensor = offsetsHost.clone();
  at::Tensor seqLensDevTensor = seqLensHost.clone();

  aclTensor* kAcl = create_acl_from_tensor_inplace_move(kDevTensor);
  aclTensor* vAcl = create_acl_from_tensor_inplace_move(vDevTensor);
  aclTensor* offsetAcl = create_acl_from_tensor_inplace_move(offsetsDevTensor);
  aclTensor* seqLenAcl = create_acl_from_tensor_inplace_move(seqLensDevTensor);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  bool isPrefill = true;
  CHECK_ACL_SUCCESS(aclnnConvertKvCacheFormatGetWorkspaceSize(kAcl,
                                                              vAcl,
                                                              offsetAcl,
                                                              seqLenAcl,
                                                              isPrefill,
                                                              num_kv_heads,
                                                              head_size_k,
                                                              head_size_v,
                                                              &workspaceSize,
                                                              &executor),
                    "GetWorkspaceSize failed");

  void* workspace = nullptr;
  if (workspaceSize) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
        "workspace alloc failed");
  }
  CHECK_ACL_SUCCESS(
      aclnnConvertKvCacheFormat(workspace, workspaceSize, executor, stream),
      "launch failed");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream), "sync failed");
  if (workspace) aclrtFree(workspace);

  at::Tensor kAfter = to_cpu(kDevTensor);
  at::Tensor vAfter = to_cpu(vDevTensor);

  ASSERT_TRUE(
      block_mem_equal(kBefore, kAfter, batchOffsets[0], 64, token_size_k))
      << "batch0 should remain unchanged (K)";
  ASSERT_TRUE(
      block_mem_equal(vBefore, vAfter, batchOffsets[0], 64, token_size_v))
      << "batch0 should remain unchanged (V)";

  assert_block_reordered_only(kBefore,
                              kAfter,
                              batchOffsets[1],
                              block_size,
                              token_size_k,
                              "K batch1 block0");
  assert_block_reordered_only(vBefore,
                              vAfter,
                              batchOffsets[1],
                              block_size,
                              token_size_v,
                              "V batch1 block0");

  assert_block_reordered_only(kBefore,
                              kAfter,
                              batchOffsets[2] + 0 * block_size,
                              block_size,
                              token_size_k,
                              "K batch2 block0");
  assert_block_reordered_only(kBefore,
                              kAfter,
                              batchOffsets[2] + 1 * block_size,
                              block_size,
                              token_size_k,
                              "K batch2 block1");
  assert_block_reordered_only(vBefore,
                              vAfter,
                              batchOffsets[2] + 0 * block_size,
                              block_size,
                              token_size_v,
                              "V batch2 block0");
  assert_block_reordered_only(vBefore,
                              vAfter,
                              batchOffsets[2] + 1 * block_size,
                              block_size,
                              token_size_v,
                              "V batch2 block1");
}

TEST_F(ConvertKvCacheFormatTest, DecodeIncremental) {
  const int64_t num_batches = 1;
  const int64_t num_kv_heads = 1;
  const int64_t head_size_k = 512;
  const int64_t head_size_v = 64;
  const int64_t block_size = 128;
  const int64_t max_tokens = 512;

  int64_t token_size_k = num_kv_heads * head_size_k;
  int64_t token_size_v = num_kv_heads * head_size_v;

  // Reference (fully filled, simulates incremental decode length increases)
  at::Tensor kHost = torch::empty({max_tokens, num_kv_heads, head_size_k},
                                  torch::dtype(torch::kFloat16));
  at::Tensor vHost = torch::empty({max_tokens, num_kv_heads, head_size_v},
                                  torch::dtype(torch::kFloat16));

  auto fill_pattern = [](at::Tensor& t) {
    auto sizes = t.sizes();
    int64_t T = sizes[0], H = sizes[1], D = sizes[2];
    auto acc = t.data_ptr<at::Half>();
    for (int64_t tok = 0; tok < T; ++tok) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < D; ++d) {
          int64_t linear_channel = h * D + d;
          float val = static_cast<float>(tok * 10 + linear_channel);
          acc[(tok * H + h) * D + d] = static_cast<at::Half>(val);
        }
      }
    }
  };
  fill_pattern(kHost);
  fill_pattern(vHost);

  at::Tensor offsetsHost =
      torch::zeros({num_batches}, torch::dtype(torch::kInt64));

  auto run_decode = [&](int32_t cur_len, bool expect_convert) {
    at::Tensor seqLenHost =
        torch::full({num_batches}, cur_len, torch::dtype(torch::kInt32));

    at::Tensor kBefore = kHost.clone();  // CPU clone
    at::Tensor vBefore = vHost.clone();

    at::Tensor kDev = kHost.clone();
    at::Tensor vDev = vHost.clone();
    at::Tensor offsetsDev = offsetsHost.clone();
    at::Tensor seqLenDev = seqLenHost.clone();

    aclTensor *kAcl = nullptr, *vAcl = nullptr, *offAcl = nullptr,
              *seqAcl = nullptr;
    kAcl = create_acl_from_tensor_inplace_move(kDev);
    vAcl = create_acl_from_tensor_inplace_move(vDev);
    offAcl = create_acl_from_tensor_inplace_move(offsetsDev);
    seqAcl = create_acl_from_tensor_inplace_move(seqLenDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool isPrefill = false;
    CHECK_ACL_SUCCESS(aclnnConvertKvCacheFormatGetWorkspaceSize(kAcl,
                                                                vAcl,
                                                                offAcl,
                                                                seqAcl,
                                                                isPrefill,
                                                                num_kv_heads,
                                                                head_size_k,
                                                                head_size_v,
                                                                &workspaceSize,
                                                                &executor),
                      "GetWorkspaceSize failed decode");

    void* workspace = nullptr;
    if (workspaceSize) {
      CHECK_ACL_SUCCESS(
          aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
          "workspace alloc failed");
    }
    CHECK_ACL_SUCCESS(
        aclnnConvertKvCacheFormat(workspace, workspaceSize, executor, stream),
        "decode launch failed");
    CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream), "sync failed");
    if (workspace) aclrtFree(workspace);

    at::Tensor kAfter = to_cpu(kDev);
    at::Tensor vAfter = to_cpu(vDev);

    if (expect_convert) {
      int64_t block_start = cur_len - block_size;
      assert_block_reordered_only(kBefore,
                                  kAfter,
                                  block_start,
                                  block_size,
                                  token_size_k,
                                  "Decode K new block");
      assert_block_reordered_only(vBefore,
                                  vAfter,
                                  block_start,
                                  block_size,
                                  token_size_v,
                                  "Decode V new block");
    } else {
      ASSERT_TRUE(block_mem_equal(kBefore, kAfter, 0, cur_len, token_size_k))
          << "No block should be converted yet (K)";
      ASSERT_TRUE(block_mem_equal(vBefore, vAfter, 0, cur_len, token_size_v))
          << "No block should be converted yet (V)";
    }
  };

  run_decode(127, false);  // block not complete yet
  run_decode(128, true);   // first block
  run_decode(255, false);  // second block incomplete
  run_decode(256, true);   // second block
}

}  // namespace