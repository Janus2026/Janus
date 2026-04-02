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

#pragma once
#ifndef MULTI_LATENT_ATTENTION_H
#define MULTI_LATENT_ATTENTION_H

#include "aclnn_multi_latent_attention.h"
#include "utils_print.h"
#include "utils_tensor.h"

#define print_tensor_shape(tensor) \
    do { \
        std::cout << #tensor " shape: "; \
        for (const auto& dim : tensor.sizes()) { \
            std::cout << dim << " "; \
        } \
        std::cout << std::endl; \
    } while (0)

#define print_tensor_value(tensor) \
    do { \
        std::cout << #tensor " value: " << tensor << std::endl; \
    } while (0)

namespace multi_latent_attention {
struct TestParams {
    int32_t num_tokens, num_heads, kv_heads;
    int32_t head_size_qk, head_size_v;
    int32_t block_size, kv_seqlen;
    float qk_scale;
};

struct TensorShapes {
    std::vector<int64_t> query_shape;
    std::vector<int64_t> query_rope_shape;
    std::vector<int64_t> kv_cache_shape;
    std::vector<int64_t> kv_cache_rope_shape;
    std::vector<int64_t> block_tables_shape;
    std::vector<int64_t> context_lens_shape;
    std::vector<int64_t> output_shape;
};

class MultiLatentAttentionBase {
public:
    MultiLatentAttentionBase(int32_t num_tokens,
                             int32_t num_heads,
                             int32_t kv_seqlen,
                             int32_t block_size
                             ) {
        int32_t head_size_qk = 576;
        int32_t head_size_v = 512;
        int32_t kv_heads = 1;
        int32_t block_nums_one_seq = (kv_seqlen + block_size - 1) / block_size;
        int32_t block_nums = (kv_seqlen + block_size - 1) / block_size * num_tokens;
        float qk_scale = 1.0 / sqrt(static_cast<float>(head_size_qk));
        params.num_tokens = num_tokens;
        params.num_heads = num_heads;
        params.kv_heads = kv_heads;
        params.head_size_qk = head_size_qk;
        params.head_size_v = head_size_v;
        params.block_size = block_size;
        params.kv_seqlen = kv_seqlen;
        params.qk_scale = qk_scale;
        shapes.query_shape = {num_tokens, num_heads, head_size_v};
        shapes.query_rope_shape = {num_tokens, num_heads, head_size_qk - head_size_v};
        shapes.kv_cache_shape = {block_nums, block_size , kv_heads, head_size_v};
        shapes.kv_cache_rope_shape = {block_nums, block_size, kv_heads, head_size_qk - head_size_v};
        shapes.block_tables_shape = {num_tokens, block_nums_one_seq};
        shapes.context_lens_shape = {num_tokens, };
        shapes.output_shape = {num_tokens, num_heads, head_size_v};
    }

    void create_torch_tensors() {
        torch::manual_seed(12);
        query = torch::rand({params.num_tokens, params.num_heads, params.head_size_v}, torch::kFloat16);
        query_rope = torch::rand({params.num_tokens, params.num_heads, params.head_size_qk - params.head_size_v}, torch::kFloat16);
        kv_cache = torch::rand({shapes.kv_cache_shape[0], shapes.kv_cache_shape[1], shapes.kv_cache_shape[2], shapes.kv_cache_shape[3]}, torch::kFloat16);
        kv_cache_rope = torch::rand({shapes.kv_cache_rope_shape[0], shapes.kv_cache_rope_shape[1], shapes.kv_cache_rope_shape[2], shapes.kv_cache_rope_shape[3]}, torch::kFloat16);
        block_tables = torch::arange(0, shapes.block_tables_shape[0] * shapes.block_tables_shape[1], torch::kInt32).reshape({shapes.block_tables_shape[0], shapes.block_tables_shape[1]});
        context_lens = torch::full({shapes.context_lens_shape[0]}, params.kv_seqlen, torch::kInt32);
        output_native = torch::zeros({params.num_tokens, params.num_heads, params.head_size_v}, torch::kFloat16);
        output_op = torch::zeros({params.num_tokens, params.num_heads, params.head_size_v}, torch::kFloat16);
    }
    
    TestParams params;
    TensorShapes shapes;
    torch::Tensor query;
    torch::Tensor query_rope;
    torch::Tensor kv_cache;
    torch::Tensor kv_cache_rope;
    torch::Tensor block_tables;
    torch::Tensor context_lens;
    torch::Tensor output_native;
    torch::Tensor output_op;
};

class MultiLatentAttentionNative {
public:
    MultiLatentAttentionNative(MultiLatentAttentionBase& base) : base_(base) {}

    void process(aclrtStream stream) {
        (void)stream;
        this->create_tensors();
        CHECK_ACL_SUCCESS(
            this->execute_multi_latent_attention_operator(queryNative,
                                                          queryRopeNative,
                                                          kvCacheNative,
                                                          kvCacheRopeNative,
                                                          blockTablesNative,
                                                          contextLensNative,
                                                          base_.output_native),
            "execute_multi_latent_attention_operator failed");
        base_.output_native = base_.output_native.to("npu");
    }

    void destroyTensors() {}

private:
    void create_tensors() {
        queryNative = base_.query.clone();
        queryRopeNative = base_.query_rope.clone();
        kvCacheNative = base_.kv_cache.clone();
        kvCacheRopeNative = base_.kv_cache_rope.clone();
        blockTablesNative = base_.block_tables.clone();
        contextLensNative = base_.context_lens.clone();
    }

    int execute_multi_latent_attention_operator(torch::Tensor query,
                                                torch::Tensor queryRope,
                                                torch::Tensor kvCache,
                                                torch::Tensor kvCacheRope,
                                                torch::Tensor blockTables,
                                                torch::Tensor contextLens,
                                                torch::Tensor attenOutOut) {
        for (int i = 0; i < base_.params.num_tokens; ++i) {
            torch::Tensor block_table = blockTables[i];
            int32_t context_len = contextLens[i].item<int32_t>();
            if (context_len == 0) {
                continue;
            }
            torch::Tensor q = torch::cat({query[i], queryRope[i]}, -1);
            std::vector<torch::Tensor> keys;
            std::vector<torch::Tensor> values;
            for (int j = 0; j < context_len; ++j) {
                int32_t block_index = block_table[j / base_.params.block_size].item<int32_t>();
                int32_t block_offset = j % base_.params.block_size;
                // printf("i=%d, j=%d, block_index=%d, block_offset=%d\n", i, j, block_index, block_offset);

                torch::Tensor k = kvCache[block_index][block_offset].reshape({base_.params.kv_heads, base_.params.head_size_v});
                torch::Tensor k_rope = kvCacheRope[block_index][block_offset].reshape({base_.params.kv_heads, base_.params.head_size_qk - base_.params.head_size_v});
                keys.push_back(torch::cat({k, k_rope}, -1));

                torch::Tensor v = kvCache[block_index][block_offset].reshape({base_.params.kv_heads, base_.params.head_size_v});
                values.push_back(v);
            }
            torch::Tensor k = torch::stack(keys, 0); // [context_len, kv_heads, head_size_qk]
            torch::Tensor v = torch::stack(values, 0); // [context_len, kv_heads, head_size_v]
            q = q.unsqueeze(0); // [1, num_heads, head_size_qk]
            k = k.permute({1, 0, 2}); // [1, context_len, head_size_qk]
            v = v.permute({1, 0, 2}); // [1, context_len, head_size_v]
            torch::Tensor attn_weights = torch::bmm(q, k.transpose(1, 2)) * base_.params.qk_scale; // [1, num_heads, context_len]
            attn_weights = torch::softmax(attn_weights, -1);
            torch::Tensor attn_output = torch::bmm(attn_weights, v); // [1, num_heads, head_size_v]
            attenOutOut[i] = attn_output.squeeze(0);
        }   
        return ACL_SUCCESS;
    }

    torch::Tensor queryNative;
    torch::Tensor queryRopeNative;
    torch::Tensor kvCacheNative;
    torch::Tensor kvCacheRopeNative;
    torch::Tensor blockTablesNative;
    torch::Tensor contextLensNative;
    MultiLatentAttentionBase& base_;
};

class MultiLatentAttentionOp {
public:
    MultiLatentAttentionOp(MultiLatentAttentionBase& base) : base_(base), kvSeqLenVec(base.params.num_tokens, base.params.kv_seqlen) {}
    void process(aclrtStream stream) {
        this->create_tensors();
        CHECK_ACL_SUCCESS(
            this->execute_multi_latent_attention_operator(query,
                                                          queryRope,
                                                          kvCache,
                                                          kvCacheRope,
                                                          blockTables,
                                                          contextLens,
                                                          maskOptional,
                                                          qSeqlenOptional,
                                                          qkDescaleOptional,
                                                          pvDescaleOptional,
                                                          0,
                                                          base_.params.num_heads,
                                                          base_.params.qk_scale,
                                                          base_.params.kv_heads,
                                                          0,
                                                          nullptr,
                                                          aclCreateIntArray(reinterpret_cast<int64_t *>(kvSeqLenVec.data()), kvSeqLenVec.size()),
                                                          0,
                                                          attenOutOut,
                                                          lseOutOutOptional,
                                                          stream),
            "execute_multi_latent_attention_operator failed");
    }

    void destroyTensors() {
        aclDestroyTensor(query);
        aclDestroyTensor(queryRope);
        aclDestroyTensor(kvCache);
        aclDestroyTensor(kvCacheRope);
        aclDestroyTensor(blockTables);
        aclDestroyTensor(contextLens);
    }

private:
    void create_tensors() {
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.query_shape, base_.query, &query),
                        "create query Tensor failed");
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.query_rope_shape, base_.query_rope, &queryRope),
                        "create queryRope Tensor failed");
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.kv_cache_shape, base_.kv_cache, &kvCache),
                        "create kvCache Tensor failed");
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.kv_cache_rope_shape, base_.kv_cache_rope, &kvCacheRope),
                        "create kvCacheRope Tensor failed");
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.block_tables_shape, base_.block_tables, &blockTables),
                        "create blockTables Tensor failed");
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.context_lens_shape, base_.context_lens, &contextLens),
                        "create contextLens Tensor failed");
        CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                            base_.shapes.output_shape, base_.output_op, &attenOutOut),
                        "create attenOutOut Tensor failed");
    }

    int execute_multi_latent_attention_operator(aclTensor* query,
                                                 aclTensor* queryRope,
                                                 aclTensor* kvCache,
                                                 aclTensor* kvCacheRope,
                                                 aclTensor* blockTables,
                                                 aclTensor* contextLens,
                                                 aclTensor* maskOptional,
                                                 aclTensor* qSeqlenOptional,
                                                 aclTensor* qkDescaleOptional,
                                                 aclTensor* pvDescaleOptional,
                                                 int64_t type,
                                                 int64_t headSize,
                                                 double tor,
                                                 int64_t kvHead,
                                                 int64_t maskType,
                                                 const aclIntArray* qSeqLenOptional,
                                                 const aclIntArray* kvSeqLen,
                                                 int64_t isRing,
                                                 aclTensor* attenOutOut,
                                                 aclTensor* lseOutOutOptional,
                                                 aclrtStream stream) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;

        auto ret = aclnnMultiLatentAttentionGetWorkspaceSize(query,
                                                             queryRope,
                                                             kvCache,
                                                             kvCacheRope,
                                                             blockTables,
                                                             contextLens,
                                                             maskOptional,
                                                             qSeqlenOptional,
                                                             qkDescaleOptional,
                                                             pvDescaleOptional,
                                                             type,
                                                             headSize,
                                                             tor,
                                                             kvHead,
                                                             maskType,
                                                             qSeqLenOptional,
                                                             kvSeqLen,
                                                             isRing,
                                                             attenOutOut,
                                                             lseOutOutOptional,
                                                             &workspaceSize,
                                                             &executor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMultiLatentAttentionGetWorkspaceSize failed. ERROR: %d\n",
                    ret);
            return ret);
        
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                      return ret);
        }

        ret = aclnnMultiLatentAttention(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnMultiLatentAttention failed. ERROR: %d\n", ret);
                  return ret);
        
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                  return ret);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        return ACL_SUCCESS;

    }
    aclTensor* query = nullptr;
    aclTensor* queryRope = nullptr;
    aclTensor* kvCache = nullptr;
    aclTensor* kvCacheRope = nullptr;
    aclTensor* blockTables = nullptr;
    aclTensor* contextLens = nullptr;
    aclTensor* maskOptional = nullptr;
    aclTensor* qSeqlenOptional = nullptr;
    aclTensor* qkDescaleOptional = nullptr;
    aclTensor* pvDescaleOptional = nullptr;
    aclTensor* attenOutOut = nullptr;
    aclTensor* lseOutOutOptional = nullptr;
    std::vector<int32_t> kvSeqLenVec;
    MultiLatentAttentionBase& base_;
};

} // namespace multi_latent_attention

#endif