#pragma once
// #ifndef ATTENTION_TEST_H
// #define ATTENTION_TEST_H

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "aclnn_x_attention.h"
#include "utils_tensor.h"

namespace xattention {
struct TensorShapes {
  std::vector<std::vector<int64_t>> shared_query_shape;
  std::vector<std::vector<int64_t>> shared_key_shape;
  std::vector<std::vector<int64_t>> shared_value_shape;
  std::vector<std::vector<int64_t>> unshared_key_shape;
  std::vector<std::vector<int64_t>> unshared_value_shape;
  std::vector<std::vector<int64_t>> unshared_query_shape;
  std::vector<std::vector<int64_t>> output_shared_shape;
  std::vector<std::vector<int64_t>> output_unshared_shape;
  std::vector<std::vector<int64_t>> shared_exp_shape;
  std::vector<std::vector<int64_t>> unshared_exp_shape;
  std::vector<std::vector<int64_t>> shared_max_shape;
  std::vector<std::vector<int64_t>> unshared_max_shape;
  std::vector<std::vector<int64_t>> output_shape;
};
class XAttentionBase {
 public:
  XAttentionBase(int32_t batch_size,
                      int32_t beam_size,
                      int32_t head_num,
                      int32_t q_length,
                      int32_t head_size,
                      int32_t shared_kv_length,
                      int32_t unshared_kv_length
                    ) {
    this->batch_size_ = batch_size;
    this->beam_size_ = beam_size;
    this->head_num_ = head_num;
    this->q_length_ = q_length;
    this->head_size_ = head_size;
    this->shared_kv_length_ = shared_kv_length;
    this->unshared_kv_length_ = unshared_kv_length;
    this->block_m_ = 64;
    this->block_n_ = 64;
    this->tiles_q_val_ = (q_length + this->block_m_ - 1) / this->block_m_;
    shapes.shared_query_shape = {{batch_size_, beam_size_, head_num_,  head_size_}};
    shapes.shared_key_shape = {{batch_size_, head_num_, shared_kv_length_, head_size_}};
    shapes.shared_value_shape = {{batch_size_, head_num_, shared_kv_length_, head_size_}};
    shapes.unshared_key_shape = {{batch_size_ * beam_size_, head_num_, unshared_kv_length_, head_size_}};
    shapes.unshared_value_shape = {{batch_size_ * beam_size_, head_num_, unshared_kv_length_, head_size_}};
    shapes.unshared_query_shape = {{batch_size_ * beam_size_, head_num_,head_size_}};
    shapes.shared_exp_shape = {{batch_size_, head_num_, beam_size_}};
    shapes.unshared_exp_shape = {{batch_size_, head_num_, beam_size_}};
    shapes.shared_max_shape = {{batch_size_, head_num_, beam_size_}};
    shapes.unshared_max_shape = {{batch_size_, head_num_, beam_size_}};
    shapes.output_shared_shape = {{batch_size_ * beam_size_, head_num_, q_length_, head_size_}};
    shapes.output_unshared_shape = {{batch_size_ * beam_size_, head_num_, q_length_, head_size_}};
    shapes.output_shape = {{batch_size_ * beam_size_, head_num_, q_length_, head_size_}};
    }
  void create_torch_tensor() {
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32);
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32);
    auto opt_f16 = torch::TensorOptions().dtype(torch::kFloat16);
    auto opt_i8 = torch::TensorOptions().dtype(torch::kInt8);
    this->shared_query =
        torch::randint(/*low=*/4,
                       /*high=*/10,
                       {batch_size_, head_num_, beam_size_, head_size_},
                       opt_f16);
    this->shared_key =
        torch::randint(/*low=*/4,
                       /*high=*/10,
                       {batch_size_, head_num_, shared_kv_length_, head_size_},
                       opt_f16);
    this->shared_value =
        torch::randint(/*low=*/4,
                       /*high=*/10,
                       {batch_size_, head_num_, shared_kv_length_, head_size_},
                       opt_f16);
    
                       this->output =
        torch::zeros({batch_size_, head_num_, q_length_, head_size_}, opt_f16);
    this->unshared_query = this->shared_query.view({batch_size_ * beam_size_, head_num_, q_length_, head_size_});
    this->unshared_key =
        torch::randint(/*low=*/4,
                       /*high=*/10,
                       {batch_size_ * beam_size_, head_num_, unshared_kv_length_, head_size_},
                       opt_f16);
    this->unshared_value =
        torch::randint(/*low=*/4,
                       /*high=*/10,
                       {batch_size_ * beam_size_, head_num_, unshared_kv_length_, head_size_},
                       opt_f16);
    this->output_shared =
        torch::zeros({batch_size_ * beam_size_, head_num_, q_length_, head_size_}, opt_f16);
    this->output_unshared =
        torch::zeros({batch_size_ * beam_size_, head_num_, q_length_, head_size_}, opt_f16);
    this->output =
        torch::zeros({batch_size_ * beam_size_, head_num_, q_length_, head_size_}, opt_f16);
    this->shared_exp =
        torch::zeros({batch_size_, head_num_, beam_size_}, opt_f16);
    this->unshared_exp =
        torch::zeros({batch_size_, head_num_, beam_size_}, opt_f16);
    this->shared_max =
        torch::zeros({batch_size_, head_num_, beam_size_}, opt_f16);
    this->unshared_max =
        torch::zeros({batch_size_, head_num_, beam_size_}, opt_f16);
  }

  int32_t q_length_;
  int32_t shared_kv_length_;
  int32_t unshared_kv_length_;
  int32_t batch_size_;
  int32_t beam_size_;
  int32_t head_num_;
  int32_t head_size_;
  int32_t tiles_q_val_;
  int32_t block_m_;
  int32_t block_n_;
  torch::Tensor shared_query;
  torch::Tensor shared_key;
  torch::Tensor shared_value;
  torch::Tensor unshared_key;
  torch::Tensor unshared_value;
  torch::Tensor unshared_query;
  torch::Tensor output_shared;
  torch::Tensor output_unshared;
  torch::Tensor shared_exp;
  torch::Tensor unshared_exp;
  torch::Tensor shared_max;
  torch::Tensor unshared_max;
  torch::Tensor output;
  TensorShapes shapes;
};
class XAttentionOp {
 public:
  XAttentionOp(XAttentionBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->create_tensors();
    CHECK_ACL_SUCCESS(execute_x_attention_operator(
                          shared_query, shared_key, shared_value, unshared_query, unshared_key, unshared_value, output_shared, output_unshared, shared_exp, unshared_exp, shared_max, unshared_max, output, stream),
                      "execute_x_attention_operator failed");
  }
  void create_tensors() {
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.shared_query_shape[0], base.shared_query, &shared_query),
                      "create query Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.shared_key_shape[0], base.shared_key, &shared_key),
        "create shared_key Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(base.shapes.shared_value_shape[0],
                                        base.shared_value,
                                        &shared_value),
        "create shared_value Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.unshared_query_shape[0],
                                        base.unshared_query,
                                        &unshared_query),
        "create unshared_query Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.unshared_key_shape[0],
                                        base.unshared_key,
                                        &unshared_key),
        "create unshared_key Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.unshared_value_shape[0],
                                        base.unshared_value,
                                        &unshared_value),
        "create unshared_value Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.output_shared_shape[0],
                                        base.output_shared,
                                        &output_shared),
        "create output_shared Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.output_unshared_shape[0],
                                        base.output_unshared,
                                        &output_unshared),
        "create output_unshared Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.shared_exp_shape[0],
                                        base.shared_exp,
                                        &shared_exp),
        "create shared_exp Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.unshared_exp_shape[0],
                                        base.unshared_exp,
                                        &unshared_exp),
        "create unshared_exp Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.shared_max_shape[0],
                                        base.shared_max,
                                        &shared_max),
        "create shared_max Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.unshared_max_shape[0],
                                        base.unshared_max,
                                        &unshared_max),
        "create unshared_max Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(base.shapes.output_shape[0],
                                        base.output,
                                        &output),
        "create output Tensor failed");
  }
  int execute_x_attention_operator(aclTensor* shared_query,
                                        aclTensor* shared_key,
                                        aclTensor* shared_value,
                                        aclTensor* unshared_query,
                                        aclTensor* unshared_key,
                                        aclTensor* unshared_value,
                                        aclTensor* output_shared,
                                        aclTensor* output_unshared,
                                        aclTensor* shared_exp,
                                        aclTensor* unshared_exp,
                                        aclTensor* shared_max,
                                        aclTensor* unshared_max,
                                        aclTensor* output,
                                        aclrtStream stream);
 void destroy_tensors() {
    aclDestroyTensor(shared_query);
    aclDestroyTensor(shared_key);
    aclDestroyTensor(shared_value);
    aclDestroyTensor(unshared_query);
    aclDestroyTensor(unshared_key);
    aclDestroyTensor(unshared_value);
    aclDestroyTensor(output_shared);
    aclDestroyTensor(output_unshared);
    aclDestroyTensor(shared_exp);
    aclDestroyTensor(unshared_exp);
    aclDestroyTensor(shared_max);
    aclDestroyTensor(unshared_max);
    aclDestroyTensor(output);
  }
 private:
  aclTensor* shared_query = nullptr;
  aclTensor* shared_key = nullptr;
  aclTensor* shared_value = nullptr;
  aclTensor* unshared_key = nullptr;
  aclTensor* unshared_value = nullptr;
  aclTensor* unshared_query = nullptr;
  aclTensor* output_shared = nullptr;
  aclTensor* output_unshared = nullptr;
  aclTensor* shared_exp = nullptr;
  aclTensor* unshared_exp = nullptr;
  aclTensor* shared_max = nullptr;
  aclTensor* unshared_max = nullptr;
  aclTensor* output = nullptr;
 
  XAttentionBase& base;
};
int XAttentionOp::execute_x_attention_operator(
    aclTensor* shared_query,
    aclTensor* shared_key,
    aclTensor* shared_value,
    aclTensor* unshared_query,
    aclTensor* unshared_key,
    aclTensor* unshared_value,
    aclTensor* output_shared,
    aclTensor* output_unshared,
    aclTensor* shared_exp,
    aclTensor* unshared_exp,
    aclTensor* shared_max,
    aclTensor* unshared_max,
    aclTensor* output,
    aclrtStream stream
) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto ret = aclnnXAttentionGetWorkspaceSize(
      shared_query, shared_key, shared_value, unshared_query, unshared_key, unshared_value, output_shared, output_unshared, shared_exp, unshared_exp, shared_max, unshared_max, output,&workspaceSize, &executor);
  CHECK_ACL_SUCCESS(
      ret, "aclnn_x_attention_operator get workspace size failed");
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL_SUCCESS(ret, "aclnn_x_attention_operator malloc workspace failed");
  }
  ret = aclnnXAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_x_attention_operator execute failed");
  ret = aclrtSynchronizeStream(stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_x_attention_operator synchronize stream failed");
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return ACL_SUCCESS;
}
}