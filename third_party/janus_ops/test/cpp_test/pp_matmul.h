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
#ifndef PP_MATMUL_H
#define PP_MATMUL_H

#include "aclnn_pp_mat_mul.h"
#include "aclnn_pp_matmul_opt.h"
#include "utils_print.h"
#include "utils_tensor.h"

namespace pp_matmul {
struct TensorShapes {
  std::vector<std::vector<int64_t>> x1_shape;
  std::vector<std::vector<int64_t>> x2_shape;
  std::vector<std::vector<int64_t>> y_shape;
};
class PPMatmulBase {
 public:
  PPMatmulBase(int64_t m, int64_t n, int64_t k) : m_(m), n_(n), k_(k) {
    shapes.x1_shape = {{m_, k_}};
    shapes.x2_shape = {{n_, k_}};
    shapes.y_shape = {{m_, n_}};
  }
  void create_torch_tensors() {
    input_a = torch::randint(0, 3, {m_, k_}, torch::kBFloat16);
    input_b = torch::randint(0, 3, {n_, k_}, torch::kBFloat16);
    output_native = torch::zeros({m_, n_}, torch::kBFloat16);
    output_op = torch::zeros({m_, n_}, torch::kBFloat16);
  }
  int32_t m_;
  int32_t n_;
  int32_t k_;
  torch::Tensor input_a;
  torch::Tensor input_b;
  torch::Tensor output_native;
  torch::Tensor output_op;
  TensorShapes shapes;
};

class PPMatmulNative {
 public:
  PPMatmulNative(PPMatmulBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->create_tensors();
    CHECK_ACL_SUCCESS(execute_pp_matmul_operator(a, b, y, stream),
                      "execute_pp_matmul_operator failed");
  }
  int execute_pp_matmul_operator(aclTensor* input_a,
                                 aclTensor* input_b,
                                 aclTensor* output_native,
                                 aclrtStream stream);
  void destroy_tensors() {
    aclDestroyTensor(a);
    aclDestroyTensor(b);
    aclDestroyTensor(y);
  }

 private:
  void create_tensors() {
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.x1_shape[0], base.input_a, &a),
                      "create input_a Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.x2_shape[0], base.input_b, &b),
                      "create input_b Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.y_shape[0], base.output_native, &y),
                      "create output_native Tensor failed");
  }
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* y = nullptr;
  PPMatmulBase& base;
};
int PPMatmulNative::execute_pp_matmul_operator(aclTensor* input_a,
                                               aclTensor* input_b,
                                               aclTensor* output_native,
                                               aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclTensor* bias = nullptr;
  aclTensor* yRef = nullptr;
  bool trans_a = false;
  bool trans_b = true;
  bool en_shuff = false;
  int64_t compute_type = 0;

  auto ret = aclnnPpMatMulGetWorkspaceSize(input_a,
                                           input_b,
                                           bias,
                                           yRef,
                                           trans_a,
                                           trans_b,
                                           en_shuff,
                                           compute_type,
                                           output_native,
                                           &workspaceSize,
                                           &executor);
  CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_operator get workspace size failed");
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_operator malloc workspace failed");
  }
  ret = aclnnPpMatMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_operator execute failed");

  ret = aclrtSynchronizeStream(stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_operator synchronize stream failed");

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return ACL_SUCCESS;
}

class PPMatmulOp {
 public:
  PPMatmulOp(PPMatmulBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->create_tensors();
    CHECK_ACL_SUCCESS(execute_pp_matmul_op_operator(a, b, y, stream),
                      "execute_pp_matmul_op_operator failed");
  }
  int execute_pp_matmul_op_operator(aclTensor* input_a,
                                    aclTensor* input_b,
                                    aclTensor* output_op,
                                    aclrtStream stream);
  void destroy_tensors() {
    aclDestroyTensor(a);
    aclDestroyTensor(b);
    aclDestroyTensor(y);
  }

 private:
  void create_tensors() {
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.x1_shape[0], base.input_a, &a),
                      "create input_a Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.x2_shape[0], base.input_b, &b),
                      "create input_b Tensor failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_from_torch(
                          base.shapes.y_shape[0], base.output_op, &y),
                      "create output_op Tensor failed");
  }
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* y = nullptr;
  PPMatmulBase& base;
};
int PPMatmulOp::execute_pp_matmul_op_operator(aclTensor* input_a,
                                              aclTensor* input_b,
                                              aclTensor* output_op,
                                              aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto ret = aclnnPpMatmulOptGetWorkspaceSize(
      input_a, input_b, output_op, &workspaceSize, &executor);
  CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_op_operator failed");
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_op_operator failed");
  }
  ret = aclnnPpMatmulOpt(workspaceAddr, workspaceSize, executor, stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_op_operator failed");

  ret = aclrtSynchronizeStream(stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_pp_matmul_op_operator synchronize stream failed");

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return ACL_SUCCESS;
}
}  // namespace pp_matmul

#endif