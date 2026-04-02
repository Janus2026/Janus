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
#ifndef GROUP_GEMM_H
#define GROUP_GEMM_H

#include "aclnn_index_group_matmul.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "utils_print.h"
#include "utils_tensor.h"
namespace group_gemm {
class GroupGemmBase {
 public:
  GroupGemmBase(int32_t m,
                int32_t n,
                int32_t k,
                int32_t group_size,
                int32_t weights_per_token) {
    params.m = m;
    params.n = n;
    params.k = k;
    params.group_size = group_size;
    params.weights_per_token = weights_per_token;
    shapes.x_shape = {{params.m * 8, params.k}};
    shapes.weight_shape = {{params.group_size, params.n, params.k}};
    shapes.scale_shape = {{params.group_size, params.n}};
    shapes.per_token_scale_shape = {{params.m * 8}};
    shapes.group_list_shape = {{params.group_size}};
    shapes.y_shape = {{params.m * 8, params.n}};
  }
  void create_torch_tensors() {
    input_a = torch::randint(0, 3, {params.m * 8, params.k}, torch::kInt8);
    input_b = torch::randint(
        0, 3, {params.group_size, params.n, params.k}, torch::kInt8);
    scales = torch::rand({params.group_size, params.n}, torch::kBFloat16);
    per_token_scales = torch::rand({params.m * 8}, torch::kFloat);
    group_offset_list = GroupGemmBase::generate_groupoffset(
        params.m, params.group_size, params.weights_per_token);
    group_offset = torch::from_blob(
        group_offset_list.data(), {params.group_size}, torch::kInt64);
    output_grouped = torch::zeros({params.m * 8, params.n}, torch::kBFloat16);
    output_index = torch::zeros({params.m * 8, params.n}, torch::kBFloat16);
  }
  torch::Tensor output_grouped;
  torch::Tensor output_index;

 private:
  struct TestParams {
    int64_t m, n, k;
    int64_t group_size;
    int64_t weights_per_token;
  };

  struct TensorShapes {
    std::vector<std::vector<int64_t>> x_shape;
    std::vector<std::vector<int64_t>> weight_shape;
    std::vector<std::vector<int64_t>> scale_shape;
    std::vector<std::vector<int64_t>> per_token_scale_shape;
    std::vector<std::vector<int64_t>> group_list_shape;
    std::vector<std::vector<int64_t>> sorted_list_shape;
    std::vector<std::vector<int64_t>> y_shape;
  };

  static std::vector<int64_t> generate_groupoffset(int64_t m,
                                                   int64_t group_size,
                                                   int64_t weights_per_token) {
    std::vector<int64_t> sorted_list(m * weights_per_token);
    for (int64_t i = 0; i < m; ++i) {
      std::vector<int64_t> choices(group_size);
      std::iota(choices.begin(), choices.end(), 0);
      std::random_shuffle(choices.begin(), choices.end());
      std::copy(choices.begin(),
                choices.begin() + weights_per_token,
                sorted_list.begin() + i * weights_per_token);
    }

    std::vector<int64_t> s(group_size, 0);
    for (const auto& idx : sorted_list) {
      s[idx]++;
    }

    std::vector<int64_t> group_offset(s.size());
    std::partial_sum(s.begin(), s.end(), group_offset.begin());

    return group_offset;
  }

  int process_weight_tensor(
      const std::vector<std::vector<int64_t>>& weight_shape,
      torch::Tensor& input_b,
      aclTensorList** weight) {
    input_b = input_b.transpose(1, 2).contiguous();
    auto weight_tmp = input_b;
    weight_tmp = at_npu::native::npu_format_cast(weight_tmp.to("npu"), 29);
    std::vector<int64_t> trans_shape = {
        weight_shape[0][0], weight_shape[0][2], weight_shape[0][1]};
    int8_t* data1 = weight_tmp.data_ptr<int8_t>();
    std::vector<int64_t> strides(trans_shape.size(), 1);
    for (int64_t i = trans_shape.size() - 2; i >= 0; i--) {
      strides[i] = trans_shape[i + 1] * strides[i + 1];
    }
    std::vector<int64_t> storageShape =
        utils::get_weight_storage_shape(trans_shape);

    aclTensor* weight_nz = aclCreateTensor(trans_shape.data(),
                                           trans_shape.size(),
                                           aclDataType::ACL_INT8,
                                           strides.data(),
                                           0,
                                           aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                           storageShape.data(),
                                           storageShape.size(),
                                           data1);
    std::vector<aclTensor*> tmp{weight_nz};
    *weight = aclCreateTensorList(tmp.data(), tmp.size());

    return ACL_SUCCESS;
  }

  TestParams params;
  TensorShapes shapes;
  torch::Tensor input_a;
  torch::Tensor input_b;
  torch::Tensor scales;
  torch::Tensor per_token_scales;
  torch::Tensor group_offset;
  std::vector<int64_t> group_offset_list;

  friend class GroupGemmIndx;
  friend class GroupGemmNative;
};

class GroupGemmIndx {
 public:
  GroupGemmIndx(GroupGemmBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->createTensors();
    CHECK_ACL_SUCCESS(
        this->execute_group_gemm_operator(
            x, weight, scale, perTokenScale, groupedList, y_index, stream),
        "executeMatmulOperator failed");
  }
  void destroyTensors() {
    GroupGemmIndx::clean_up(x, weight, scale, perTokenScale, y_index);
  }

 private:
  void createTensors() {
    CHECK_ACL_SUCCESS(utils::create_tensor_list_from_torch(
                          this->base.shapes.x_shape[0], this->base.input_a, &x),
                      "create x TensorList failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_list_from_torch(
            this->base.shapes.scale_shape[0], this->base.scales, &scale),
        "create scale TensorList failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_list_from_torch(
                          this->base.shapes.per_token_scale_shape[0],
                          this->base.per_token_scales,
                          &perTokenScale),
                      "create perTokenScale TensorList failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(this->base.shapes.group_list_shape[0],
                                        this->base.group_offset,
                                        &groupedList),
        "create groupedList Tensor failed");
    CHECK_ACL_SUCCESS(
        this->base.process_weight_tensor(
            this->base.shapes.weight_shape, this->base.input_b, &weight),
        "process weight tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_list_from_torch(
            this->base.shapes.y_shape[0], this->base.output_index, &y_index),
        "create y_index TensorList failed");
  }
  static void clean_up(aclTensorList* x,
                       aclTensorList* weight,
                       aclTensorList* scale,
                       aclTensorList* perTokenScale,
                       aclTensorList* y) {
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(scale);
    aclDestroyTensorList(perTokenScale);
    aclDestroyTensorList(y);
  }
  int execute_group_gemm_operator(aclTensorList* x,
                                  aclTensorList* weight,
                                  aclTensorList* scale,
                                  aclTensorList* perTokenScale,
                                  aclTensor* groupedList,
                                  aclTensorList* y,
                                  aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    auto ret = aclnnIndexGroupMatmulGetWorkspaceSize(x,
                                                     weight,
                                                     scale,
                                                     perTokenScale,
                                                     groupedList,
                                                     y,
                                                     &workspaceSize,
                                                     &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n",
                  ret);
        return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret =
          aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                return ret);
    }

    ret = aclnnIndexGroupMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnIndexGroupMatmul failed. ERROR: %d\n", ret);
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
  aclTensor* groupedList = nullptr;
  aclTensorList* perTokenScale = nullptr;
  aclTensorList* x = nullptr;
  aclTensorList* weight = nullptr;
  aclTensorList* scale = nullptr;
  aclTensorList* y_index = nullptr;
  GroupGemmBase base;
};

class GroupGemmNative {
 public:
  GroupGemmNative(GroupGemmBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->createTensors();
    CHECK_ACL_SUCCESS(
        this->execute_group_gemm_native_operator(
            x, weight, scale, perTokenScale, groupedList, y_grouped, stream),
        "executeMatmulOperator failed");
  }
  void destroyTensors() {
    GroupGemmNative::clean_up(x, weight, scale, perTokenScale, y_grouped);
  }

 private:
  void createTensors() {
    CHECK_ACL_SUCCESS(utils::create_tensor_list_from_torch(
                          this->base.shapes.x_shape[0], this->base.input_a, &x),
                      "create x TensorList failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_list_from_torch(
            this->base.shapes.scale_shape[0], this->base.scales, &scale),
        "create scale TensorList failed");
    CHECK_ACL_SUCCESS(utils::create_tensor_list_from_torch(
                          this->base.shapes.per_token_scale_shape[0],
                          this->base.per_token_scales,
                          &perTokenScale),
                      "create perTokenScale TensorList failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(this->base.shapes.group_list_shape[0],
                                        this->base.group_offset,
                                        &groupedList),
        "create groupedList Tensor failed");
    CHECK_ACL_SUCCESS(
        this->base.process_weight_tensor(
            this->base.shapes.weight_shape, this->base.input_b, &weight),
        "process weight tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_list_from_torch(this->base.shapes.y_shape[0],
                                             this->base.output_grouped,
                                             &y_grouped),
        "create y_grouped TensorList failed");
  }
  static void clean_up(aclTensorList* x,
                       aclTensorList* weight,
                       aclTensorList* scale,
                       aclTensorList* perTokenScale,
                       aclTensorList* y) {
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(scale);
    aclDestroyTensorList(perTokenScale);
    aclDestroyTensorList(y);
  }
  int execute_group_gemm_native_operator(aclTensorList* x,
                                         aclTensorList* weight,
                                         aclTensorList* scale,
                                         aclTensorList* perTokenScale,
                                         aclTensor* groupedList,
                                         aclTensorList* y,
                                         aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclTensorList* bias = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* activationInput = nullptr;
    aclTensorList* activationQuantScale = nullptr;
    aclTensorList* activationQuantOffset = nullptr;
    aclTensorList* activationFeatureOut = nullptr;
    aclTensorList* dynQuantScaleOut = nullptr;
    int64_t splitItem = 2;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;

    auto ret = aclnnGroupedMatmulV4GetWorkspaceSize(x,
                                                    weight,
                                                    bias,
                                                    scale,
                                                    offset,
                                                    antiquantScale,
                                                    antiquantOffset,
                                                    perTokenScale,
                                                    groupedList,
                                                    activationInput,
                                                    activationQuantScale,
                                                    activationQuantOffset,
                                                    splitItem,
                                                    groupType,
                                                    groupListType,
                                                    actType,
                                                    y,
                                                    activationFeatureOut,
                                                    dynQuantScaleOut,
                                                    &workspaceSize,
                                                    &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n",
                  ret);
        return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret =
          aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                return ret);
    }

    ret = aclnnGroupedMatmulV4(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnIndexGroupMatmul failed. ERROR: %d\n", ret);
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
  aclTensor* groupedList = nullptr;
  aclTensorList* perTokenScale = nullptr;
  aclTensorList* x = nullptr;
  aclTensorList* weight = nullptr;
  aclTensorList* scale = nullptr;
  aclTensorList* y_grouped = nullptr;
  GroupGemmBase base;
};

}  // namespace group_gemm

#endif