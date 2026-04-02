#pragma once
#ifndef BASE_UTILS_H
#define BASE_UTILS_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <limits>
#include <string>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <numeric>

#include "acl/acl.h"
#include "aclnn_index_group_matmul.h"
#include "acl/acl_op_compiler.h"
#include "shape.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
// #include "utils_macros.h"
#include "utils_print.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)

#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR] " fmt "\n", ##args)

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)

namespace utils {

struct type_info {
  static aclDataType get_acl_type(const torch::ScalarType& dtype) {
    switch (dtype) {
      case torch::kInt64:
        return ACL_INT64;
      case torch::kInt32:
        return ACL_INT32;
      case torch::kFloat32:
        return ACL_FLOAT;
      case torch::kInt16:
        return ACL_INT16;
      case torch::kFloat16:
        return ACL_FLOAT16;
      case torch::kBFloat16:
        return ACL_BF16;
      case torch::kInt8:
        return ACL_INT8;
      default:
        return ACL_INT32;
    }
  }
};


int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    
    return 0;
}

std::vector<int64_t> get_weight_storage_shape(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> storageTensorDims (5, 0);  // ND格式下，storageShape和originalShape一致
        storageTensorDims[0] = shape[0];
        storageTensorDims[1] = 1 + ((shape[1] - 1) / 16);  // 1, 16：1: 维度, 16: padding大小
        storageTensorDims[2] = 1 + ((shape[2] - 1) / 16);  // 2, 16：1: 维度, 16: padding大小
        storageTensorDims[3] = 16;  // 3, 16：NZ格式要求
        storageTensorDims[4] = 16;  // 4, 16：NZ格式要求
    return storageTensorDims;
}

std::vector<int64_t> get_transpose_tensor_stride(std::vector<int64_t> shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    strides[shape.size() - 1] = shape[shape.size() - 1];
    if (shape.size() == 3) {
        strides[0] = shape[1]*shape[2];
    }
    return strides;
}

void create_acltensor(aclTensor** tensor,
                             torch::Tensor& tensor_data) {
  aclDataType acl_tensor_type =
      utils::type_info::get_acl_type(tensor_data.scalar_type());
  tensor_data = tensor_data.to("npu");
  void* deviceData = const_cast<void*>(tensor_data.storage().data());
  c10::SmallVector<int64_t, 8> storageDims;
  storageDims.push_back(tensor_data.storage().nbytes() /
                        tensor_data.itemsize());
  *tensor = aclCreateTensor(tensor_data.sizes().data(),
                            tensor_data.sizes().size(),
                            acl_tensor_type,
                            tensor_data.strides().data(),
                            tensor_data.storage_offset(),
                            aclFormat::ACL_FORMAT_ND,
                            storageDims.data(),
                            storageDims.size(),
                            deviceData);
  if (*tensor == nullptr) {
    LOG(ERROR) << "create_acltensor: failed to create acltensor";
    throw std::runtime_error("create_acltensor: failed to create acltensor");
  }
}

int create_tensor_list_from_torch(const std::vector<int64_t>& shape,
                          torch::Tensor& tensor,
                          aclTensorList** tensor_list) {
  aclTensor* tensor_acl;
  utils::create_acltensor(&tensor_acl, tensor);
  std::vector<aclTensor*> tmp{tensor_acl};
  *tensor_list = aclCreateTensorList(tmp.data(), tmp.size());
  return ACL_SUCCESS;
}

int create_tensor_from_torch(const std::vector<int64_t>& shape,
                          torch::Tensor& tensor,
                          aclTensor** tensor_acl) {
  utils::create_acltensor(tensor_acl, tensor);
  return ACL_SUCCESS;
}

// 初始化ACL环境
int initialize_acl(int32_t deviceId, aclrtStream* stream) {
  auto ret = utils::Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);
  return ACL_SUCCESS;
}
} // namespace utils

#endif // BASE_UTILS_H 