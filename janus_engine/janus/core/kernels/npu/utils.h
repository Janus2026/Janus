#pragma once

#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <string>
#include <vector>

#include "acl/acl.h"
#include "util/tensor_helper.h"

namespace janus::kernel::npu {
struct type_info {
  static aclDataType get_acl_type(const torch::ScalarType& dtype);
};

void create_acltensor(aclTensor** tensor, const torch::Tensor& tensor_data);
void check_tensor(const torch::Tensor& t,
                  const std::string& name,
                  const std::string& func_name = "");
void check_tensor_shapes_equal(const torch::Tensor& a,
                               const torch::Tensor& b,
                               const std::string& func_name = "");
}  // namespace janus::kernel::npu
