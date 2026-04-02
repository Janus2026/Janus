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
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "pp_matmul.h"
#include "utils_tensor.h"

class PPMatmulTest : public ::testing::Test {
protected:
    int32_t deviceId = 15;
    aclrtStream stream1;
    
    void SetUp() override {
        utils::initialize_acl(deviceId, &stream1);
    }

    void TearDown() override {
        aclrtDestroyStream(stream1);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

TEST_F(PPMatmulTest, BasicMatmulCorrectness) {
    pp_matmul::PPMatmulBase inputs(16, 12800, 5120);
    inputs.create_torch_tensors();
    pp_matmul::PPMatmulOp op_gemm(inputs);
    op_gemm.process(stream1);
    pp_matmul::PPMatmulNative native_gemm(inputs);
    native_gemm.process(stream1);
    EXPECT_TRUE(torch::equal(inputs.output_native, inputs.output_op))
        << "Custom op output does not match native output";
    native_gemm.destroy_tensors();
    op_gemm.destroy_tensors();
}

TEST_F(PPMatmulTest, DifferentSizes) {
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {1, 12800, 5120},
        {2, 12800, 5120},
        {4, 12800, 5120}
    };
    
    for (auto [m, n, k] : test_sizes) {
        pp_matmul::PPMatmulBase inputs(m, n, k);
        inputs.create_torch_tensors();
        
        pp_matmul::PPMatmulOp op_gemm(inputs);
        op_gemm.process(stream1);
        
        pp_matmul::PPMatmulNative native_gemm(inputs);
        native_gemm.process(stream1);
        
        EXPECT_TRUE(torch::equal(inputs.output_native, inputs.output_op))
            << "Failed for size: m=" << m << ", n=" << n << ", k=" << k;
        
        native_gemm.destroy_tensors();
        op_gemm.destroy_tensors();
    }
}