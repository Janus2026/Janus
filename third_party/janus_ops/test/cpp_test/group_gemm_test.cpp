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
#include <tuple>

#include "group_gemm.h"
#include "utils_tensor.h"

class GroupGemmTest : public ::testing::Test {
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

TEST_F(GroupGemmTest, BasicMatmulCorrectness) {
    group_gemm::GroupGemmBase inputs(56, 7168, 256, 16, 8);
    inputs.create_torch_tensors();
    group_gemm::GroupGemmIndx op_gemm(inputs);
    op_gemm.process(stream1);
    group_gemm::GroupGemmNative native_gemm(inputs);
    native_gemm.process(stream1);
    EXPECT_TRUE(torch::equal(inputs.output_index, inputs.output_grouped))
        << "Custom op output does not match native output";
    native_gemm.destroyTensors();
    op_gemm.destroyTensors();
}

TEST_F(GroupGemmTest, DifferentSizes) {
    std::vector<std::tuple<int, int, int, int, int>> test_sizes = {
        {56, 7168, 256, 16, 8},
        {112, 7168, 512, 32, 8},
        {224, 7168, 1024, 64, 8}
    };

    for (auto [m, n, k, group_size, weights_per_token] : test_sizes) {
        group_gemm::GroupGemmBase inputs(m, n, k, group_size, weights_per_token);
        inputs.create_torch_tensors();
        group_gemm::GroupGemmIndx op_gemm(inputs);
        op_gemm.process(stream1);
        group_gemm::GroupGemmNative native_gemm(inputs);
        native_gemm.process(stream1);
        EXPECT_TRUE(torch::equal(inputs.output_index, inputs.output_grouped))
            << "Failed for size: m=" << m << ", n=" << n << ", k=" << k;
        native_gemm.destroyTensors();
        op_gemm.destroyTensors();
    }
}
