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

#include "beam_search.h"
#include "utils_tensor.h"

class BeamSearchTest : public ::testing::Test {
protected:
    int32_t deviceId = 9;
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

TEST_F(BeamSearchTest, BasicCorrectness) {
    // basic test：only set log_probs[0]=1, top_probs[0][0]=1
    beam_search::BeamSearchBase::ProbGenerator basic_gen =
        [](int64_t n_sequences, int64_t top_k, const torch::TensorOptions& opt_f32,
           torch::Tensor& log_probs, torch::Tensor& top_probs) {
            log_probs = torch::zeros({n_sequences, 1}, opt_f32);
            top_probs = torch::zeros({n_sequences, top_k}, opt_f32);
            log_probs.index_put_({0, 0}, 1.0f);
            top_probs.index_put_({0, 0}, 1.0f);
        };

    beam_search::BeamSearchBase inputs(2, 2, 2, 2, basic_gen);
    inputs.create_torch_tensors();

    beam_search::BeamSearchTorch op_torch(inputs);
    op_torch.process();
    beam_search::BeamSearchOp op_cann(inputs);
    op_cann.process(stream1);

    // Compare on CPU
    auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
    auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
    auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
    auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
    auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
    auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});

    EXPECT_TRUE(torch::equal(out_torch_log_cpu, out_op_log_cpu)) << "Log probs mismatch";

    op_cann.destroy_tensors();
}

void run_beam_search_test(int beam_width,
                          int top_k,
                          int request_num,
                          int sequence_length,
                          aclrtStream& stream1) {
    beam_search::BeamSearchBase inputs(beam_width, top_k, request_num, sequence_length);
    inputs.create_torch_tensors();

    beam_search::BeamSearchTorch op_torch(inputs);
    op_torch.process();
    beam_search::BeamSearchOp op_cann(inputs);
    try {
        op_cann.process(stream1);
    } catch (...) {
        std::cout << "Exception occurred for size: beam_width=" << beam_width
                  << ", top_k=" << top_k << ", request_num=" << request_num
                  << ", sequence_length=" << sequence_length << std::endl;
        op_cann.destroy_tensors();
        throw;
    }
    auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
    auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
    auto out_torch_index_cpu = inputs.output_token_index_torch.to(torch::kCPU);
    auto out_op_index_cpu = inputs.output_token_index_op.to(torch::kCPU);
    auto out_torch_log_cpu = inputs.output_log_probs_torch.to(torch::kCPU).view({-1, 1});
    auto out_op_log_cpu = inputs.output_log_probs_op.to(torch::kCPU).view({-1, 1});

    bool all_equal = torch::equal(out_torch_cpu, out_op_cpu) &&
                     torch::equal(out_torch_index_cpu, out_op_index_cpu) &&
                     torch::equal(out_torch_log_cpu, out_op_log_cpu);
    if (!all_equal) {
       int64_t out_mismatch_count = (out_torch_cpu != out_op_cpu).sum().item<int64_t>();
       int64_t out_index_mismatch_count = (out_torch_index_cpu != out_op_index_cpu).sum().item<int64_t>();
       int64_t out_log_mismatch_count = (out_torch_log_cpu != out_op_log_cpu).sum().item<int64_t>();
       std::cout << "Mismatch counts for size: beam_width=" << beam_width
                 << ", top_k=" << top_k << ", request_num=" << request_num
                 << ", sequence_length=" << sequence_length << std::endl;
       std::cout << "  output_token_ids mismatch count: " << out_mismatch_count << std::endl;
       std::cout << "  output_token_index mismatch count: " << out_index_mismatch_count << std::endl;
       std::cout << "  output_log_probs mismatch count: " << out_log_mismatch_count << std::endl;
    }
    EXPECT_TRUE(all_equal)
        << "Failed for size: beam_width=" << beam_width
        << ", top_k=" << top_k << ", request_num=" << request_num
        << ", sequence_length=" << sequence_length;

    op_cann.destroy_tensors();
}

TEST_F(BeamSearchTest, DifferentSizes) {
    std::vector<std::tuple<int, int, int, int>> test_sizes = {
        {1, 1, 2, 2},
        {2, 2, 2, 2},
        {4, 4, 2, 2},
        {5, 5, 2, 2},
        {8, 8, 2, 2},
        {16, 16, 2, 2},
        {31, 31, 2, 2},
        {32, 32, 2, 2},
        {47, 47, 2, 2},
        {64, 64, 2, 2},
        {128, 128, 2, 2},
        {256, 256, 2, 2},
        {511, 511, 2, 2},
        {512, 512, 2, 2},
        {1024, 1024, 2, 2},
        {2048, 2048, 2, 2}
    };
    for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes)
    {
        run_beam_search_test(beam_width, top_k, request_num, sequence_length,
        stream1);
    }
}

TEST_F(BeamSearchTest, DifferentRequestNums) {
    std::vector<std::tuple<int, int, int, int>> test_sizes = {
        {512, 512, 1, 1},
        {512, 512, 2, 2},
        {512, 512, 4, 4},
        {512, 512, 8, 8}
    };
    for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes) {
        run_beam_search_test(beam_width, top_k, request_num, sequence_length, stream1);
    }
}

TEST_F(BeamSearchTest, RandomSizeTests) {
    std::vector<std::tuple<int, int, int, int>> test_sizes{};
    int testcases = 10;
    for (int i = 0; i < testcases; ++i) {
        int beam_width = rand() % 512 + 1;       // 1 to 512
        int top_k = beam_width;
        int request_num = rand() % 16 + 1;       // 1 to 16
        int sequence_length = request_num;
        test_sizes.emplace_back(beam_width, top_k, request_num, sequence_length);
    }
    for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes) {
        run_beam_search_test(beam_width, top_k, request_num, sequence_length, stream1);
    }
}