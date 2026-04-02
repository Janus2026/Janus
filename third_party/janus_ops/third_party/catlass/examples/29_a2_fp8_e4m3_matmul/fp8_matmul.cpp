/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/fp8_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using Options = GemmOptions;

static void Run(const Options &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    constexpr uint32_t mScalar = 2;
    constexpr uint32_t nScalar = 2;
    constexpr uint32_t splitkLength = 1024;

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenWA = static_cast<size_t>(splitkLength) * 128 * mScalar;
    size_t lenWB = static_cast<size_t>(splitkLength) * 256 * nScalar;
    size_t lenWC = static_cast<size_t>(128 * mScalar) * 256 * nScalar;

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeC = lenC * sizeof(half);
    size_t sizeWA = aicCoreNum * lenWA * sizeof(half) * 2; // 双缓冲
    size_t sizeWB = aicCoreNum * lenWB * sizeof(half) * 2; // 双缓冲
    size_t sizeWC = aicCoreNum * lenWC * sizeof(float);

    size_t sizeWorkspace;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // input init
    half scalar = 1.0;
    half zeroPoint = 0;

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::string inFileAName = "../../examples/29_a2_fp8_e4m3_matmul/input/a_8.bin";
    std::ifstream inFileA(inFileAName, std::ios::binary);
    if (!inFileA.is_open()) {
        std::cerr << "Failed to open inFileA: " << inFileAName << std::endl;
    } else {
        inFileA.read(reinterpret_cast<char *>(hostA.data()), sizeA);
        inFileA.close();
    }
    std::string inFileBName = "../../examples/29_a2_fp8_e4m3_matmul/input/b_8.bin";
    std::ifstream inFileB(inFileBName, std::ios::binary);
    if (!inFileB.is_open()) {
        std::cerr << "Failed to open inFileB: " << inFileBName << std::endl;
    } else {
        inFileB.read(reinterpret_cast<char *>(hostB.data()), sizeB);
        inFileB.close();
    }

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceWA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWC), sizeWC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWorkspace{nullptr};
    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PingpongSliceK<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>; // 原子加以float类型进行累加

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    if (options.problemShape.m() > options.problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel =
            Gemm::Kernel::FP8Matmul<BlockMmad, BlockEpilogue, BlockScheduler, mScalar, nScalar, splitkLength>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB, deviceWC, scalar, zeroPoint};
        MatmulAdapter matmulOp;
        matmulOp.CanImplement(arguments);
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
            );
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum, fftsAddr);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel =
            Gemm::Kernel::FP8Matmul<BlockMmad, BlockEpilogue, BlockScheduler, mScalar, nScalar, splitkLength>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB, deviceWC, scalar, zeroPoint};
        MatmulAdapter matmulOp;
        matmulOp.CanImplement(arguments);
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
            );
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<half> hostC(lenC);
    std::vector<float> hostWC(aicCoreNum * lenWC);
    std::vector<half> hostWA(aicCoreNum * lenWA * 2);
    std::vector<half> hostWB(aicCoreNum * lenWB * 2);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostWC.data(), sizeWC, deviceWC, sizeWC, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostWA.data(), sizeWA, deviceWA, sizeWA, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostWB.data(), sizeWB, deviceWB, sizeWB, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(m * n);
    std::string outputFileName = "../../examples/29_a2_fp8_e4m3_matmul/output/expected_data.bin";
    ReadFile(outputFileName, hostGolden.data(), sizeof(float) * hostGolden.size());

    std::vector<float> hostCFP32(hostC.begin(), hostC.end());
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceWA));
    ACL_CHECK(aclrtFree(deviceWB));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}