/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LIBRARY_GEMM_OPERATION_H
#define CATLASS_LIBRARY_GEMM_OPERATION_H

#include <type_traits>
#include "catlass/library/operation.h"
#include "library_utils.h"

namespace Catlass {
namespace Library {

template <typename Operator_>
class GemmOperationBase : public Operation {
public:
    using Operator = Operator_;
    using OperatorArguments = typename Operator::Arguments;
    using OperatorKernel = typename Operator::Kernel;

    using ElementA = typename OperatorKernel::ElementA;
    using ElementB = typename OperatorKernel::ElementB;
    using ElementC = typename OperatorKernel::ElementC;
    using LayoutA = typename OperatorKernel::LayoutA;
    using LayoutB = typename OperatorKernel::LayoutB;
    using LayoutC = typename OperatorKernel::LayoutC;
    using BlockMmad = typename OperatorKernel::BlockMmad;
    using ArchTag = typename OperatorKernel::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using BlockScheduler = typename OperatorKernel::BlockScheduler;

    GemmOperationBase(char const *name = "")
    {
        this->description_.name = name;
        this->description_.kind = OperationKind::Gemm;

        this->description_.A = MakeTensorDescription<ElementA, LayoutA>();
        this->description_.B = MakeTensorDescription<ElementB, LayoutB>();
        this->description_.C = MakeTensorDescription<ElementC, LayoutC>();

        this->description_.tileDescription.L1TileShape =
            GemmShapeDescription(L1TileShape::M, L1TileShape::N, L1TileShape::K);
        this->description_.tileDescription.L0TileShape =
            GemmShapeDescription(L0TileShape::M, L0TileShape::N, L0TileShape::K);
    }

    virtual OperationDescription const &GetDescription() const override
    {
        return this->description_;
    }

    virtual Status CanImplement(void *argsPtr, void *configPtr) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.CanImplement(this->args_);
    }

    virtual size_t GetWorkspaceSize(void *argsPtr, void *configPtr) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.GetWorkspaceSize(this->args_);
    }

    virtual Status Initialize(
        void *argsPtr,
        void *configPtr,
        uint8_t *workspace,
        aclrtStream stream
    ) override
    {
        BuildArgs(argsPtr, configPtr);
        return op_.Initialize(this->args_, workspace, stream);
    }

    virtual Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr) override
    {
        return op_.Run(stream, blockDim, fftsAddr);
    }

protected:
    virtual void BuildArgs(void *argsPtr, void *configPtr) = 0;

    GemmOperationDescription description_;
    OperatorArguments args_{};
    Operator op_;
};

/********************* basic matmul *********************/
template <typename Operator_>
class BasicMatmulGemmOperation : public GemmOperationBase<Operator_> {
public:
    BasicMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_>(name)
    {
        this->description_.gemmKind = GemmKind::BasicMatmul;
    }

    virtual Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr) override
    {
        (void)fftsAddr;
        return this->op_.Run(stream, blockDim, 0);
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        BasicMatmulGemmArguments *arguments = (BasicMatmulGemmArguments *)argsPtr;
        BasicMatmulGemmConfiguration *config = (BasicMatmulGemmConfiguration *)configPtr;
        this->args_.problemShape = GemmCoord{config->m, config->n, config->k};
        this->args_.ptrA = arguments->A;
        this->args_.ptrB = arguments->B;
        this->args_.ptrC = arguments->C;
    }
};
/********************* basic matmul end *********************/

/********************* grouped matmul *********************/
template <typename Operator_>
class GroupedMatmulGemmOperation : public GemmOperationBase<Operator_> {
public:
    GroupedMatmulGemmOperation(char const *name = "") : GemmOperationBase<Operator_>(name)
    {
        this->description_.gemmKind = GemmKind::GroupedMatmul;
    }

private:
    virtual void BuildArgs(void *argsPtr, void *configPtr) override
    {
        GroupedMatmulGemmArguments *arguments = (GroupedMatmulGemmArguments *)argsPtr;
        GroupedMatmulGemmConfiguration *config = (GroupedMatmulGemmConfiguration *)configPtr;

        this->args_.problemCount = config->groupCount;
        this->args_.ptrProblemShape = arguments->problemShapeList;
        this->args_.ptrA = arguments->A;
        this->args_.ptrLayoutA = arguments->layoutAList;
        this->args_.ptrB = arguments->B;
        this->args_.ptrLayoutB = arguments->layoutBList;
        this->args_.ptrC = arguments->C;
        this->args_.ptrLayoutC = arguments->layoutCList;
    }

    virtual Status Run(
        aclrtStream stream,
        uint32_t blockDim,
        uint64_t fftsAddr
    ) override
    {
        (void)fftsAddr;
        return this->op_.Run(stream, blockDim, 0);
    }
};
/********************* grouped matmul end *********************/

}
}

#endif // CATLASS_LIBRARY_GEMM_OPERATION_H
