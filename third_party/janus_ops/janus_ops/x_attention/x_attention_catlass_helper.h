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

#ifndef X_ATTN_CATLASS_HELPER_H
#define X_ATTN_CATLASS_HELPER_H
#include "x_attention_catlass_kernel.h"

template <typename INPUT_T, bool isPAEnabled>
CATLASS_DEVICE void CallSharedInferKernelShort(const XAttnKernelParams& params, XAttentionTilingData* tilingData) {
    using ArchTag = Arch::AtlasA2;
    using ElementQ = INPUT_T;
    using LayoutQ = layout::RowMajor;
    using ElementK = INPUT_T;
    using LayoutK = layout::ColumnMajor;
    using ElementV = INPUT_T;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = INPUT_T;
    using LayoutP = layout::RowMajor;
    using ElementO = INPUT_T;
    using LayoutO = layout::RowMajor;
    using ElementMask = INPUT_T;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = L1TileShape;
    // GEMM Block, implement Q @ K^T of Flash Attention Infer
    // using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKSplitRow<isPAEnabled, false>;
    
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Shared Epilogue Block, update rowsum rowmax and copyOut on lastStackTile 
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxCopySumMax;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    // GEMM Block, implement P @ V of Flash Attention Infer
    // using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVSplitRow<isPAEnabled, false>;

    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Shared Epilogue RescaleO，do not div rowSum or cast on lastStackTile
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOWithoutDivSum;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType>;
    
    using SharedFAInferKernel = SharedFAInferKernelShort<
                BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, isPAEnabled>;
    
    SharedFAInferKernel sharedInferKernel(tilingData);
    sharedInferKernel(params);
}

template <typename INPUT_T, bool isPAEnabled>
CATLASS_DEVICE void CallUnsharedInferKernel(const XAttnKernelParams& params, XAttentionTilingData* tilingData) {
using ArchTag = Arch::AtlasA2;
using ElementQ = INPUT_T;
using LayoutQ = layout::RowMajor;
using ElementK = INPUT_T;
using LayoutK = layout::ColumnMajor;
using ElementV = INPUT_T;
using LayoutV = layout::RowMajor;
using ElementS = float;
using LayoutS = layout::RowMajor;
using ElementP = INPUT_T;
using LayoutP = layout::RowMajor;
using ElementO = INPUT_T;
using LayoutO = layout::RowMajor;
using ElementMask = INPUT_T;
using LayoutMask = layout::RowMajor;
using ElementOTmp = float;
using LayoutOTmp = layout::RowMajor;
using QType = Gemm::GemmType<ElementQ, LayoutQ>;
using KType = Gemm::GemmType<ElementK, LayoutK>;
using SType = Gemm::GemmType<ElementS, LayoutS>;
using PType = Gemm::GemmType<ElementP, LayoutP>;
using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
using VType = Gemm::GemmType<ElementV, LayoutV>;
using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;

using QKL1TileShape = GemmShape<128, 256, 128>;
using QKL0TileShape = QKL1TileShape;
using MmadDispatchPolicyQK = Gemm::MmadAtlasA2UnsharedFAQK;
using BlockMmadQK = Gemm::Block::BlockMmad<MmadDispatchPolicyQK, QKL1TileShape, QKL0TileShape, QType, KType, SType>;

using DispatchPolicyFAUnsharedSoftmax = Epilogue::EpilogueAtlasA2FAUnsharedSoftmax;
using PType = Gemm::GemmType<ElementP, LayoutP>;
using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
using EpilogueFAUnsharedSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyFAUnsharedSoftmax, PType, SType, maskType>;

using PVL1TileShape = GemmShape<128, 128, 256>;
using PVL0TileShape = PVL1TileShape;
using MmadDispatchPolicyPV = Gemm::MmadAtlasA2UnsharedFAPV;
using BlockMmadPV = Gemm::Block::BlockMmad<MmadDispatchPolicyPV, PVL1TileShape, PVL0TileShape, PType, VType, OTmpType>;
using UnsharedFAInferKernel = UnsharedFAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueFAUnsharedSoftmax, isPAEnabled>;

UnsharedFAInferKernel unsharedInferKernel(tilingData);
unsharedInferKernel(params);
}


template <typename INPUT_T>
CATLASS_DEVICE void CallCombineScale(const XAttnKernelParams& params, XAttentionTilingData* tilingData) {
using ArchTag = Arch::AtlasA2;
using ElementInput = float;
using LayoutInput = layout::RowMajor;
// MatrixShape<rowNum, columnNum>    
using ElementOutput = INPUT_T;
using LayoutOutput = layout::RowMajor;
using InputType = Gemm::GemmType<ElementInput, LayoutInput>;
using OutputType = Gemm::GemmType<ElementOutput, LayoutOutput>;
using DispatchPolicyCombineScale = Epilogue::EpilogueAtlasA2CombineScale;
using BlockEpilogueCombineScale = Epilogue::Block::BlockEpilogue<DispatchPolicyCombineScale, OutputType, InputType>;
using CombineScaleKernel = CombineScaleKernel<BlockEpilogueCombineScale>;
CombineScaleKernel combineScaleKernel(tilingData);
combineScaleKernel(params);
}

 
#endif