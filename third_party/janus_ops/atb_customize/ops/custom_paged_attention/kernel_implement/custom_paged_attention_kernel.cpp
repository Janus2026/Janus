/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "atb/utils/log.h"
#include <mki/kernel_info.h>
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling/custom_paged_attention_tiling.h"
#include "tiling/custom_paged_attention_tiling_dependency.h"
#include "mixkernels/utils/common.h"

namespace AtbOps {
namespace Custom {
using namespace Mki;
constexpr uint32_t TILINGMIN = 512;
class CustomPagedAttentionKernel : public KernelBase {
public:
    explicit CustomPagedAttentionKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE + TILING_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3, "in tensor0 dims invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dims.size() == 4, "in tensor1 dims invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(2).desc.dims.size() == 4, "in tensor2 dims invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(3).desc.dims.size() == 2, "in tensor3 dims invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention),
            "paged attention: param type invalid", return false);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        if (param.dataShapeType == OpParam::PagedAttention::DataShapeType::BNSD) {
            MKI_CHECK(param.quantType == OpParam::PagedAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                      !param.compressHead && param.scaleType == OpParam::PagedAttention::ScaleType::SCALE_TOR,
                      "BNSD does not support quant,compressHead and logn", return false);
        }
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        ATB_LOG(INFO) << "CustomPagedAttentionKernel::InitImpl called";
        auto status = AtbOps::Custom::PagedAttentionTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention),
            "paged attention: param type invalid", return false);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        auto batch = param.kvSeqLen.size();
        MKI_CHECK(batch > 0 && batch <= ND_BATCH_LIMIT, "batch is invalid", return 0);
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE * (batch - 1) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }
};

constexpr int32_t IN_TENSOR_NUM = 13;
// 14 after add custom_tiling_para_gm
constexpr int32_t QUANT_EYE_CONST_TENSOR_IDX = 14;
class CustomPagedAttentionMaskNdKernel : public CustomPagedAttentionKernel {
public:
    explicit CustomPagedAttentionMaskNdKernel(const std::string &kernelName, const BinHandle *handle)
        : CustomPagedAttentionKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == IN_TENSOR_NUM, "in tensor num invalid", return false);
        MKI_CHECK(CustomPagedAttentionKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        return CustomPagedAttentionKernel::GetTilingSize(launchParam) + Utils::GetConstTensorSize<int8_t>(param.identityM);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        ATB_LOG(INFO) << "CustomPagedAttentionMaskNdKernel::InitImpl called, InTensorCount: " << launchParam.GetInTensorCount();
        Status st = CustomPagedAttentionKernel::InitImpl(launchParam);
        MKI_CHECK_NO_LOG(st.Ok(), return st);
        kernelInfo_.SetConstTensorOffset(CustomPagedAttentionKernel::GetTilingSize(launchParam));
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        kernelInfo_.AddConstTensorData<int8_t>(QUANT_EYE_CONST_TENSOR_IDX, param.identityM);
        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(CustomPagedAttentionMaskNdKernel);
} // namespace Custom
} // namespace AtbOps

