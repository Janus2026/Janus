/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_preprocess_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> qDtype = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
static const std::vector<ge::DataType> int8_Dtype = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8};
static const std::vector<ge::DataType> int8_fp16_Dtype = {ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                          ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                          ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                                                          ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
static const std::vector<ge::DataType> int64_fp32_Dtype = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                           ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                           ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                           ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::DataType> int32_Dtype = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
static const std::vector<ge::DataType> fp16_int8_Dtype = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8,
                                                          ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8,
                                                          ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8,
                                                          ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8};
static const std::vector<ge::Format> qFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::Format> nzFormat = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ};
static const std::vector<ge::Format> nd_nz_Format = {ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ}; 
static const std::vector<ge::Format> nd_nd_nz_Format = {ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ};
constexpr int HUNDRED_TWENTY_EIGHT = 128;
constexpr int SIXTY_FOUR = 64;
constexpr int TWO = 2;

class MlaPreprocess : public OpDef {
public:
    explicit MlaPreprocess(const char *name) : OpDef(name)
    {
        this->Input("input").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("gamma_0").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("beta_0").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("quant_scale_0").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("quant_offset_0").ParamType(REQUIRED).DataType(int8_Dtype).Format(qFormat).AutoContiguous();
        this->Input("wdqkv").ParamType(REQUIRED).DataType(int8_fp16_Dtype).Format(nzFormat).AutoContiguous();
        this->Input("descale_0").ParamType(REQUIRED).DataType(int64_fp32_Dtype).Format(qFormat).AutoContiguous();
        this->Input("bias_0").ParamType(REQUIRED).DataType(int32_Dtype).Format(qFormat).AutoContiguous();
        this->Input("gamma_1").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("beta_1").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("quant_scale_1").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("quant_offset_1").ParamType(REQUIRED).DataType(int8_Dtype).Format(qFormat).AutoContiguous();
        this->Input("wuq").ParamType(REQUIRED).DataType(int8_fp16_Dtype).Format(nzFormat).AutoContiguous();
        this->Input("descale_1").ParamType(REQUIRED).DataType(int64_fp32_Dtype).Format(qFormat).AutoContiguous();
        this->Input("bias_1").ParamType(REQUIRED).DataType(int32_Dtype).Format(qFormat).AutoContiguous();
        this->Input("gamma_2").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("cos").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("sin").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("wuk").ParamType(REQUIRED).DataType(qDtype).Format(nd_nd_nz_Format).AutoContiguous();
        this->Input("kv_cache").ParamType(REQUIRED).DataType(fp16_int8_Dtype).Format(nd_nz_Format).AutoContiguous();
        this->Input("kv_cache_rope").ParamType(REQUIRED).DataType(qDtype).Format(nd_nz_Format).AutoContiguous();
        this->Input("slot_mapping").ParamType(REQUIRED).DataType(int32_Dtype).Format(qFormat).AutoContiguous();
        this->Input("ctkv_scale").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Input("q_nope_scale").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).AutoContiguous();
        this->Output("q_out").ParamType(REQUIRED).DataType(fp16_int8_Dtype).Format(qFormat);
        this->Output("kv_cache_out").ParamType(REQUIRED).DataType(fp16_int8_Dtype).Format(nd_nz_Format);
        this->Output("q_rope_out").ParamType(REQUIRED).DataType(qDtype).Format(qFormat);
        this->Output("kr_cache_out").ParamType(REQUIRED).DataType(qDtype).Format(nd_nz_Format);
        this->Attr("wdq_dim").AttrType(OPTIONAL).Int(HUNDRED_TWENTY_EIGHT);
        this->Attr("q_rope_dim").AttrType(OPTIONAL).Int(SIXTY_FOUR);
        this->Attr("k_rope_dim").AttrType(OPTIONAL).Int(SIXTY_FOUR);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-05f);
        this->Attr("q_rotary_coeff").AttrType(OPTIONAL).Int(TWO);
        this->Attr("k_rotary_coeff").AttrType(OPTIONAL).Int(TWO);
        this->Attr("transepose_wdq").AttrType(OPTIONAL).Bool(true);
        this->Attr("transepose_wuq").AttrType(OPTIONAL).Bool(true);
        this->Attr("transepose_wuk").AttrType(OPTIONAL).Bool(true);
        this->Attr("cache_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("do_rms_norm").AttrType(OPTIONAL).Bool(true);
        this->Attr("wdkv_split_count").AttrType(OPTIONAL).Int(1);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mla_preprocess");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(MlaPreprocess);
} // namespace ops