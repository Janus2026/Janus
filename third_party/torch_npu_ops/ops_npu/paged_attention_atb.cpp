#include <acl/acl.h>
#include <customize/customize_op_params.h>

#include "custom_functions_npu/atb_common.h"

namespace atb {

// Helper function to create PagedAttentionParam with common settings
static atb::infer::PagedAttentionParam createPagedAttentionParam(
    int64_t num_heads,
    int64_t num_kv_heads,
    double scale_value) {
  atb::infer::PagedAttentionParam pagedparam;
  pagedparam.headNum = num_heads;
  pagedparam.qkScale = scale_value;
  pagedparam.kvHeadNum = num_kv_heads;
  pagedparam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
  pagedparam.batchRunStatusEnable = false;
  pagedparam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_UNDEFINED;
  pagedparam.outDataType = ACL_DT_UNDEFINED;
  pagedparam.hasQuantOffset = false;
  pagedparam.compressType =
      atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
  pagedparam.calcType = atb::infer::PagedAttentionParam::CALC_TYPE_UNDEFINED;
  pagedparam.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_TOR;
  pagedparam.inputLayout = atb::infer::TYPE_BSND;
  pagedparam.mlaVHeadSize = 0;
  return pagedparam;
}

void npu_paged_attention(const at::Tensor& query,
                         const at::Tensor& key_cache,
                         const at::Tensor& value_cache,
                         int64_t num_kv_heads,
                         int64_t num_heads,
                         double scale_value,
                         const at::Tensor& block_table,
                         const at::Tensor& context_lens,
                         at::Tensor& out) {
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  OpParamCache<atb::infer::PagedAttentionParam>& pagedAttentionParamCache =
      OpParamCache<atb::infer::PagedAttentionParam>::getInstance();
  atb::infer::PagedAttentionParam pagedparam =
      createPagedAttentionParam(num_heads, num_kv_heads, scale_value);

  ParamSetter paramsetter;
  paramsetter.Input(query, true)
      .Input(key_cache)
      .Input(value_cache)
      .Input(block_table, true)
      .Input(context_lens, true)
      .Output(out);
  auto opPaged = pagedAttentionParamCache.get_operation(
      pagedparam, "PagedAttentionOperation");
  run_atb_cmd(opPaged, paramsetter, "PagedAttentionOperation");

  return;
}

void npu_custom_paged_attention(const at::Tensor& query,
                                const at::Tensor& key_cache,
                                const at::Tensor& value_cache,
                                int64_t num_kv_heads,
                                int64_t num_heads,
                                double scale_value,
                                const at::Tensor& block_table,
                                const at::Tensor& context_lens,
                                const at::Tensor& tiling_data,
                                at::Tensor& out) {
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  OpParamCache<atb::customize::CustomPagedAttentionParam>&
      customPagedAttentionParamCache = OpParamCache<
          atb::customize::CustomPagedAttentionParam>::getInstance();

  // Create base PagedAttentionParam and then convert to
  // CustomPagedAttentionParam
  atb::infer::PagedAttentionParam pagedparam =
      createPagedAttentionParam(num_heads, num_kv_heads, scale_value);
  atb::customize::CustomPagedAttentionParam customPagedParam(pagedparam);

  ParamSetter paramsetter;
  paramsetter.Input(query, true)
      .Input(key_cache)
      .Input(value_cache)
      .Input(block_table, true)
      .Input(context_lens, true)
      .Input(tiling_data, true)  // tiling_data for ACL graph execution
      .Output(out);
  auto opCustomPaged = customPagedAttentionParamCache.get_operation(
      customPagedParam, "CustomPagedAttentionOperation");
  run_atb_cmd(opCustomPaged, paramsetter, "CustomPagedAttentionOperation");

  return;
}

}  // namespace atb