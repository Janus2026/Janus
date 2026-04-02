#include <acl/acl.h>

#include "custom_functions_npu/atb_common.h"

using namespace std;
namespace atb {
void npu_flash_attention(const at::Tensor& query,
                         const at::Tensor& key,
                         const at::Tensor& value,
                         const at::Tensor& mask,
                         const at::Tensor& seq_len,
                         const double scale_value,
                         const int64_t num_heads,
                         const int64_t num_kv_heads,
                         at::Tensor& out) {
  const c10::OptionalDeviceGuard device_guard(device_of(query));
  OpParamCache<atb::infer::SelfAttentionParam>& selfAttentionParamCache =
      OpParamCache<atb::infer::SelfAttentionParam>::getInstance();
  atb::infer::SelfAttentionParam selfattentionparam;

  selfattentionparam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
  selfattentionparam.kernelType =
      atb::infer::SelfAttentionParam::KERNELTYPE_DEFAULT;
  selfattentionparam.clampType =
      atb::infer::SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
  selfattentionparam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
  selfattentionparam.kvcacheCfg =
      atb::infer::SelfAttentionParam::K_CACHE_V_CACHE;
  selfattentionparam.scaleType = atb::infer::SelfAttentionParam::SCALE_TYPE_TOR;
  selfattentionparam.quantType =
      atb::infer::SelfAttentionParam::TYPE_QUANT_UNDEFINED;
  selfattentionparam.cacheType =
      atb::infer::SelfAttentionParam::CACHE_TYPE_NORM;
  selfattentionparam.outDataType = ACL_DT_UNDEFINED;
  selfattentionparam.headNum = num_heads;
  selfattentionparam.kvHeadNum = num_kv_heads;
  selfattentionparam.qScale = 1;
  selfattentionparam.qkScale = scale_value;
  selfattentionparam.batchRunStatusEnable = false;
  selfattentionparam.isTriuMask = 0;
  selfattentionparam.clampMin = 0;
  selfattentionparam.clampMax = 0;
  selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
  selfattentionparam.mlaVHeadSize = 0;
  selfattentionparam.windowSize = 0;

  ParamSetter parametter;
  parametter.Input(query, true)
      .Input(key, true)
      .Input(value, true)
      .Input(mask)
      .Input(seq_len, true)
      .Output(out);

  auto opSelfattention = selfAttentionParamCache.get_operation(
      selfattentionparam, "SelfAttentionOperation");
  run_atb_cmd(opSelfattention, parametter, "SelfAttentionOperation");

  return;
}

}  // namespace atb