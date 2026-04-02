#include <acl/acl.h>

#include "custom_functions_npu/atb_common.h"

using namespace std;
namespace atb {
void npu_reshape_and_cache(const at::Tensor& key,
                           const at::Tensor& value,
                           at::Tensor& key_cache,
                           at::Tensor& value_cache,
                           const at::Tensor& slot_indices) {
  const c10::OptionalDeviceGuard device_guard(device_of(key));
  OpParamCache<atb::infer::ReshapeAndCacheParam>& reshapeAndCacheParamCache =
      OpParamCache<atb::infer::ReshapeAndCacheParam>::getInstance();
  atb::infer::ReshapeAndCacheParam reshapeparam;
  reshapeparam.compressType =
      atb::infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;

  auto key_cache_format = at_npu::native::get_npu_format(key_cache);
  auto value_cache_format = at_npu::native::get_npu_format(value_cache);
  bool is_key_cache_nz = (key_cache_format == ACL_FORMAT_FRACTAL_NZ);
  bool is_value_cache_nz = (value_cache_format == ACL_FORMAT_FRACTAL_NZ);

  if (is_key_cache_nz && is_value_cache_nz) {
    reshapeparam.kvCacheCfg =
        atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ;
  } else {
    reshapeparam.kvCacheCfg = atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE;
  }

  ParamSetter parametter;
  parametter.Input(key, true)
      .Input(value, true)
      .Input(key_cache)
      .Input(value_cache)
      .Input(slot_indices, true)
      .Output(key_cache)
      .Output(value_cache);
  auto opReshape = reshapeAndCacheParamCache.get_operation(
      reshapeparam, "ReshapeCacheOperation");
  run_atb_cmd(opReshape, parametter, "ReshapeCacheOperation");

  return;
}

}  // namespace atb
