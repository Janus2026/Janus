#include <acl/acl.h>

#include "custom_functions_npu/atb_common.h"

namespace atb {
using RopeParam = atb::infer::RopeParam;

static at::Tensor sequenceLength;
static int64_t previousTokenCount = -1;

static at::Tensor cosCacheNeox;
static at::Tensor sinCacheNeox;
static at::Tensor cosCache;
static at::Tensor sinCache;

void InitializeCosSinCache(const at::Tensor& cos_sin_cache) {
  auto cosSinChunks = cos_sin_cache.chunk(2, -1);

  cosCache = cosSinChunks[0].repeat_interleave(2, 1);
  sinCache = cosSinChunks[1].repeat_interleave(2, 1);
  cosCacheNeox = cosSinChunks[0].repeat({1, 2});
  sinCacheNeox = cosSinChunks[1].repeat({1, 2});
}

void npu_rotary_embedding(const at::Tensor& positions,
                          at::Tensor& query,
                          at::Tensor& key,
                          int64_t head_size,
                          const at::Tensor& cos_sin_cache,
                          bool is_neox_style) {
  const c10::OptionalDeviceGuard device_guard(device_of(positions));
  if (!cosCache.defined() || !sinCache.defined()) {
    InitializeCosSinCache(cos_sin_cache);
  }

  at::Tensor flatPositions = positions.flatten();
  int32_t currentTokenCount = flatPositions.size(0);

  at::Tensor cos = is_neox_style ? cosCacheNeox.index_select(0, flatPositions)
                                 : cosCache.index_select(0, flatPositions);
  at::Tensor sin = is_neox_style ? sinCacheNeox.index_select(0, flatPositions)
                                 : sinCache.index_select(0, flatPositions);

  if (!sequenceLength.defined() || previousTokenCount != currentTokenCount) {
    previousTokenCount = currentTokenCount;
    sequenceLength =
        at::full({1},
                 currentTokenCount,
                 at::TensorOptions().device(query.device()).dtype(at::kInt));
  }

  RopeParam ropeparam;
  ropeparam.rotaryCoeff = is_neox_style ? 2 : head_size;

  ParamSetter parametter;
  parametter.Input(query, true)
      .Input(key, true)
      .Input(cos, true)
      .Input(sin, true)
      .Input(sequenceLength, true)
      .Output(query)
      .Output(key);

  OpParamCache<RopeParam>& ropeParamCache =
      OpParamCache<RopeParam>::getInstance();
  auto opRope = ropeParamCache.get_operation(ropeparam, "RopeOperation");
  run_atb_cmd(opRope, parametter, "RopeOperation");
}
}  // namespace atb
