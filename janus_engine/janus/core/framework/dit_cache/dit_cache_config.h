#pragma once

namespace janus {

enum class PolicyType {
  None,
  FBCache,
  TaylorSeer,
  FBCacheTaylorSeer,
};

struct DiTBaseCacheOptions {
  // the number of warmup steps.
  int warmup_steps = 0;
};

struct FBCacheOptions : public DiTBaseCacheOptions {
  // the residual difference threshold for cache reuse.
  float residual_diff_threshold = 0.09f;
};

struct TaylorSeerOptions : public DiTBaseCacheOptions {
  // the number of derivatives to use in TaylorSeer.
  int n_derivatives = 3;

  // the interval steps to skip for derivative calculation.
  int skip_interval_steps = 3;
};

struct FBCacheTaylorSeerOptions : public DiTBaseCacheOptions {
  // the residual difference threshold for cache reuse.
  float residual_diff_threshold = 0.09f;

  // the number of derivatives to use in TaylorSeer.
  int n_derivatives = 3;
};

struct DiTCacheConfig {
  DiTCacheConfig() = default;

  // the selected cache policy.
  PolicyType selected_policy = PolicyType::None;

  // the configuration for FBCache policy.
  FBCacheOptions fbcache;

  // the configuration for TaylorSeer policy.
  TaylorSeerOptions taylorseer;

  // the configuration for combined FBCache with TaylorSeer policy.
  FBCacheTaylorSeerOptions fbcachetaylorseer;
};

}  // namespace janus
