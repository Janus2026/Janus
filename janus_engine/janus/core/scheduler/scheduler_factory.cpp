#include "scheduler/scheduler_factory.h"

#include "core/common/global_flags.h"
#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"
#include "scheduler/mix_scheduler.h"
#include "scheduler/pd_ooc_scheduler.h"
#include "scheduler/prefill_only_scheduler.h"
#include "scheduler/zero_eviction_scheduler.h"

namespace janus {

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  if (FLAGS_use_mix_scheduler) {
    CHECK(options.enable_chunked_prefill())
        << "mix scheduler requires enabling chunked prefill";
    return std::make_unique<MixScheduler>(engine, options);
  }

  if (options.enable_disagg_pd()) {
    if (options.enable_pd_ooc()) {
      return std::make_unique<PDOOCScheduler>(engine, options);
    } else {
      return std::make_unique<DisaggPDScheduler>(engine, options);
    }
  }

  if (options.enable_chunked_prefill()) {
    if (FLAGS_enable_prefill_sp || options.num_speculative_tokens() > 0) {
      return std::make_unique<PrefillOnlyScheduler>(engine, options);
    } else {
      return std::make_unique<ChunkedPrefillScheduler>(engine, options);
    }
  }

  if (FLAGS_use_zero_evict) {
    return std::make_unique<ZeroEvictionScheduler>(engine, options);
  }

  return std::make_unique<ContinuousScheduler>(engine, options);
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options) {
  return std::make_unique<DiTDynamicBatchScheduler>(engine, options);
}

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  return std::make_unique<FixedStepsScheduler>(engine, options);
}

}  // namespace janus
