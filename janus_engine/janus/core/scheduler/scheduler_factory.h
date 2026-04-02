#pragma once

#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"

namespace janus {

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options);

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

}  // namespace janus
