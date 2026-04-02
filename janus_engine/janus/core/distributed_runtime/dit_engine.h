#pragma once

#include <gflags/gflags.h>

#include <memory>

#include "common/macros.h"
#include "framework/batch/dit_batch.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "runtime/dit_worker.h"

namespace janus {

class DiTEngine {
 public:
  DiTEngine(const runtime::Options& options);

  ~DiTEngine() = default;

  DiTForwardOutput step(std::vector<DiTBatch>& batch);

  const runtime::Options& options() const { return options_; }

  bool init();

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const;

 private:
  bool init_model();
  // options
  runtime::Options options_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<DiTWorker>> workers_;
};

}  // namespace janus