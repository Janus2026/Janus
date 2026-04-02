#include "dit_executor.h"

#include <glog/logging.h>

#include "common/metrics.h"

namespace janus {

DiTExecutor::DiTExecutor(DiTModel* model, const runtime::Options& options)
    : model_(model), options_(options) {}

DiTForwardInput DiTExecutor::prepare_inputs(DiTBatch& batch) {
  return batch.prepare_forward_input();
}

DiTForwardOutput DiTExecutor::forward(const DiTForwardInput& input) {
  return model_->forward(input);
}

}  // namespace janus
