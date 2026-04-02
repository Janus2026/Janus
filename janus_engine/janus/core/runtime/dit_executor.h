#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "core/framework/dit_model_loader.h"
#include "forward_params.h"
#include "framework/batch/dit_batch.h"
#include "framework/model/dit_model.h"
#include "framework/model/model_input_params.h"
#include "framework/request/dit_request_state.h"
#include "runtime/options.h"

namespace janus {

class DiTExecutor {
 public:
  DiTExecutor(DiTModel* model, const runtime::Options& options);

  ~DiTExecutor() = default;

  DiTForwardInput prepare_inputs(DiTBatch& batch);

  DiTForwardOutput forward(const DiTForwardInput& input);

 private:
  // not own
  DiTModel* model_;
  runtime::Options options_;
};

}  // namespace janus
