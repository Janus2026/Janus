#pragma once

#include <torch/torch.h>

#include <limits>
#include <vector>

#include "framework/request/dit_request.h"
#include "runtime/dit_forward_params.h"

namespace janus {

struct DiTBatch {
 public:
  DiTBatch() = default;
  void add(const std::shared_ptr<DiTRequest>& request) {
    request_vec_.emplace_back(request);
  }
  size_t size() const { return request_vec_.size(); }
  bool empty() const { return request_vec_.empty(); }

  // prepare forward input
  DiTForwardInput prepare_forward_input();

  void process_forward_output(const DiTForwardOutput& output);

 private:
  std::vector<std::shared_ptr<DiTRequest>> request_vec_;
};

}  // namespace janus
