#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/request/mm_data.h"
#include "core/framework/request/mm_input.h"

namespace janus {

class ImageProcessor {
 public:
  virtual ~ImageProcessor() = default;

  virtual bool process(const MMInput& mm_inputs, MMData& mm_datas) = 0;
  virtual torch::Tensor resize(const torch::Tensor& image,
                               const std::vector<int64_t>& size,
                               int resample,
                               bool antialias = true);
  virtual torch::Tensor centerCrop(const torch::Tensor& image,
                                   const std::pair<int, int>& cropSize);
  virtual torch::Tensor rescale(const torch::Tensor& image, double scale);
  virtual torch::Tensor normalize(const torch::Tensor& image,
                                  const std::vector<double>& mean,
                                  const std::vector<double>& std);
};

}  // namespace janus
