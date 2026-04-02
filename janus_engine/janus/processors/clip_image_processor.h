#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/util/tensor_helper.h"
#include "image_processor.h"

namespace janus {

class CLIPImageProcessor : public ImageProcessor {
 public:
  CLIPImageProcessor(const ModelArgs& args);
  ~CLIPImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;
  torch::Tensor process_images(const torch::Tensor& images);

 private:
  std::vector<int64_t> get_resize_output_image_size(const torch::Tensor& image,
                                                    int shortest_edge);

 private:
  bool do_resize_;
  bool do_center_crop_;
  bool do_rescale_;
  bool do_normalize_;
  int shortest_edge_;
  int resample_;
  double rescale_factor_;
  std::pair<int, int> crop_size_;
  std::vector<double> image_mean_;
  std::vector<double> image_std_;
};

}  // namespace janus
