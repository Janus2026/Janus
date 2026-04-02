#pragma once

#include <torch/torch.h>

#include <limits>
#include <vector>

namespace janus {

class Sequence;
struct ModelArgs;

class MPositionHelper {
 public:
  MPositionHelper(Sequence& seq, const ModelArgs& args)
      : seq_(seq), args_(args) {}

  torch::Tensor get_positions();

 private:
  std::tuple<torch::Tensor, int> get_positions_p(
      torch::Tensor image_grid_thw,
      torch::Tensor video_grid_thw,
      torch::Tensor second_per_grid_ts);
  std::tuple<torch::Tensor, int> get_positions_glm(
      torch::Tensor image_grid_thw,
      torch::Tensor video_grid_thw);

  torch::Tensor get_positions_d();

 private:
  Sequence& seq_;
  const ModelArgs& args_;
};

}  // namespace janus
