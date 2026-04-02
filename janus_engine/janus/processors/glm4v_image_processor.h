#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>

#include "image_processor.h"

namespace janus {

class Glm4VImageProcessor : public ImageProcessor {
 public:
  Glm4VImageProcessor(const ModelArgs&);
  ~Glm4VImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_image(torch::Tensor image,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);
  bool process_videos(std::vector<torch::Tensor> videos,
                      std::vector<VideoMetadata> video_meta_list,
                      MMData& mm_datas);
  bool process_video(torch::Tensor video,
                     VideoMetadata& metadata,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);
  torch::Tensor sample_frames(const VideoMetadata& metadata,
                              int temporal_patch_size);

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;

  bool do_rescale_ = true;
  bool do_resize_ = true;

  std::vector<double> image_mean_;
  std::vector<double> image_std_;

  int max_pixels_ = 12845056;
  int min_pixels_ = 3136;

  int merge_size_ = 2;
  int patch_size_ = 14;

  std::vector<double> video_mean_;
  std::vector<double> video_std_;

  int video_max_pixels_ = 47040000;
  int video_min_pixels_ = 12544;

  int video_merge_size_ = 2;
  int video_patch_size_ = 14;

  int resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  std::unordered_map<std::string, int> size_;
  int temporal_patch_size_ = 2;
  int video_temporal_patch_size_ = 2;

  bool do_sample_frame_ = true;

  int min_frames_ = 4;
  int max_frames_ = 768;
};

}  // namespace janus
