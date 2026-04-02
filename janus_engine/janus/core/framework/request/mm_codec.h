#pragma once
#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <string>

#include "mm_type.h"

namespace janus {

class OpenCVImageDecoder {
 public:
  OpenCVImageDecoder() = default;
  ~OpenCVImageDecoder() = default;

  bool decode(const std::string& raw_data, torch::Tensor& t);
};

class OpenCVImageEncoder {
 public:
  OpenCVImageEncoder() = default;
  ~OpenCVImageEncoder() = default;

  bool encode(const torch::Tensor& t, std::string& raw_data);

 private:
  bool valid(const torch::Tensor& t);
};

class FFmpegVideoDecoder {
 public:
  FFmpegVideoDecoder() = default;
  ~FFmpegVideoDecoder() = default;

  bool decode(const std::string& raw_data,
              torch::Tensor& t,
              VideoMetadata& meta);
};

class FFmpegAudioDecoder {
 public:
  FFmpegAudioDecoder() = default;
  ~FFmpegAudioDecoder() = default;

  bool decode(const std::string& raw_data,
              torch::Tensor& t,
              AudioMetadata& meta);
};
}  // namespace janus
