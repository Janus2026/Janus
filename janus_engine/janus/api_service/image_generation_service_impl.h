#pragma once
#include <absl/container/flat_hash_set.h>

#include "api_service/api_service_impl.h"
#include "api_service/non_stream_call.h"
#include "image_generation.pb.h"

namespace janus {

using ImageGenerationCall = NonStreamCall<proto::ImageGenerationRequest,
                                          proto::ImageGenerationResponse>;
class DiTMaster;
// a class to handle image generation requests
class ImageGenerationServiceImpl : public APIServiceImpl<ImageGenerationCall> {
 public:
  ImageGenerationServiceImpl(DiTMaster* master,
                             const std::vector<std::string>& models);

  // brpc call_data needs to use shared_ptr
  void process_async_impl(std::shared_ptr<ImageGenerationCall> call);

 private:
  DISALLOW_COPY_AND_ASSIGN(ImageGenerationServiceImpl);
  DiTMaster* master_ = nullptr;
};

}  // namespace janus
