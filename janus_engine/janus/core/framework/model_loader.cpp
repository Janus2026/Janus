#include "model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <vector>

#include "hf_model_loader.h"

namespace janus {

std::unique_ptr<ModelLoader> ModelLoader::create(
    const std::string& model_weights_path) {
  ModelType model_type;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".safetensors" ||
        entry.path().extension() == ".bin") {
      model_type = ModelType::HF_MODEL_TYPE;
      break;
    }
  }

  if (model_type == ModelType::HF_MODEL_TYPE) {
    return std::make_unique<HFModelLoader>(model_weights_path);
  } else {
    LOG(FATAL) << "Only support HF model type currently.";
  }

  return nullptr;
}

}  // namespace janus
