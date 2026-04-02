#pragma once

#include "api_service/api_service_impl.h"

namespace janus {

template <typename T>
class ServiceImplFactory {
 public:
  static std::unique_ptr<T> create_service_impl(
      LLMMaster* master,
      const std::vector<std::string>& model_names) {
    auto service_impl = std::make_unique<T>(master, model_names);
    if (!service_impl) {
      LOG(ERROR) << "handler is nullptr";
    }
    return service_impl;
  }

  static std::unique_ptr<T> create_service_impl(
      const std::vector<std::string>& model_names,
      const std::vector<std::string>& model_versions) {
    if (model_names.size() != model_versions.size()) {
      LOG(ERROR)
          << "Models and model_versions size mismatch: model_names.size()="
          << model_names.size()
          << ", model_versions.size()=" << model_versions.size();
    }
    auto service_impl = std::make_unique<T>(model_names, model_versions);
    if (!service_impl) {
      LOG(ERROR) << "handler is nullptr";
    }
    return service_impl;
  }
};

}  // namespace janus
