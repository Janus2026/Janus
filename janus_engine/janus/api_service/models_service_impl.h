#pragma once

#include <string>

#include "core/common/macros.h"
#include "models.pb.h"

namespace janus {

class ModelsServiceImpl final {
 public:
  ModelsServiceImpl(const std::vector<std::string>& model_names,
                    const std::vector<std::string>& model_versions);

  bool list_models(const proto::ModelListRequest* request,
                   proto::ModelListResponse* response);
  std::string list_model_versions();

 private:
  DISALLOW_COPY_AND_ASSIGN(ModelsServiceImpl);

  std::vector<std::string> model_names_;
  std::vector<std::string> model_versions_;
  uint32_t created_;
};

}  // namespace janus
