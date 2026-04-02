#include "models_service_impl.h"

#include <nlohmann/json.hpp>
#include <string>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "models.pb.h"

namespace janus {

ModelsServiceImpl::ModelsServiceImpl(
    const std::vector<std::string>& model_names,
    const std::vector<std::string>& model_versions)
    : model_names_(model_names),
      model_versions_(model_versions),
      created_(absl::ToUnixSeconds(absl::Now())) {}

bool ModelsServiceImpl::list_models(const proto::ModelListRequest* request,
                                    proto::ModelListResponse* response) {
  for (const auto& model_id : model_names_) {
    auto* model_card = response->add_data();
    model_card->set_id(model_id);
    model_card->set_created(created_);
    model_card->set_object("model");
    model_card->set_owned_by("janus");
  }
  return true;
}

std::string ModelsServiceImpl::list_model_versions() {
  nlohmann::json model_states_array = nlohmann::json::array();

  for (size_t i = 0; i < model_names_.size(); ++i) {
    nlohmann::json model_state;
    model_state["name"] = model_names_[i];
    model_state["version"] = model_versions_[i];
    model_state["state"] = "READY";
    model_state["reason"] = "normal";
    model_states_array.push_back(model_state);
  }

  return model_states_array.dump();
}

}  // namespace janus
