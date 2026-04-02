#pragma once

#include <functional>
#include <memory>

#include "common/macros.h"
#include "parser/reasoning_detector.h"

namespace janus {

using DetectorFactory =
    std::function<std::unique_ptr<ReasoningDetector>(bool, bool)>;
class DetectorRegistry {
 public:
  static DetectorRegistry& get_instance() {
    static DetectorRegistry instance;
    return instance;
  }

  std::unique_ptr<ReasoningDetector> get_detector(const std::string& model_type,
                                                  bool stream_reasoning,
                                                  bool force_reasoning);

  bool has_detector(const std::string& parser_name) const;

  std::string get_supported_parsers() const;

  // Get the reasoning parser name for auto mode based on model_type
  // Returns empty string if not found
  std::string get_parser_name_by_model_type(
      const std::string& model_type) const;

 private:
  DetectorRegistry() = default;
  ~DetectorRegistry() = default;

  DISALLOW_COPY_AND_ASSIGN(DetectorRegistry);
};

}  // namespace janus