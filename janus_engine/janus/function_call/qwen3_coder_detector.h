#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base_format_detector.h"

namespace janus {
namespace function_call {

class Qwen3CoderDetector : public BaseFormatDetector {
 public:
  Qwen3CoderDetector();

  virtual ~Qwen3CoderDetector() = default;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::string tool_call_start_token_;
  std::string tool_call_end_token_;
  std::string tool_call_prefix_;
  std::string function_end_token_;
  std::string parameter_prefix_;
  std::string parameter_end_token_;

  size_t parsed_pos_;
  int32_t current_tool_param_count_;
  bool json_started_;
  bool is_inside_tool_call_;
  std::optional<std::string> current_func_name_;

  static bool starts_with(std::string_view text, std::string_view prefix);
  static std::string to_lower_copy(const std::string& input);
  static std::string trim_ascii_whitespace(std::string_view input);

  nlohmann::json get_arguments_config(const std::string& func_name,
                                      const std::vector<JsonTool>& tools) const;

  nlohmann::json convert_param_value(const std::string& param_value,
                                     const std::string& param_name,
                                     const nlohmann::json& param_config,
                                     const std::string& func_name) const;

  void parse_parameters(const std::string& params_text,
                        const std::string& func_name,
                        const std::vector<JsonTool>& tools,
                        nlohmann::json* parsed_params) const;

  void parse_tool_call_content(const std::string& tool_content,
                               const std::vector<JsonTool>& tools,
                               int32_t* tool_idx,
                               std::vector<ToolCallItem>* calls) const;
};

}  // namespace function_call
}  // namespace janus
