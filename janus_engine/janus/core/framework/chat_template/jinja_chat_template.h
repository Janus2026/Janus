#pragma once

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "framework/tokenizer/tokenizer_args.h"

namespace janus {

// A chat template implementation that uses jinja2 as the template engine.
class JinjaChatTemplate {
 public:
  JinjaChatTemplate(const TokenizerArgs& args);

  std::optional<std::string> apply(const ChatMessages& messages) const;

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<janus::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const;

 protected:
  // apply the template to the values in the json object
  std::optional<std::string> apply(nlohmann::ordered_json& messages) const;

  std::optional<std::string> apply(
      nlohmann::ordered_json& messages,
      const nlohmann::ordered_json& tools,
      const nlohmann::ordered_json& chat_template_kwargs) const;

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const nlohmann::ordered_json& chat_template_kwargs) const;

  nlohmann::ordered_json get_mm_content(const MMContentVec& vec) const;

 private:
  TokenizerArgs args_;
  std::unique_ptr<minja::chat_template> template_;
};

}  // namespace janus
