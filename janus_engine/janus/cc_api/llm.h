#pragma once

#include <atomic>
#include <optional>
#include <string>

#include "macros.h"
#include "types.h"

namespace janus {

// Forward declaration
struct LLMCore;

// A wrapper for loading, initializing, and text generation functions of large
// language models
class JANUS_CAPI_EXPORT LLM {
 public:
  LLM();
  virtual ~LLM();

  LLM(const LLM&) = delete;
  LLM& operator=(const LLM&) = delete;

  LLM(LLM&&) noexcept = delete;
  LLM& operator=(LLM&&) noexcept = delete;

  /**
   * @brief Initialize the model: Load model files and configure runtime
   * environment
   * @param model_path Path to model files
   * @param devices Device configuration (format: "npu:1" for specific NPU,
   * "auto" for auto-selection)
   * @param init_options Advanced initialization options, Provided default
   * configuration
   * @return bool true if initialization succeeds; false if fails
   * @note Must be called before Completions/ChatCompletions, and only needs to
   * be called once
   */
  bool Initialize(const std::string& model_path,
                  const std::string& devices,
                  const JANUS_InitLLMOptions& init_options);

  /**
   * @brief Generate completions for the given prompt
   * @param model_id ID of the loaded model
   * @param prompt Input prompt text
   * @param timeout_ms Timeout in milliseconds
   * @param request_params Request parameters (temperature, max tokens, etc.)
   * @return JANUS_Response Response containing generated text and
   * metadata
   */
  JANUS_Response Completions(const std::string& model_id,
                            const std::string& prompt,
                            uint32_t timeout_ms,
                            const JANUS_RequestParams& request_params);

  /**
   * @brief Generates chat completions based on a sequence of conversation
   * messages
   * @param model_id ID of the loaded model
   * @param messages A list of JANUS_ChatMessage objects representing the
   * conversation history, each message contains a role (user/assistant/system)
   * and content text
   * @param timeout_ms Timeout in milliseconds
   * @param request_params Request parameters (temperature, max tokens, etc.)
   * @return JANUS_Response Response containing generated text and
   * metadata
   */
  JANUS_Response ChatCompletions(const std::string& model_id,
                                const std::vector<JANUS_ChatMessage>& messages,
                                uint32_t timeout_ms,
                                const JANUS_RequestParams& request_params);

 private:
  // Opaque pointer to internal LLM core implementation
  LLMCore* llm_core_ = nullptr;
};
}  // namespace janus
