#ifndef JANUS_LLM_API_H
#define JANUS_LLM_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "types.h"

/**
 * @brief Opaque handle to an LLM inference instance
 *
 * This handle encapsulates all internal state of an LLM inference runtime,
 * including model weights, device context, and generation cache.
 * The handle MUST be created via janus_llm_create() and destroyed via
 * janus_llm_destroy() to prevent memory/device resource leaks.
 */
typedef struct JANUS_LLM_Handler JANUS_LLM_Handler;

/**
 * @brief Create a new LLM inference instance handle
 *
 * Allocates memory and initializes a new LLM handler with default internal
 * state (empty model, uninitialized device context). This is the first function
 * that must be called before using any other LLM APIs.
 *
 * @return Valid JANUS_LLM_Handler* on success; NULL if memory allocation fails
 * @see janus_llm_destroy
 */
JANUS_CAPI_EXPORT JANUS_LLM_Handler* janus_llm_create(void);

/**
 * @brief Destroy an LLM instance handle and release all associated resources
 *
 * Frees all memory allocated for the LLM instance, including:
 * - Model weights (host/device memory)
 * - Runtime context (CUDA/NPU streams, compute graphs)
 * - Generation cache and temporary buffers
 * - Device resources (contexts, queues)
 *
 * This function is idempotent—calling with NULL has no effect.
 *
 * @param handler LLM instance handle (NULL = no operation)
 * @note Mandatory: Must be called to avoid memory/device resource leaks
 * @see janus_llm_create
 */
JANUS_CAPI_EXPORT void janus_llm_destroy(JANUS_LLM_Handler* handler);

/**
 * @brief Initialize JANUS_InitOptions with canonical default values
 *
 * Populates the JANUS_InitOptions struct with industry-standard default values
 *
 * @param init_options Pointer to JANUS_InitOptions to initialize (NULL = no-op)
 * @see janus_llm_initialize, JANUS_INIT_LLM_OPTIONS_DEFAULT
 */
JANUS_CAPI_EXPORT void janus_llm_init_options_default(
    JANUS_InitOptions* init_options);

/**
 * @brief Initialize the LLM model and runtime environment
 *
 * Loads model weights from the specified path, configures target devices,
 * initializes compute contexts, and prepares the inference runtime.
 * Must be called exactly once per handler before using completion/chat APIs.
 *
 * If init_options is NULL, this function automatically uses the default values
 * from JANUS_INIT_LLM_OPTIONS_DEFAULT (via janus_llm_init_options_default()).
 *
 * @param handler Valid LLM instance handle (must not be NULL)
 * @param model_path Null-terminated string of the model directory/file path
 *                   (supports .bin/.pth/.safetensors formats)
 * @param devices Null-terminated string specifying target devices (format:
 *                "npu:0,1" (specific NPUs), "cuda:0" (single GPU), "auto"
 * (automatic selection))
 * @param init_options Advanced initialization options (NULL = use defaults)
 *
 * @return true if initialization succeeds; false on failure (see failure causes
 * below)
 *
 * @failure_causes
 * - Invalid handler (NULL or already destroyed)
 * - Invalid model_path (non-existent, corrupted, or unsupported format)
 * - Invalid devices string (malformed format or unavailable devices)
 * - Model load error (mismatched model architecture or weight corruption)
 * - Device initialization failure (out of memory, driver error)
 *
 * @see janus_llm_init_options_default, JANUS_INIT_LLM_OPTIONS_DEFAULT,
 * janus_llm_create
 */
JANUS_CAPI_EXPORT bool janus_llm_initialize(JANUS_LLM_Handler* handler,
                                          const char* model_path,
                                          const char* devices,
                                          const JANUS_InitOptions* init_options);

/**
 * @brief Initialize JANUS_RequestParams with canonical generation defaults
 *
 * Populates the JANUS_RequestParams struct with safe default generation values
 *
 * @param request_params Pointer to JANUS_RequestParams to initialize (NULL =
 * no-op)
 * @see janus_llm_completions, janus_llm_chat_completions,
 * JANUS_LLM_REQUEST_PARAMS_DEFAULT
 */
JANUS_CAPI_EXPORT void janus_llm_request_params_default(
    JANUS_RequestParams* request_params);

/**
 * @brief Generate text completions for a single prompt
 *
 * Generates continuation text for the input prompt using the initialized LLM
 * model. Returns a dynamically allocated response struct that MUST be freed
 * with janus_llm_free_response() to avoid memory leaks.
 *
 * If request_params is NULL, this function automatically uses the default
 * values from JANUS_LLM_REQUEST_PARAMS_DEFAULT (via
 * janus_llm_request_params_default()).
 *
 * @param handler Valid, initialized LLM instance handle (must not be NULL)
 * @param model_id Null-terminated string of the loaded model ID (must match
 * model_path)
 * @param prompt Null-terminated string of input text to complete (non-empty)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use defaults)
 *
 * @return Pointer to JANUS_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 *
 * @response_status_codes
 * - kSuccess: Valid response generated (check response->choices for results)
 * - kNotInitialized: Handler not initialized with janus_llm_initialize()
 * - kInvalidRequest: Invalid prompt (empty/NULL) or model_id (mismatch)
 * - kTimeout: Generation exceeded timeout_ms (partial results may be available)
 *
 * @warning Mandatory: Call janus_llm_free_response() to release response memory
 * @see janus_llm_request_params_default, JANUS_LLM_REQUEST_PARAMS_DEFAULT,
 * janus_llm_free_response
 */
JANUS_CAPI_EXPORT JANUS_Response* janus_llm_completions(
    JANUS_LLM_Handler* handler,
    const char* model_id,
    const char* prompt,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

/**
 * @brief Generate chat completions from a conversation history
 *
 * Generates model responses for a multi-turn conversation using chat-formatted
 * message history (user/assistant/system roles). Returns a dynamically
 * allocated response struct that MUST be freed with janus_llm_free_response().
 *
 * If request_params is NULL, this function automatically uses the default
 * values from JANUS_LLM_REQUEST_PARAMS_DEFAULT (via
 * janus_llm_request_params_default()).
 *
 * @param handler Valid, initialized LLM instance handle (must not be NULL)
 * @param model_id Null-terminated string of the loaded model ID
 * @param messages Array of JANUS_ChatMessage structs (conversation history)
 * @param messages_count Number of messages in the messages array (must be ≥ 0)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @param request_params Generation parameters (NULL = use defaults)
 *
 * @return Pointer to JANUS_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 *
 * @response_status_codes
 * - kSuccess: Valid chat response generated (check
 * response->choices[0].message)
 * - kNotInitialized: Handler not initialized
 * - kInvalidRequest: Invalid messages (NULL with count>0, empty role/content)
 * - kTimeout: Generation exceeded timeout_ms
 *
 * @warning Mandatory: Call janus_llm_free_response() to release response memory
 * @see janus_llm_request_params_default, JANUS_LLM_REQUEST_PARAMS_DEFAULT,
 * janus_llm_free_response
 */
JANUS_CAPI_EXPORT JANUS_Response* janus_llm_chat_completions(
    JANUS_LLM_Handler* handler,
    const char* model_id,
    const JANUS_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

/**
 * @brief Free all dynamically allocated memory in an JANUS_Response
 *
 * Releases all heap memory used by the response struct
 *
 * After freeing, all fields are reset to safe defaults (NULL/0) to prevent
 * use-after-free.
 *
 * @param resp Pointer to JANUS_Response to free (NULL = no operation)
 *
 * @note Idempotent: Safe to call multiple times on the same response
 * @warning Mandatory: Must be called after using completions/chat completions
 * responses
 * @see janus_llm_completions, janus_llm_chat_completions
 */
JANUS_CAPI_EXPORT void janus_llm_free_response(JANUS_Response* resp);

#ifdef __cplusplus
}
#endif

#endif  // JANUS_LLM_API_H