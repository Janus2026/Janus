#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>
#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "c_api/default.h"
#include "c_api/types.h"
#include "core/common/instance_name.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"

/**
 * @brief Opaque handle for LLM inference instance
 */
struct JANUS_LLM_Handler {
  /** Flag indicating if LLM instance is initialized and ready for inference */
  bool initialized{false};

  /** List of loaded model IDs (for model existence validation) */
  std::vector<std::string> model_ids;

  /** Core controller for LLM runtime management */
  std::unique_ptr<janus::LLMMaster> master;

  /** Thread pool for asynchronous inference task scheduling */
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor;
};

/**
 * @brief Opaque handle for REC (Recommendation) inference instance
 */
struct JANUS_REC_Handler {
  /** Flag indicating if REC instance is initialized and ready for inference */
  bool initialized{false};

  /** List of loaded recommendation model IDs */
  std::vector<std::string> model_ids;

  /** Core controller for REC runtime management */
  std::unique_ptr<janus::RecMaster> master;

  /** Thread pool for asynchronous recommendation task scheduling */
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor;
};

namespace janus {
namespace helper {

enum class BackendType { LLM = 0, VLM = 1, REC = 2 };

enum class InferenceType {
  LLM_COMPLETIONS = 0,
  LLM_CHAT_COMPLETIONS = 1,
  REC_COMPLETIONS = 2,
  REC_CHAT_COMPLETIONS = 3,
  REC_TOKENID_COMPLETIONS = 4,
};

#define JANUS_SET_META_STRING_FIELD(DST, SRC_STR)                              \
  do {                                                                        \
    static_assert(sizeof(DST) > 1, "Destination buffer is too small");        \
    strncpy(                                                                  \
        (char*)(DST), (SRC_STR).c_str(), JANUS_META_STRING_FIELD_MAX_LEN - 1); \
    (DST)[JANUS_META_STRING_FIELD_MAX_LEN - 1] = '\0';                         \
  } while (0)

/**
 * @brief Thread-safe glog initialization for Janus framework
 * @note This API is idempotent (multiple calls have same effect as single call)
 * @note Thread-safe: protected by pthread mutex to prevent race condition
 * @param log_dir Directory to store log files (empty = current directory)
 */
void init_log(const std::string& log_dir);

/**
 * @brief Safely shutdown glog and release resources
 * @note Call this function before program exit (optional but recommended)
 */
void shutdown_log();

/**
 * @brief Set init options, merge default options
 */
void set_init_options(BackendType backend_type,
                      const JANUS_InitOptions* init_options,
                      JANUS_InitOptions* janus_init_options);

/**
 * @brief Transfer C API request params to Janus internal request params
 */
void transfer_request_params(InferenceType inference_type,
                             const JANUS_RequestParams* request_params,
                             janus::RequestParams* janus_request_params);

/**
 * @brief Build error response for failed inference requests
 */
JANUS_Response* build_error_response(const std::string& request_id,
                                    JANUS_StatusCode status_code,
                                    const std::string& error_info);

/**
 * @brief Build success response for completed inference requests
 */
JANUS_Response* build_success_response(const InferenceType& inference_type,
                                      const janus::RequestOutput& output,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model);

/**
 * @brief Generic inference request handler (template function)
 */
template <typename HandlerType, typename InputType>
JANUS_Response* handle_inference_request(
    HandlerType* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const InputType& input,
    void* extra,
    uint32_t timeout_ms,
    const JANUS_RequestParams* request_params);

/**
 * @brief Safely free all memory allocated in JANUS_Response
 */
void janus_free_response(JANUS_Response* resp);

/**
 * @brief Generate unique request ID for tracing
 */
std::string generate_request_id();

torch::ScalarType janus_dtype_to_torch_scalar_type(JANUS_DataType dtype);

torch::Tensor convert_janus_tensor_to_torch(const JANUS_Tensor& janus_tensor);

janus::MMDataItem convert_janus_mm_item_to_internal(
    const JANUS_MM_Item& janus_item);

bool convert_janus_mm_data_to_internal(const JANUS_MM_Data* mm_data,
                                      janus::MMData& internal_mm_data);
}  // namespace helper
}  // namespace janus