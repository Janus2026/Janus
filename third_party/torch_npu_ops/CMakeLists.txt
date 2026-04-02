include(cc_library)

set(TORCH_NPU_KERNELS_HEADERS
  custom_functions_npu/atb_common.h
  custom_functions_npu/operation_cache_compute.h
  custom_functions_npu/operation_create.h
  custom_functions_npu/utils.h
  ops_npu/npu_ops.h
)

set(TORCH_NPU_KERNELS_SOURCES
  custom_functions_npu/atb_common.cpp
  custom_functions_npu/operation_cache_compute.cpp
  custom_functions_npu/utils.cpp
  ops_npu/paged_attention_atb.cpp
  ops_npu/reshape_and_cach_atb.cpp
  ops_npu/self_attention_atb.cpp
  ops_npu/rope_atb.cpp
)


set(TRITON_NPU_API_SRCS
  triton_npu/torch_api/npu_triton_rope_inplace.cpp
  triton_npu/torch_api/npu_triton_fused_gdn_gating.cpp
  triton_npu/torch_api/npu_triton_layernorm_fwd.cpp
  triton_npu/torch_api/npu_triton_causal_conv1d_update.cpp
  triton_npu/torch_api/npu_triton_recurrent_gated_delta_rule.cpp
)

cc_library(
  NAME
    triton_adapter
  HDRS
    triton_npu/kernel_registry.h
    triton_npu/operation_base.h
    triton_npu/args_builder.h
    triton_npu/operations.h
    triton_npu/torch_api/operation_factory.h
    triton_npu/torch_api/triton_ops_api.h
  SRCS
    triton_npu/kernel_registry.cpp
    ${TRITON_NPU_API_SRCS}
  DEPS
    "$ENV{ASCEND_HOME_PATH}/lib64/libruntime.so"
    ascendcl
    nlohmann_json::nlohmann_json
    glog::glog
)

if(DEFINED ENV{TRITON_BINARY_PATH} AND NOT "$ENV{TRITON_BINARY_PATH}" STREQUAL "")
  set(TRITON_BIN_PATH "$ENV{TRITON_BINARY_PATH}")
else()
  set(TRITON_BIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/triton_npu/binary")
endif()

message(STATUS "TRITON_BINARY_PATH = ${TRITON_BIN_PATH}")

target_compile_definitions(triton_adapter PRIVATE
  TRITON_BINARY_PATH="${TRITON_BIN_PATH}"
)

target_include_directories(triton_adapter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu
)

cc_library(
  NAME
    torch_npu_kernels
  HDRS
    ${TORCH_NPU_KERNELS_HEADERS}
  SRCS
    ${TORCH_NPU_KERNELS_SOURCES}
  DEPS
    glog::glog
    torch
    torch_npu
    atb_customize
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
file(GLOB _TRITON_NPUBIN_FILES "${TRITON_BIN_PATH}/*.npubin")
list(LENGTH _TRITON_NPUBIN_FILES _TRITON_NPUBIN_COUNT)
if(_TRITON_NPUBIN_COUNT EQUAL 0)
  message(STATUS "No pre-compiled triton .npubin found in ${TRITON_BIN_PATH}, will regenerate after build (requires NPU device).")
  add_custom_command(
    TARGET triton_adapter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
    TRITON_BINARY_PATH=${TRITON_BIN_PATH}
    ${Python3_EXECUTABLE} setup.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu
    COMMENT "Running triton_npu/setup.py to build/update Triton kernels."
  )
else()
  message(STATUS "Found ${_TRITON_NPUBIN_COUNT} pre-compiled triton .npubin in ${TRITON_BIN_PATH}, skipping regeneration.")
endif()

target_include_directories(torch_npu_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(torch_npu_kernels PRIVATE
  TORCH_NPU_KERNELS_VERSION="1.0.0"
)

add_subdirectory(triton_npu/test)
