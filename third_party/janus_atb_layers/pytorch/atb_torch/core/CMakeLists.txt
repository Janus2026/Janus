include(cc_library)

file(GLOB_RECURSE SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

include_directories(
  $ENV{PYTHON_INCLUDE_PATH}
  $ENV{PYTORCH_INSTALL_PATH}/include
  $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
  $ENV{PYTORCH_NPU_INSTALL_PATH}/include
  ${PROJECT_SOURCE_DIR}/pytorch/atb_torch/core/include
)

cc_library(
  NAME
    atb_torch
  SRCS
    ${SOURCE_FILES}
  COPTS
    "-Wno-unused-function"
    "-Wno-unused-variable"
    "-Wno-unused-parameter"
    "-Wno-attributes"
  DEPS
    :atb_speed_core
    :atb_speed_operations
    atb
    c10
    nlohmann_json::nlohmann_json
    python3
    spdlog::spdlog
    torch
    torch_cpu
    torch_python
    torch_npu
)
