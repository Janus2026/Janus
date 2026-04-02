include(cc_library)

file(GLOB_RECURSE HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.hpp"
)

file(GLOB_RECURSE SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

cc_library(
  NAME
    atb_speed_operations
  HDRS
    ${HEADER_FILES}
  SRCS
    ${SOURCE_FILES}
  DEPS
    atb
    gflags::gflags
    nlohmann_json::nlohmann_json
    opapi
    spdlog::spdlog
)