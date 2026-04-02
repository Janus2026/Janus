include(cc_library)

set(JANUS_ATB_LAYERS_PREBUILT_LIB
  "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/libjanus_atb_layers.a")

if(EXISTS "${JANUS_ATB_LAYERS_PREBUILT_LIB}")
  message(STATUS "Using prebuilt janus_atb_layers from ${JANUS_ATB_LAYERS_PREBUILT_LIB}")

  add_library(janus_atb_layers STATIC IMPORTED GLOBAL)
  set_target_properties(janus_atb_layers PROPERTIES
    IMPORTED_LOCATION "${JANUS_ATB_LAYERS_PREBUILT_LIB}"
  )
  target_include_directories(janus_atb_layers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/operations
    ${CMAKE_CURRENT_SOURCE_DIR}/pytorch
    ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/atb_torch/core/include
  )

else()
  message(STATUS "Building janus_atb_layers from source")

  file(GLOB_RECURSE JANUS_ATB_LAYERS_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/*.h"
  )

  file(GLOB_RECURSE JANUS_ATB_LAYERS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/*.cpp"
  )

  # This source file is intentionally included by linear.cpp directly.
  list(REMOVE_ITEM JANUS_ATB_LAYERS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/operations/aclnn/ops/aclnn_matmul_operation.cpp"
  )

  cc_library(
    NAME
      janus_atb_layers
    HDRS
      ${JANUS_ATB_LAYERS_HEADERS}
    SRCS
      ${JANUS_ATB_LAYERS_SOURCES}
    DEPS
      nlohmann_json::nlohmann_json
      gflags::gflags
      spdlog::spdlog
  )

  target_include_directories(janus_atb_layers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/operations
    ${CMAKE_CURRENT_SOURCE_DIR}/pytorch
    ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/atb_torch/core/include
  )

  if(DEFINED JANUS_ATB_LAYERS_STAGE_INCLUDE_DIR)
    target_include_directories(janus_atb_layers PUBLIC
      ${JANUS_ATB_LAYERS_STAGE_INCLUDE_DIR}
    )
  endif()

  target_include_directories(janus_atb_layers PRIVATE
    ${PROJECT_SOURCE_DIR}/../third_party/spdlog/include
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
    $ENV{NPU_HOME_PATH}/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{NPU_HOME_PATH}/opp/vendors/janus/op_api/include/
  )

  if(DEFINED VCPKG_TARGET_TRIPLET)
    target_include_directories(janus_atb_layers PRIVATE
      "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include"
    )
  endif()

  if(DEFINED JANUS_ATB_LAYERS_STAGE_LIB_DIR)
    set_target_properties(janus_atb_layers PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${JANUS_ATB_LAYERS_STAGE_LIB_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${JANUS_ATB_LAYERS_STAGE_LIB_DIR}"
    )
  endif()

endif()
