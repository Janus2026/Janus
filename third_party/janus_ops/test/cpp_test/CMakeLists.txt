cmake_minimum_required(VERSION 3.14)


if(DEFINED ENV{VCPKG_ROOT})
  set(VCPKG_ROOT $ENV{VCPKG_ROOT})
else()
  message(FATAL_ERROR "VCPKG_ROOT is not set")
endif()
set(CMAKE_PREFIX_PATH "${VCPKG_ROOT}/installed/x64-linux")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

project(ACLNN_EXAMPLE LANGUAGES CXX)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Build acceleration options
option(ENABLE_UNITY "Enable Unity (Jumbo) builds" ON)
if(ENABLE_UNITY)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
    set(CMAKE_UNITY_BUILD ON)
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)
  else()
    message(STATUS "Unity build requested but requires CMake >= 3.16; skipping")
  endif()
endif()

option(ENABLE_PCH "Enable precompiled headers" ON)

find_program(LLD_PROGRAM ld.lld)
if(LLD_PROGRAM)
  add_link_options(-fuse-ld=lld)
endif()

# C++ standard and general settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#  Output executables to bin directory under source directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Build type compile options
add_compile_options(-Wall -fPIC)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0 -g)
else()
  add_compile_options(-O2 -DNDEBUG)
endif()

# Runtime dependencies (as needed)
set(ENV{LD_LIBRARY_PATH} "/usr/local/lib64/python3.11/site-packages/torch/lib:$ENV{LD_LIBRARY_PATH}")

# Enable testing support
enable_testing()

# Debug information
message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "GTest_DIR: ${GTest_DIR}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Output debug info to confirm GTest was found
if(TARGET GTest::gtest)
    message(STATUS "GTest::gtest target is available")
endif()
if(TARGET GTest::gtest_main)
    message(STATUS "GTest::gtest_main target is available")
endif()

# Ascend paths
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
set(OP_API_PATH "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/janus/op_api")

# Common include directories/compile options/link options/libraries
set(COMMON_INCLUDE_DIRS
  "${INCLUDE_BASE_DIR}"
  "${INCLUDE_BASE_DIR}/aclnn"
  "${INCLUDE_BASE_DIR}/aclnn/op_dev"
  "${INCLUDE_BASE_DIR}/platform"
  "${ASCEND_PATH}/x86_64-linux/include/exe_graph/runtime"
  "/usr/local/lib64/python3.11/site-packages/torch_npu/include"
  "/usr/local/lib64/python3.11/site-packages/torch/include"
  "/usr/local/lib64/python3.11/site-packages/torch/include/torch/csrc/api/include"
  "/usr/local/libtorch_npu/include"
  "${OP_API_PATH}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)


set(COMMON_COMPILE_OPTS
  -O2
  -std=c++17
  -D_GLIBCXX_USE_CXX11_ABI=1  # Compatible with Torch
  -Wall
)

set(COMMON_LINK_OPTS
  "-Wl,-rpath,${OP_API_PATH}/lib"
)

set(COMMON_LIBS
  "${ASCEND_PATH}/lib64/libascendcl.so"
  "${ASCEND_PATH}/lib64/libnnopbase.so"
  "${ASCEND_PATH}/lib64/libacl_op_compiler.so"
  "${ASCEND_PATH}/lib64/libascendalog.so"
  "${ASCEND_PATH}/lib64/libtiling_api.a"
  "${ASCEND_PATH}/lib64/libplatform.so"
  "${OP_API_PATH}/lib/libcust_opapi.so"
  "${ASCEND_PATH}/lib64/libopapi.so"
  "/usr/local/lib64/python3.11/site-packages/torch/lib/libc10.so"
  "/usr/local/lib64/python3.11/site-packages/torch/lib/libtorch.so"
  "/usr/local/lib64/python3.11/site-packages/torch/lib/libtorch_cpu.so"
  "/usr/local/libtorch_npu/lib/libtorch_npu.so"
  stdc++
)

# =============================================================================
# Common library configuration
# =============================================================================
# Package common configuration as INTERFACE library for reuse by executable targets
add_library(aclnn_common INTERFACE)
target_include_directories(aclnn_common INTERFACE ${COMMON_INCLUDE_DIRS})
target_compile_options(aclnn_common INTERFACE ${COMMON_COMPILE_OPTS})
target_link_options(aclnn_common INTERFACE ${COMMON_LINK_OPTS})
target_link_libraries(aclnn_common INTERFACE ${COMMON_LIBS})

# Optional precompiled headers (requires CMake >= 3.16)
if(ENABLE_PCH)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
    target_precompile_headers(aclnn_common INTERFACE
      <vector>
      <string>
      <memory>
      <map>
      <iostream>
    )
  else()
    message(STATUS "PCH requested but requires CMake >= 3.16; skipping")
  endif()
endif()

# GTest version tests
add_executable(pp_matmul_test
  pp_matmul_test.cpp
)
target_link_libraries(pp_matmul_test PRIVATE 
  aclnn_common
  GTest::gtest 
  GTest::gtest_main
)

# Group GEMM GTest version
add_executable(group_gemm_gtest
  group_gemm_test.cpp
)
target_link_libraries(group_gemm_gtest PRIVATE 
  aclnn_common
  GTest::gtest 
  GTest::gtest_main
)

# Beam Search GTest version
add_executable(beam_search_test
  beam_search_test.cpp
)
target_link_libraries(beam_search_test PRIVATE 
  aclnn_common
  GTest::gtest 
  GTest::gtest_main
)

# Group MLA GTest version
add_executable(multi_latent_attention_gtest
  multi_latent_attention_test.cpp
)
target_link_libraries(multi_latent_attention_gtest PRIVATE 
  aclnn_common
  GTest::gtest 
  GTest::gtest_main
)

# Convert KV Cache Format GTest version
add_executable(convert_kv_cache_format_test
  convert_kv_cache_format_test.cpp
)
target_link_libraries(convert_kv_cache_format_test PRIVATE 
  aclnn_common
  GTest::gtest 
  GTest::gtest_main
)

# Add tests
# add_test(AllTestsInBeamSearch beam_search_test)
add_test(AllTestsInPPMatmul pp_matmul_test)
add_test(AllTestsInGroupGemm group_gemm_gtest)
add_test(AllTestsInMultiLatentAttention multi_latent_attention_gtest)
add_test(AllTestsInConvertKvCacheFormat convert_kv_cache_format_test)

# GoogleTest automatic discovery
include(GoogleTest)
# gtest_discover_tests(beam_search_test)
gtest_discover_tests(pp_matmul_test)
gtest_discover_tests(group_gemm_gtest)
gtest_discover_tests(multi_latent_attention_gtest)
gtest_discover_tests(convert_kv_cache_format_test)
