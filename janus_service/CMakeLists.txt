cmake_minimum_required(VERSION 3.26)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()

# TODO: FIXME later
option(USE_ASCEND "Enable NPU support" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CXX11_ABI "Use the new C++-11 ABI, which is not backwards compatible." ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CHECK_TORCH_ABI_SCRIPT
"
import torch
print(0 if not torch.compiled_with_cxx11_abi() else 1)
"
)

execute_process(
    COMMAND python -c "${CHECK_TORCH_ABI_SCRIPT}"
    OUTPUT_VARIABLE TORCH_CXX11_ABI_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE 
    RESULT_VARIABLE TORCH_ABI_CHECK_RESULT
)

if(NOT TORCH_ABI_CHECK_RESULT EQUAL 0)
    message(WARNING "Failed to detect PyTorch CXX11 ABI! "
            "Reason: ${TORCH_ABI_CHECK_RESULT}\n"
            "Fallback to _GLIBCXX_USE_CXX11_ABI=0 (default)")
    set(TORCH_CXX11_ABI_VALUE 0)
else()
    message(STATUS "Detected PyTorch CXX11 ABI: ${TORCH_CXX11_ABI_VALUE} "
            "(0=OFF, 1=ON)")
endif()

if(TORCH_CXX11_ABI_VALUE EQUAL 0)
    set(USE_CXX11_ABI OFF)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
else()
    set(USE_CXX11_ABI ON)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" 
      CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." 
      FORCE
  )
endif()

# Convert the bool variable to integer.
if(USE_CXX11_ABI)
  set(USE_CXX11_ABI 1)
  message(STATUS "Using the C++-11 ABI.")
else()
  set(USE_CXX11_ABI 0)
  message(STATUS "Using the pre C++-11 ABI.")
endif()

if (DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  file(MAKE_DIRECTORY $ENV{VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "Using VCPKG_DEFAULT_BINARY_CACHE: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
endif()

if (DEFINED ENV{DEPENDENCES_ROOT})
  message(STATUS "Using DEPENDENCES_ROOT: $ENV{DEPENDENCES_ROOT}")
endif()

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
else()
  include(FetchContent)
  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(VCPKG_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/vcpkg-src)
  else()
    set(VCPKG_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/vcpkg-src)
  endif()

  if (USE_CXX11_ABI)
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
      GIT_TAG "2024.02.14"
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
  else()
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
      GIT_TAG "ffc42e97c866ce9692f5c441394832b86548422c" # disable cxx11_abi
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
    message(STATUS "Using custom vcpkg with cxx11_abi disabled")
  endif()
  FetchContent_MakeAvailable(vcpkg)

  message(STATUS "Downloading and using vcpkg at ${vcpkg_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
      CACHE STRING "Vcpkg toolchain file")
endif()

project("janus_service" LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_language(Rust)
find_package(Rust REQUIRED)

# include custom cmake modules
include(static_analyzers)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
endif()

# find all dependencies from vcpkg
find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(leveldb CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if (USE_CXX11_ABI)
  # only use jemalloc if using the new C++-11 ABI
  find_package(Jemalloc)
  if(Jemalloc_FOUND)
    link_libraries(Jemalloc::jemalloc)
  endif()
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/janus_service)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/janus_service/include)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/etcd_cpp_apiv3)

set(VCPKG_ARCH x64)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(VCPKG_ARCH arm64)
endif()

# cpprestsdk
if (DEFINED ENV{VCPKG_ROOT} AND USE_ASCEND)
    include_directories("$ENV{VCPKG_ROOT}/packages/boost-locale_${VCPKG_ARCH}-linux/include")
    link_directories("$ENV{VCPKG_ROOT}/packages/boost-locale_${VCPKG_ARCH}-linux/lib")
    include_directories("$ENV{VCPKG_ROOT}/packages/boost-interprocess_${VCPKG_ARCH}-linux/include")
    link_directories("$ENV{VCPKG_ROOT}/packages/boost-interprocess_${VCPKG_ARCH}-linux/lib")
endif()
set(CPPREST_EXCLUDE_WEBSOCKETS ON CACHE BOOL "Exclude websockets functionality." FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation")




# etcd client v3
set(ETCD_CLIENT_OUTPUT_DIR ${CMAKE_BINARY_DIR}/third_party/etcd_cpp_apiv3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpprestsdk_service)
link_directories(${ETCD_CLIENT_OUTPUT_DIR}/src)

# brpc
if (DEFINED ENV{VCPKG_ROOT} AND USE_ASCEND)
    include_directories("$ENV{VCPKG_ROOT}/packages/zlib_${VCPKG_ARCH}-linux/include")
    link_directories("$ENV{VCPKG_ROOT}/packages/zlib_${VCPKG_ARCH}-linux/lib")
endif()
set(BRPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/third_party/brpc/output)
include_directories(${BRPC_OUTPUT_DIR}/include)
link_directories(${BRPC_OUTPUT_DIR}/lib)

# ---------------------------------------------------------------------------
# Third-party libraries (from shared janus/third_party/)
# ---------------------------------------------------------------------------
set(SHARED_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
set(TP_BUILD_DIR ${CMAKE_BINARY_DIR}/third_party)

include(cc_library)

set(BUILD_TESTS OFF CACHE BOOL "Disable cpprestsdk tests" FORCE)
set(BUILD_SAMPLES OFF CACHE BOOL "Disable cpprestsdk samples" FORCE)
add_subdirectory(${SHARED_THIRD_PARTY_DIR}/cpprestsdk_service ${TP_BUILD_DIR}/cpprestsdk)
set(CPPREST_INCLUDE_DIR ${SHARED_THIRD_PARTY_DIR}/cpprestsdk_service/Release/include)
set(CPPREST_LIB ${TP_BUILD_DIR}/cpprestsdk/Release/Binaries/libcpprest.a)
add_subdirectory(${SHARED_THIRD_PARTY_DIR}/etcd_cpp_apiv3 ${TP_BUILD_DIR}/etcd_cpp_apiv3)
add_subdirectory(${SHARED_THIRD_PARTY_DIR}/brpc_service ${TP_BUILD_DIR}/brpc)
add_subdirectory(${SHARED_THIRD_PARTY_DIR}/minja_service ${TP_BUILD_DIR}/minja)
add_subdirectory(${SHARED_THIRD_PARTY_DIR}/sentencepiece ${TP_BUILD_DIR}/sentencepiece)
add_subdirectory(${SHARED_THIRD_PARTY_DIR}/smhasher/src ${TP_BUILD_DIR}/smhasher)

# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
add_subdirectory(janus_service)

