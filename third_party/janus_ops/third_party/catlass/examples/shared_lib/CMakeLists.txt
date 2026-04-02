# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# if aicore-arch is different, maybe crash, to find the solution
add_compile_definitions(TILING_KEY_VAR)
add_compile_options(--cce-aicore-arch=${CCE_AICORE_ARCH_BASE})
file(GLOB CATLASS_SHARED_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp)
set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

add_library(catlass_kernel SHARED ${CATLASS_SHARED_LIB_SRC})
add_library(catlass_kernel_static STATIC ${CATLASS_SHARED_LIB_SRC})
set_target_properties(catlass_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(catlass_kernel_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_custom_target(shared_lib)
add_dependencies(shared_lib catlass_kernel)
add_dependencies(shared_lib catlass_kernel_static)

install(TARGETS catlass_kernel catlass_kernel_static
    DESTINATION shared_lib/lib
    COMPONENT shared_lib)
install(DIRECTORY include DESTINATION shared_lib
    COMPONENT shared_lib)
