add_ops_compile_options(
	OP_NAME XAttention
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                #-Werror
                -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
                -I${CMAKE_SOURCE_DIR}/common
                # -I${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/experiment/runtime
)



target_sources(op_host_aclnn PRIVATE
        x_attention_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        x_attention_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        x_attention_cpu.cpp
)

install(FILES x_attention.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
