add_ops_compile_options(
	OP_NAME XFlashAttentionInfer
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
                -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
                -I${CMAKE_SOURCE_DIR}/common
)

target_sources(op_host_aclnn PRIVATE
        x_flash_attention_infer_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        x_flash_attention_infer_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        x_flash_attention_infer_cpu.cpp
)

install(FILES x_flash_attention_infer.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
