add_ops_compile_options(
	OP_NAME SparseFlashAttention
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -DASCENDC_DUMP=0
)

target_sources(op_host_aclnn PRIVATE
        sparse_flash_attention_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        sparse_flash_attention_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)


# opproto
target_sources(opsproto PRIVATE
        sparse_flash_attention_proto.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
install(FILES sparse_flash_attention.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES sparse_flash_attention_kernel_mla.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
