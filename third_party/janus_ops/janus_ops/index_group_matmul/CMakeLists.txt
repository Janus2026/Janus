add_ops_compile_options(
	OP_NAME IndexGroupMatmul
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnn PRIVATE
        index_group_matmul_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        index_group_matmul_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        index_group_matmul_cpu.cpp
)

install(FILES index_group_matmul.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES index_group_matmul.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES index_group_matmul_mlt512.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

