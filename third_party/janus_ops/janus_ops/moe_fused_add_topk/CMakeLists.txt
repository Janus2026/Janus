add_ops_compile_options(
	OP_NAME MoeFusedAddTopk
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnn PRIVATE
        moe_fused_add_topk_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        moe_fused_add_topk_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        moe_fused_add_topk_cpu.cpp
)

install(FILES moe_fused_add_topk.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES moe_fused_add_topk.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
